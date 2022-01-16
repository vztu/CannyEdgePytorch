import math
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=True):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class Net(nn.Module):
    def __init__(self, threshold1=2/255., threshold2=12/255., use_cuda=False):
        super(Net, self).__init__()

        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))
        self.median_filter = MedianPool2d(kernel_size=3)


    def rgb_to_ycbcr(self, image: torch.Tensor) -> torch.Tensor:
        r"""Convert an RGB image to YCbCr.

        Args:
            image (torch.Tensor): RGB Image to be converted to YCbCr.

        Returns:
            torch.Tensor: YCbCr version of the image.
        """

        if not torch.is_tensor(image):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(
                type(image)))

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                            .format(image.shape))

        r: torch.Tensor = image[..., 0, :, :]
        g: torch.Tensor = image[..., 1, :, :]
        b: torch.Tensor = image[..., 2, :, :]

        delta = .5
        y: torch.Tensor = .299 * r + .587 * g + .114 * b
        cb: torch.Tensor = (b - y) * .564 + delta
        cr: torch.Tensor = (r - y) * .713 + delta
        return torch.clip(torch.stack((y, cb, cr), -3), 0, 1.)


    def forward(self, img):
        # img_r = img[:,0:1]
        # img_g = img[:,1:2]
        # img_b = img[:,2:3]
        # print(img.shape)
        img_yuv = self.rgb_to_ycbcr(img)
        img_y = img_yuv.clone()[:, 0:1, :, :]

        # blur_horizontal = self.gaussian_filter_horizontal(img_r)
        # blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        # blur_horizontal = self.gaussian_filter_horizontal(img_g)
        # blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        # blur_horizontal = self.gaussian_filter_horizontal(img_b)
        # blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_y)
        blurred_img = self.gaussian_filter_vertical(blur_horizontal)

        # blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        # blurred_img = torch.stack([torch.squeeze(blurred_img)])

        # grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        # grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        # grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        # grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        # grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        # grad_y_b = self.sobel_filter_vertical(blurred_img_b)
        grad_x = self.sobel_filter_horizontal(blurred_img)
        grad_y = self.sobel_filter_vertical(blurred_img)

        # COMPUTE THICK EDGES

        # grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        # grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        # grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        grad_orientation = (torch.atan2(grad_y, grad_x) * (180.0/3.14159))
        grad_orientation += 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)])
        if self.use_cuda:
            pixel_range = torch.cuda.FloatTensor([range(pixel_count)])

        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1,height,width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,height,width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # THRESHOLD

        thresholded = thin_edges.clone()
        # thresholded[thin_edges<self.threshold1] = 0.0
        # thresholded[thin_edges>self.threshold2] = 0.0
        # thresholded[thin_edges<self.threshold1] = 0.0


        # Threshold on Grad_mag to get banding areas
        flat_pixels = torch.zeros_like(grad_mag)
        flat_pixels[grad_mag<self.threshold1] = 1
        flat_pixels[grad_mag>=self.threshold1] = 0
        flat_pixels = self.median_filter(flat_pixels)

        text_pixels = torch.zeros_like(grad_mag)
        text_pixels[grad_mag>self.threshold2] = 1
        text_pixels[grad_mag<=self.threshold2] = 0
        text_pixels = self.median_filter(text_pixels)

        early_threshold = grad_mag.clone()
        bband_mask = ~text_pixels.bool()
        early_threshold = early_threshold * bband_mask.to(dtype=torch.float)

        # threshold thin edges
        thresholded = thresholded * bband_mask.to(dtype=torch.float)

        bband_score = early_threshold.mean([2, 3], keepdim=False)
        # print(bband_score)
        # print(bband_score.shape)

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold, bband_score


if __name__ == '__main__':
    Net()
