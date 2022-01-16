# from scipy.misc import imread, imsave
import cv2
from matplotlib.pyplot import imsave
import torch
from scipy import io as sio
from torch.autograd import Variable
from net_bband import Net


def canny(raw_img, use_cuda=False):
    img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
    batch = torch.stack([img]).float()

    net = Net(threshold1=2/255, threshold2=6/255, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()
    net.eval()

    data = Variable(batch)
    if use_cuda:
        data = Variable(batch).cuda()

    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold, bband_score = net(data)

    imsave('gradient_magnitude.png', grad_mag.permute(0, 2, 3, 1).cpu().detach().numpy()[0,:,:,0])
    sio.savemat('gradient_magnitude.mat', mdict={'grad_mg': grad_mag.permute(0, 2, 3, 1).cpu().detach().numpy()[0,:,:,0]})

    imsave('thin_edges.png', thresholded.permute(0, 2, 3, 1).cpu().detach().numpy()[0,:,:,0])
    imsave('final.png', (thresholded.permute(0, 2, 3, 1).cpu().detach().numpy()[0,:,:,0] > 0.0).astype(float))
    imsave('early_thresholded.png', early_threshold.permute(0, 2, 3, 1).cpu().detach().numpy()[0,:,:,0])
    sio.savemat('early_thresholded.mat', mdict={'early_threshold': early_threshold.permute(0, 2, 3, 1).cpu().detach().numpy()[0,:,:,0]})

if __name__ == '__main__':
    image_name = '0001_fps25_0001.png'
    img = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB) / 255.

    # canny(img, use_cuda=False)
    canny(img, use_cuda=True)
