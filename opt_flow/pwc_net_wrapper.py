import cv2
import torch
import numpy as np
from math import ceil
from .models import *

__all__ = ['PWCNetWrapper']


class PWCNetWrapper:
    def __init__(self, model_pathname='./pwc_net.pth.tar'):
        """
        :param model_pathname: path to the file for pretrained weights
        """
        # create model
        self.net = pwc_dc_net(model_pathname)
        self.net = self.net.cuda()
        self.net.eval()
        
    def infer_flow_field(self, img1, img2, divisor: float = 64.):
        # TODO adapt for larger batch sizes
        # TODO test speed differences between multiple runs or larger batches
        """
        :param img1: first image(s) as np array [h, w, 3] or [batch, h, w, 3]
        :param img2: second image(s) as np array [h, w, 3] or [batch, h, w, 3]
        :param divisor: must be divisible by 64
        :return: flow field between images as np array [h, w, 2]
        """
        assert int(divisor) % 64 == 0

        img_all = []
        if type(img1) == list:
            img_all.extend(img1)
        else:
            img_all.append(img1)
        if type(img2) == list:
            img_all.extend(img2)
        else:
            img_all.append(img2)

        # rescale the image size to be multiples of 64
        H, W = img_all[0].shape[0], img_all[0].shape[1]
        H_, W_ = int(ceil(H/divisor) * divisor), int(ceil(W/divisor) * divisor)

        print(H_, W_)

        for i in range(len(img_all)):
            img_all[i] = cv2.resize(img_all[i], (W_, H_))

        for _i, _inputs in enumerate(img_all):
            img_all[_i] = img_all[_i][:, :, ::-1]
            img_all[_i] = 1.0 * img_all[_i]/255.0

            img_all[_i] = np.transpose(img_all[_i], (2, 0, 1))
            img_all[_i] = torch.from_numpy(img_all[_i])
            img_all[_i] = img_all[_i].expand(1, img_all[_i].size()[0], img_all[_i].size()[1], img_all[_i].size()[2])
            img_all[_i] = img_all[_i].float()

        with torch.no_grad():
            img_all = torch.autograd.Variable(torch.cat(img_all, 1).cuda())
        
        flo = self.net(img_all)
        flo = flo[0] * 20.0
        flo = flo.cpu().data.numpy()
        
        # scale the flow back to the input size 
        flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2)
        u_ = cv2.resize(flo[:, :, 0], (W, H))
        v_ = cv2.resize(flo[:, :, 1], (W, H))
        u_ *= H / float(H_)
        v_ *= W / float(W_)
        flo = np.dstack((u_, v_))

        return flo
