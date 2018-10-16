from __future__ import absolute_import, division, print_function

from abc import ABC, abstractmethod
from copy import deepcopy
from opt_flow.model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS

# declare for import *
__all__ = ['OpticalFlowNetwork', 'TensorFlowPWCNet']


class OpticalFlowNetwork(ABC):
    """
    Optical flow class (wrapper) that represents a general optical flow predictor.
    """

    @abstractmethod
    def infer_from_image_pair(self, img1, img2):
        """
        Infers a flow field from two images (as separate arguments).
        :param img1: previous image [h, w, 3]
        :param img2: current image [h, w, 3]
        :return: flow field between images [h, w, 2]
        """
        pass

    @abstractmethod
    def infer_from_image_stack(self, imgs):
        """
        Infers a flow field from two images (stacked as a tensor).
        :param imgs: previous image, current image [h, w, 6]
        :return: flow field between images [h, w, 2]
        """
        pass

    @abstractmethod
    def __call__(self, input_tensors):
        """
        If a tensorflow model, enables functional combining of networks.
        :param input_tensors: tensor or list of tf tensors
        :return: output tensor
        """
        pass


class TensorFlowPWCNet(OpticalFlowNetwork):

    def __init__(self, image_size: tuple,
                 model_pathname='./opt_flow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000',
                 verbose=False):
        gpu_devices = ['/device:GPU:0']
        controller = '/device:GPU:0'

        nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
        nn_opts['verbose'] = verbose
        nn_opts['ckpt_path'] = model_pathname
        nn_opts['batch_size'] = 1
        nn_opts['gpu_devices'] = gpu_devices
        nn_opts['controller'] = controller

        # PWC-Net-large model in quarter-resolution mode:
        # - 6 level pyramid
        # - upsampling of level 2 by 4 in each dimension --> final flow prediction
        nn_opts['use_dense_cx'] = True
        nn_opts['use_res_cx'] = True
        nn_opts['pyr_lvls'] = 6
        nn_opts['flow_pred_lvl'] = 2

        # cropping of output images back to original size
        nn_opts['adapt_info'] = (1, image_size[0], image_size[1], 2)

        self.nn = ModelPWCNet(mode='test', options=nn_opts)

        if verbose:
            self.nn.print_config()

    def infer_from_image_pair(self, img1, img2):
        return self.nn.predict_from_img_pairs([(img1, img2)], batch_size=1, verbose=False)[0]

    def infer_from_image_stack(self, imgs):
        return self.infer_from_image_pair(imgs[..., :3], imgs[..., 3:])

    @property
    def graph(self):
        return self.nn.graph

    def __call__(self, *args, **kwargs):
        raise NotImplemented("not yet implemented for tf pwcnet")

