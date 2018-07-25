############################################################
# Optical Flow (PWC Net)
############################################################

############################################################
# Image Segmentation (Mask RCNN)
############################################################

############################################################
# Mask Propagation
############################################################

############################################################
# Instance ID
############################################################

############################################################
# Mask Fusion
############################################################

############################################################
# MultiSeg Model
############################################################


# Boot Strap
# Image t is put through Image Segmentation, and Masks and ID info are stored
############################################################


# With Previous Mask

# Images t-1, t are put through optical flow, output is recieved as two channel image

# t-1 Masks and ID, as well as output from optical flow are put through Mask Propogation to
#       make predictive masks for all previous IDs

# In parallel, Image t is put through Image Segmentation Masks and ID info are stored

# The predictive masks, previous ID and featuremap, and the masks for the current image are
#       run through Instance ID to determine new masks ID

# If there are masks with no matching ID (t-1) new mask is used
# If there are masks with no matching ID (t) predictive mask is used in place
# If there are matching masks, both the new mask and predictive mask will be
#       fused through the Mask Fusion module where a refined mask will be created

# All masks will be compiled into a mask layer and placed on a frame in the video
import keras.layers as kl
import keras.models as km
import numpy as np

from image_seg.model import MaskRCNN
from mask_fusion.mask_fusion import MaskFusion
from mask_prop.mask_propagation import MaskPropagation
from opt_flow.pwc_net_wrapper import PWCNetWrapper


class MultiSegNet:

    def __init__(self, mrcnn_config, log_dir='./logs'):
        self.mrcnn_config = mrcnn_config
        self.log_dir = log_dir
        self._build_model()

    def _build_model(self):
        # OPTICAL FLOW: create optical flow model for use later
        self._optical_flow_model = PWCNetWrapper()

        #

        # set up inputs
        prev_image = kl.Input((None, None, 3), dtype=np.float32)
        curr_image = kl.Input((None, None, 3), dtype=np.float32)
        flow_field = kl.Input((None, None, 2), dtype=np.float32)

        # retrieve list of masks to propagate forward ??

        # MASK PROPAGATION: take all past masks and move them into the present
        propagated_masks = MaskPropagation()()

        # IMAGE SEGMENTATION: generate masks for the current image
        segmented_masks = MaskRCNN('inference', self.mrcnn_config, self.log_dir)

        # INSTANCE IDENTIFICATION: identify masks and make decisions on mask matching

        # MASK FUSION: fuse two competing masks into a final mask output
        final_mask = MaskFusion()()

        self._model = km.Model(inputs=[prev_image, curr_image, flow_field], outputs=[final_mask])

    def predict(self, input_images):
        """
        :param input_images: should be a 2-element list [prev_image, curr_image] where each element is [batch, h, w, 3]
        """
        # TODO handle special case of first frame
        assert input_images[0].shape[0] == input_images[1].shape[0]

        flow_field = self._optical_flow_model.infer_flow_field(*input_images)

        return self._model.predict(input_images.append(flow_field), batch_size=input_images[0].shape[0])

    def __call__(self, *args):
        return self._model(*args)
