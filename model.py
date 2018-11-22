"""
MultiSeg Model
"""

from keras.backend import tf
import keras.layers as kl
import keras.models as km
import numpy as np
from typing import List, Iterable, Optional

import image_seg.model as imgseg
import mask_refine.mask_refine as mr
import opt_flow.opt_flow as of


class MultiSeg(object):
    
    def __init__(self, mode: str, image_size: Iterable[2], mrcnn_config, log_dir='./logs/'):
        if mode not in ['training', 'inference']:
            raise ValueError('MultiSeg mode must either be \'training\' or \'inference\'')
        
        self._mode = mode
        self.image_size = image_size
        
        if mode == 'training':
            self._model = self._build_model(image_size, mrcnn_config, log_dir)
        else:
            self._build_model(mrcnn_config, log_dir)
        
    def _build_model(self, image_size, mrcnn_config, log_dir: str) -> Optional[km.Model]:
        
        if self.mode == 'training':
            prev_image = kl.Input((None, None, 3), )
            curr_image = kl.Input((None, None, 3), )
            
            masks = imgseg.MaskRCNN(mode=self._mode)
            masks.keras_model()
    
            model = Model([input_image, input_image_meta, input_anchors],
                          [detections, mrcnn_class, mrcnn_bbox,
                           mrcnn_mask, roi_features, rpn_rois, rpn_class, rpn_bbox],
                          name='mask_rcnn')
            
            flow_field = of.TensorFlowPWCNet(image_size)
            
            return model
        else:  # in inference mode
            self.optical_flow = of.TensorFlowPWCNet(image_size)
            self.image_seg = imgseg.MaskRCNN(mode=self.mode, config=mrcnn_config, model_dir=log_dir)
            self.mask_refine = mr.MaskRefineSubnet()

    def predict_on_single_input(self, prev_image: np.ndarray, curr_image: np.ndarray) -> np.ndarray:
        if self.mode != 'inference':
            raise ValueError('create the model in inference mode')
        if prev_image.shape != curr_image.shape:
            raise ValueError('images must be the same size')
        
        flow_field = self.optical_flow.infer_from_image_pair(prev_image, curr_image)
        mrcnn_output = self.image_seg.detect([curr_image])[0]
        coarse_masks = mrcnn_output['masks']
        
        
        return refined_masks
    
    def predict(self, prev_imgs: List[np.ndarray], curr_imgs: List[np.ndarray]) -> List[np.ndarray]:
        pass

    @property
    def mode(self):
        return self._mode
