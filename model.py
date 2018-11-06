# TODO make integrated model here
import keras.models as km
import keras.layers as kl


class MultiSeg(object):
    
    def __init__(self, mode: str):
        if mode not in ['train', 'infer']:
            raise ValueError('MultiSeg mode must either be \'train\' or \'infer\'')
        
        self._mode = mode
        self._model = self._build_model()
        
    def _build_model(self) -> km.Model:
        
        prev_image = kl.Input((None, None, 3), )
        curr_image = input
        
        return km.Model(inputs=[], outputs=[])

    @property
    def mode(self):
        return self._mode
