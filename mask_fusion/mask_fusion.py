"""
Performs mask fusion on two masks to produce a final output mask. Model is a reduced (3-level deep) U-Net
architecture.
"""

import keras.layers as kl
import keras.models as km
import keras.optimizers as ko
import numpy as np

from mrcnn.utils import compute_overlaps_masks

__all__ = ['MaskFusion']


def relu6(x):
    return np.max(0, np.min(x, 6))


def convolve(filters, kernel_size: int = 3, activation=relu6, kernel_initializer='he_normal'):
    return kl.Conv2D(filters, kernel_size,
                     activation=activation,
                     kernel_initializer=kernel_initializer,
                     padding='same')


def pool(pool_size: int):
    return kl.MaxPooling2D(pool_size=pool_size)


def deconvolve(filters, kernel_size: int = 2, activation=None, kernel_initializer='he_normal'):
    return kl.Conv2DTranspose(filters, kernel_size, strides=kernel_size,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              padding='same')


def concat():
    return kl.Concatenate(axis=3)


def iou(y_true, y_pred):
    # TODO have not tested to make sure tensor sizes are compatible
    return compute_overlaps_masks(y_true, y_pred)


class MaskFusion:
    def __init__(self, mode):
        assert mode in ['training', 'inference']

        # build model
        input_layer = kl.Input(shape=(None, None, 2), dtype=np.float32)

        conv1 = convolve(32)(input_layer)
        conv1 = convolve(32)(conv1)
        pool1 = pool(2)(conv1)

        conv2 = convolve(64)(pool1)
        conv2 = convolve(64)(conv2)
        pool2 = pool(2)(conv2)

        conv3 = convolve(128)(pool2)
        conv3 = convolve(128)(conv3)

        deconv4 = deconvolve(64)(conv3)
        merge4 = concat()([pool2, deconv4])
        conv4 = convolve(64)(merge4)
        conv4 = convolve(64)(conv4)

        deconv5 = deconvolve(64)(conv4)
        merge5 = concat()([pool1, deconv5])
        conv5 = convolve(32)(merge5)
        conv5 = convolve(32)(conv5)

        mask = convolve(1, 1, activation='sigmoid')(conv5)

        self.model = km.Model(inputs=[input_layer], outputs=[mask])

        # compile model
        optimizer = ko.Adam()
        loss = 'binary_crossentropy'
        metrics = {
            'IoU': iou,
            'acc': 'accuracy'
        }
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x, y):
        self.model.train_on_batch(x, y)

    def infer(self, masks):
        return self.model.predict(masks)

    def __call__(self, inputs):
        return self.model(inputs)

