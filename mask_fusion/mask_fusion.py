"""
Performs mask fusion on two masks to produce a final output mask. Model is a reduced (3-level deep) U-Net
architecture.
"""

from datetime import datetime
from keras.backend import tf
import keras.callbacks as kc
import keras.layers as kl
import keras.models as km
import keras.optimizers as ko
import keras.utils as ku
import numpy as np

__all__ = ['MaskFusion']


def convolve(filters, kernel_size: int = 3, activation='relu', kernel_initializer='he_normal'):
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


def iou(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.metrics.mean_iou(y_true, y_pred, 2)


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
        merge4 = concat()([conv2, deconv4])
        conv4 = convolve(64)(merge4)
        conv4 = convolve(64)(conv4)

        deconv5 = deconvolve(64)(conv4)
        merge5 = concat()([conv1, deconv5])
        conv5 = convolve(32)(merge5)
        conv5 = convolve(32)(conv5)

        mask = convolve(1, 1, activation='sigmoid')(conv5)

        self.model = km.Model(inputs=[input_layer], outputs=[mask])

        # compile model
        optimizer = ko.Adam()
        loss = 'binary_crossentropy'
        metrics = ['accuracy', 'sparse_categorical_accuracy']
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def train(self, train_generator, val_generator, **kwargs):
        history_file = "logs/mask_fusion_history_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"

        callbacks = [
            kc.TensorBoard(
                log_dir="logs",
                histogram_freq=0,
                write_graph=True,
                write_images=False
            ),
            kc.ModelCheckpoint(
                "logs/mask_fusion_weights__{epoch:02d}__{val_loss:.2f}.h5",
                verbose=0, save_weights_only=True
            ),
            kc.CSVLogger(history_file)
        ]

        return self.model.fit_generator(train_generator, validation_data=val_generator, callbacks=callbacks, **kwargs)

    def predict(self, masks, batch_size=1, **kwargs):
        return self.model.predict(masks, batch_size=batch_size, **kwargs)

    def __call__(self, *args):
        return self.model(*args)


class EmptyMaskGenerator(ku.Sequence):
    def __len__(self):
        """
        :return: number of batches
        """
        return 20

    def __getitem__(self, index: int):
        """"
        Generates one batch of data.
        """
        return np.empty((1, 512, 512, 2), dtype=np.float32), np.empty((1, 512, 512, 1), dtype=np.float32)

