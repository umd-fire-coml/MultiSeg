"""
Performs mask fusion on two masks to produce a final output mask. Model is a reduced (3-level deep) U-Net
architecture.
"""

from datetime import datetime
import keras.callbacks as kc
import keras.layers as kl
import keras.models as km
import keras.optimizers as ko
import keras.utils as ku
import numpy as np

from mrcnn.utils import compute_overlaps_masks

__all__ = ['MaskFusion', 'DoubleMaskGenerator']


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

    def train(self, train_generator, **kwargs):
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

        return self.model.fit_generator(train_generator, callbacks=callbacks, **kwargs)

    def predict(self, masks, **kwargs):
        return self.model.predict(masks, kwargs)

    def __call__(self, inputs):
        return self.model(inputs)


class DoubleMaskGenerator(ku.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32, 32), n_channels=1, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))

        self.on_epoch_end()

    def __len__(self):
        """
        :return: number of batches
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index: int):
        """"
        Generates one batch of data.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self._generate_data(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        Updates the indices after each epoch (and sets them at the very beginning).
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_data(self, list_IDs_temp):
        """
        Generates all the data for a single batch based on the list of IDs provided.
        """
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, ), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = np.load('data/' + ID + '.npy')

        return X, y

