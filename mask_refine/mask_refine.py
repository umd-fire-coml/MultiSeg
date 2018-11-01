from datetime import datetime
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from os import path
import math
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K

from opt_flow.opt_flow import OpticalFlowNetwork

__all__ = ['MaskRefineSubnet', 'MaskRefineModule']


# functional layers
def _conv2d(filters, kernel=3, activation='relu', kernel_initializer='he_normal', **kwargs):
    return Conv2D(filters, kernel, activation=activation, padding='same',
                  kernel_initializer=kernel_initializer, **kwargs)


def _deconv2d(filters, activation=None, **kwargs):
    return Conv2DTranspose(filters, (2, 2), strides=(2, 2),
                           activation=activation, **kwargs)


def _maxpool2d(pool_size=(2, 2), **kwargs):
    return MaxPooling2D(pool_size=pool_size, **kwargs)


def _concat(axis=3, **kwargs):
    return Concatenate(axis=axis, **kwargs)


def _batchnorm():
    return BatchNormalization()


# utils & other
def rank(tensor):
    """
    Returns the rank (number of dimensions) of a tensor. Equivalent to
    len(tensor.shape).
    
    Args:
        tensor: tensor to check rank of

    Returns:
        rank (ndims) of the given tensor
    """
    
    return len(tensor.shape)


def pad64(tensor):
    """
    Pads an image with zeros to the next largest multiple of 64, centering the
    image as much as possible.
    
    Args:
        tensor: image to pad

    Returns:
        padded image with dimensions that are multiples of 64

    """
    # pads images with zeros to the next largest multiple of 64 (center fix)
    h_, w_ = math.ceil(tensor.shape[0] / 64) * 64, math.ceil(tensor.shape[1] / 64) * 64
    h_pad, w_pad = h_ - tensor.shape[0], w_ - tensor.shape[1]
    return np.pad(tensor, ((math.floor(h_pad / 2), math.ceil(h_pad / 2)),
                           (math.floor(w_pad / 2), math.ceil(w_pad / 2)),
                           (0, 0)), mode='constant')


def mask_binary_crossentropy_loss(y_true, y_pred):
    return tf.divide(tf.reduce_sum(K.binary_crossentropy(y_true, y_pred)),
                     tf.reduce_sum(y_true))


def compute_mask_binary_cross_entropy_loos(y_true, y_pred):
    binary_crossentropy = 0  # TODO finish implementing this in numpy
    
    return np.sum(binary_crossentropy) / np.sum(y_true)


# TODO check tensor data types
class MaskRefineSubnet:
    """
    Model for just the U-Net architecture within the Mask Refine Module. (Namely,
    this subnet does not handle running the optical flow network.)
    """

    def __init__(self, weights_path=None):
        self._build_model()

        if weights_path is not None:
            self.load_weights(weights_path)

    def _build_model(self, loss=mask_binary_crossentropy_loss):
        """
        Builds a U-Net for the mask refine network, 5 levels deep. Adapted from
        the original U-Net paper. Optimizes using adam schedule.
        
        Args:
            loss: loss function to use

        Batch normalization is applied after every convolutional layer (except
        the very last layer). No (spatial) dropout is used. We elect to use 2x2
        deconvolutions instead of 2x2 upsampling.
        """

        optimizer = Adam()

        inputs = Input((None, None, 6))

        # block 1 (down-1)
        conv1 = _conv2d(64, name='down1-conv1')(inputs)
        norm1 = _batchnorm()(conv1)
        conv1 = _conv2d(64, name='down1-conv2')(norm1)
        norm1 = _batchnorm()(conv1)
        pool1 = _maxpool2d()(norm1)

        # block 2 (down-2)
        conv2 = _conv2d(128, name='down2-conv1')(pool1)
        norm2 = _batchnorm()(conv2)
        conv2 = _conv2d(128, name='down2-conv2')(norm2)
        norm2 = _batchnorm()(conv2)
        pool2 = _maxpool2d()(norm2)

        # block 3 (down-3)
        conv3 = _conv2d(256, name='down3-conv1')(pool2)
        norm3 = _batchnorm()(conv3)
        conv3 = _conv2d(256, name='down3-conv2')(norm3)
        norm3 = _batchnorm()(conv3)
        pool3 = _maxpool2d()(norm3)

        # block 4 (down-4)
        conv4 = _conv2d(512, name='down4-conv1')(pool3)
        norm4 = _batchnorm()(conv4)
        conv4 = _conv2d(512, name='down4-conv2')(norm4)
        norm4 = _batchnorm()(conv4)
        pool4 = _maxpool2d()(norm4)

        # block 5 (5)
        conv5 = _conv2d(1024, name='down5-conv1')(pool4)
        norm5 = _batchnorm()(conv5)
        conv5 = _conv2d(1024, name='down5-conv2')(norm5)
        norm5 = _batchnorm()(conv5)

        # block 6 (up-4)
        up6 = _deconv2d(1024, name='up4-upconv')(norm5)
        norm6 = _batchnorm()(up6)

        merge6 = _concat()([norm4, norm6])
        conv6 = _conv2d(512, name='up4-conv1')(merge6)
        norm6 = _batchnorm()(conv6)
        conv6 = _conv2d(512, name='up4-conv2')(norm6)
        norm6 = _batchnorm()(conv6)

        # block 7 (up-3)
        up7 = _deconv2d(512, name='up3-upconv')(norm6)
        norm7 = _batchnorm()(up7)

        merge7 = _concat()([conv3, norm7])
        conv7 = _conv2d(256, name='up3-conv1')(merge7)
        norm7 = _batchnorm()(conv7)
        conv7 = _conv2d(256, name='up3-conv2')(norm7)
        norm7 = _batchnorm()(conv7)

        # block 8 (up-2)
        up8 = _deconv2d(256, name='up2-upconv')(norm7)
        norm8 = _batchnorm()(up8)

        merge8 = _concat()([conv2, norm8])
        conv8 = _conv2d(128, name='up2-conv1')(merge8)
        norm8 = _batchnorm()(conv8)
        conv8 = _conv2d(128, name='up2-conv2')(norm8)
        norm8 = _batchnorm()(conv8)

        # block 9 (up-1)
        up9 = _deconv2d(128, name='up1-upconv')(norm8)
        norm9 = _batchnorm()(up9)

        merge9 = _concat()([conv1, norm9])
        conv9 = _conv2d(64, name='up1-conv1')(merge9)
        norm9 = _batchnorm()(conv9)
        conv9 = _conv2d(64, name='up1-conv2')(norm9)
        norm9 = _batchnorm()(conv9)

        # block 10 (final outputs)
        conv10 = _conv2d(1, kernel=1, activation='sigmoid', name='final_conv')(norm9)

        model = Model(inputs=[inputs], outputs=[conv10])

        # compile model
        metrics = ['binary_accuracy', 'binary_crossentropy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self._model = model

    def load_weights(self, weights_path):
        """
        Load pretrained weights for the U-Net (in hdf5 format).
        
        Args:
            weights_path: path with filename of the weights binary
        """

        self._model.load_weights(weights_path)

    def train(self, train_generator, val_generator, epochs=30, steps_per_epoch=500, val_steps_per_epoch=1000):
        """
        Trains the U-Net using inputs and ground truth from the given generators.
        
        Args:
            train_generator: generate X, y input pairs for training
            val_generator: generate X, y input pairs for validation
            epochs: number of epochs to train
            steps_per_epoch: number of image pairs + masks per epoch for training
            val_steps_per_epoch: number of image pairs + mask per epoch for validation

        Returns: Keras history object
        """
        
        date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_directory = f'./logs/mr_training_{date_and_time}/'
        if not path.exists(log_directory):
            os.mkdir(log_directory)
        
        checkpoint_file = path.join(log_directory, 'davis_unet_weights__{epoch:02d}__{val_loss:.2f}.h5')
        history_file = path.join(log_directory, f'mr_history_{date_and_time}.csv')

        callbacks = [
            TensorBoard(
                log_dir=log_directory,
                histogram_freq=0,
                write_graph=True,
                write_images=False
            ),
            ModelCheckpoint(
                checkpoint_file,
                verbose=0, save_weights_only=True
            ),
            CSVLogger(history_file)
        ]

        history = self._model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=val_steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks
        )

        return history

    def predict(self, input_stack):
        """
        Run inference for a set of inputs (batch size must be 1).
        
        Args:
            input_stack: inputs to mask refine model (see below)

        Returns:
            refined mask of shape [1, h, w, 1]
        
        Input stack of shape [1, h, w, 6]
            IMAGE [1, h, w, 3]
            MASK  [1, h, w, 1]
            FLOW  [1, h, w, 2]
        These are all concatenated along axis=2 (3rd axis).
        """
        
        if rank(input_stack) != 4:
            raise ValueError(f'input stack must have rank 4 (provided: {rank(input_stack)})')

        return self._model.predict(input_stack, batch_size=1)

    @staticmethod
    def build_input_stack(image, mask, flow_field):
        """
        Builds an input stack tensor (ready for use in model training) with batch
        size of 1 from the image, mask, and flow field tensors.
        :param image: color image tensor of shape [h, w, 3]
        :param mask: mask tensor of shape [h, w, 1]
        :param flow_field: optical flow tensor of shape [h, w, 2]
        :return: input stack of shape [1, h, w, 6]
        """

        assert rank(image) == 3 and rank(mask) == 3 and rank(flow_field) == 3

        return np.expand_dims(np.concatenate((image, mask, flow_field), axis=2), axis=0)

    def __call__(self, *args):
        return self._model(*args)


class MaskRefineModule:
    """
    Model for the entire Mask Refine network: we don't handle the creation of
    the subnetworks here, just assembly and pipelining.
    """

    def __init__(self, optical_flow_model: OpticalFlowNetwork, mask_refine_subnet: MaskRefineSubnet):
        self.optical_flow_model = optical_flow_model
        self.mask_refine_subnet = mask_refine_subnet

    def train(self, train_generator, val_generator, **kwargs):
        def with_optical_flow(gen):
            while True:
                X, y = next(gen)

                assert rank(X) == 3 and rank(y) == 3

                # pad image and mask to multiples of 64
                X = pad64(X)
                y = pad64(y)

                flow_field = self.optical_flow_model.infer_from_image_stack(X[..., :6])
                Xnew = MaskRefineSubnet.build_input_stack(
                    X[..., 3:6],
                    np.expand_dims(X[..., 6], axis=2),
                    flow_field)

                ynew = np.expand_dims(y, axis=0)

                assert rank(Xnew) == 4 and rank(ynew) == 4

                yield Xnew, ynew

        return self.mask_refine_subnet.train(
            with_optical_flow(train_generator),
            with_optical_flow(val_generator),
            **kwargs
        )

    def refine_mask(self, input_stack):
        """
        Refines a coarse probability mask generated by the ImageSeg module into
        :param input_stack: previous image, current image, coarse_mask of shape [h, w, 7]
        :return: refined mask of shape [h, w, 1]

        input stack (concatenated along the 3rd axis (axis=2)):
        PREV IMAGE  [h, w, 3]
        CURR IMAGE  [h, w, 3]
        COARSE MASK [h, w, 1]
        """

        assert rank(input_stack) == 3

        input_stack = pad64(input_stack)  # TODO also make scale back to orig size

        flow_field = self.optical_flow_model.infer_from_image_stack(input_stack[..., :6])
        subnet_input_stack = MaskRefineSubnet.build_input_stack(input_stack[..., 3:6],
                                                                np.expand_dims(input_stack[..., 6], axis=2),
                                                                flow_field)

        assert rank(subnet_input_stack) == 4

        return self.mask_refine_subnet.predict(subnet_input_stack)

    def evaluate_mask(self, input_stack, gt_mask):
        """
        Refines a coarse probability mask generated by the ImageSeg module into
        :param input_stack: previous image, current image, coarse_mask of shape [h, w, 7]
        :param gt_mask: ground truth mask for the current image, of shape [h, w, 1]
        :return: list of metrics calculated for this mask

        input stack (concatenated along the 3rd axis (axis=2)):
        PREV IMAGE  [h, w, 3]
        CURR IMAGE  [h, w, 3]
        COARSE MASK [h, w, 1]
        """

        assert rank(input_stack) == 3
        assert rank(gt_mask) == 3

        input_stack = pad64(input_stack)
        gt_mask = np.expand_dims(pad64(gt_mask), axis=0)

        flow_field = self.optical_flow_model.infer_from_image_stack(input_stack[..., :6])
        subnet_input_stack = MaskRefineSubnet.build_input_stack(input_stack[..., 3:6],
                                                                np.expand_dims(input_stack[..., 6], axis=2),
                                                                flow_field)

        assert rank(subnet_input_stack) == 4
        assert rank(gt_mask) == 4

        return self.mask_refine_subnet._model.evaluate(subnet_input_stack, gt_mask)

    @property
    def metrics(self):
        return self.mask_refine_subnet._model.metrics_names

    @staticmethod
    def build_input_stack(prev_image, curr_image, coarse_mask):
        assert rank(prev_image) == 3 and rank(curr_image) == 3 and rank(coarse_mask) == 3

        return np.concatenate((prev_image, curr_image, coarse_mask), axis=2)
