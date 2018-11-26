from datetime import datetime
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from os import path
import math
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K

from typing import Union, Iterable

from opt_flow.opt_flow import OpticalFlowNetwork

__all__ = ['MaskRefineSubnet']


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


# functional blocks
def _down_block(input_layer, filters, n, convs=2):
    assert convs > 0
    
    conv = _conv2d(filters, name=f'down{n}-conv1')(input_layer)
    norm = _batchnorm()(conv)
    
    for i in range(2, convs + 1):
        conv = _conv2d(filters, name=f'down{n}-conv{i}')(norm)
        norm = _batchnorm()(conv)
    
    pool = _maxpool2d()(norm)
    
    return norm, pool


def _level_block(input_layer, filters, n, convs=2):
    assert convs > 0
    
    norm = input_layer
    
    for i in range(1, convs + 1):
        conv = _conv2d(filters, name=f'level{n}-conv{i}')(norm)
        norm = _batchnorm()(conv)
    
    return norm


def _up_block(input_layer, residual_layer, filters, n, convs=2):
    up = _deconv2d(2 * filters, name=f'up{n}-upconv')(input_layer)
    norm = _batchnorm()(up)
    
    merge = _concat()([residual_layer, norm])
    
    conv = _conv2d(filters, name=f'up{n}-conv1')(merge)
    norm = _batchnorm()(conv)
    
    for i in range(2, convs + 1):
        conv = _conv2d(filters, name=f'up{n}-conv{i}')(norm)
        norm = _batchnorm()(conv)
    
    return norm


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


def check_rank(*tensors: Union[Iterable[np.ndarray], np.ndarray], c_rank):
    for tensor in tensors:
        if rank(tensor) != c_rank:
            raise ValueError(f'input must have rank {c_rank} (provided: {rank(tensor)})')


def pad64(tensor, image_dims=(0, 1)):
    """
    Pads an image with zeros to the next largest multiple of 64, centering the
    image as much as possible.
    
    Args:
        tensor: image to pad

    Returns:
        padded image with dimensions that are multiples of 64

    TODO make this more extensible
    """
    # pads images with zeros to the next largest multiple of 64 (center fix)
    h_, w_ = math.ceil(tensor.shape[1] / 64) * 64, math.ceil(tensor.shape[2] / 64) * 64
    h_pad, w_pad = h_ - tensor.shape[1], w_ - tensor.shape[2]
    return np.pad(tensor, ((0, 0),
                           (math.floor(h_pad / 2), math.ceil(h_pad / 2)),
                           (math.floor(w_pad / 2), math.ceil(w_pad / 2)),
                           (0, 0)), mode='constant')


def mask_binary_crossentropy_loss(y_true, y_pred):
    return tf.divide(tf.reduce_sum(K.binary_crossentropy(y_true, y_pred)),
                     tf.reduce_sum(y_true))


def compute_mask_binary_cross_entropy_loss(y_true, y_pred):
    binary_crossentropy = -y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    
    return np.sum(binary_crossentropy) / np.sum(y_true)


# TODO check tensor data types (and ranges)
class MaskRefineSubnet:
    """
    Model for just the U-Net architecture within the Mask Refine Module. (Namely,
    this subnet does not handle running the optical flow network.)
    """

    def __init__(self, optical_flow_model: OpticalFlowNetwork):
        self._build_model()

        self.optical_flow_model = optical_flow_model

    def _build_model(self):
        """
        Builds a U-Net for the mask refine network, 5 levels deep. Adapted from
        the original U-Net paper. Optimizes using adam schedule.

        Batch normalization is applied after every convolutional layer (except
        the very last layer). No (spatial) dropout is used. We elect to use 2x2
        deconvolutions instead of 2x2 up-sampling.
        """
        
        input_image = Input((None, None, 3))
        input_masks = Input((None, None, 1))
        input_flow_field = Input((None, None, 2))
        
        inputs = _concat()([input_image, input_masks, input_flow_field])
        
        '''
        Layers are named as follows:
            * first part is the block name (either 'down' or 'up') followed by
              depth of the level
            * followed by a hyphen
            * last part is the type of layer, following by the occurrence of that
              layer in that level
        '''

        # block 1 (down-1)
        norm1, down_block1 = _down_block(inputs, 64, 1)

        # block 2 (down-2)
        norm2, down_block2 = _down_block(down_block1, 128, 2)

        # block 3 (down-3)
        norm3, down_block3 = _down_block(down_block2, 256, 3)

        # block 4 (down-4)
        norm4, down_block4 = _down_block(down_block3, 512, 4)

        # block 5 (5)
        level_block5 = _level_block(down_block4, 1024, 5)
        
        # block 6 (up-4)
        up_block4 = _up_block(level_block5, norm4, 512, 4)

        # block 7 (up-3)
        up_block3 = _up_block(up_block4, norm3, 256, 3)

        # block 8 (up-2)
        up_block2 = _up_block(up_block3, norm2, 128, 2)

        # block 9 (up-1)
        up_block1 = _up_block(up_block2, norm1, 64, 1)

        # block 10 (final outputs)
        final_conv = _conv2d(1, kernel=1, activation='sigmoid', name='final_conv')(up_block1)

        model = Model(inputs=[input_image, input_masks, input_flow_field], outputs=[final_conv])

        # compile model
        optimizer = Adam()
        loss = mask_binary_crossentropy_loss
        metrics = ['binary_accuracy', 'binary_crossentropy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self._model = model

    def load_weights(self, weights_path='./mask_refine/davis_unet_weights.h5'):
        """
        Load pre-trained weights for the U-Net (in hdf5 format).
        
        Args:
            weights_path: path with filename of the weights binary
        """
        if weights_path:
            self._model.load_weights(weights_path)

    def train(self, train_generator, val_generator, epochs=30, steps_per_epoch=500, val_steps_per_epoch=100):
        """
        Trains the U-Net using inputs and ground truth from the given generators.
        
        Args:
            train_generator: generate training inputs
            val_generator: generate input pairs for validation
            epochs: number of epochs to train
            steps_per_epoch: number of image pairs + masks per epoch for training
            val_steps_per_epoch: number of image pairs + mask per epoch for validation

        Returns: Keras history object
        
        inputs are the following in a tuple:
        previous image:     [1, h, w, 3]
        current image:      [1, h, w, 3]
        masks:              [n, h, w, 1]
        ground-truth masks: [n, h, w, 1]
        """

        # define a wrapper generator that applies optical flow to some of the
        # inputs and creates a new input stack and ground truth
        def with_optical_flow(gen):
            while True:
                prev_img, curr_img, mask_tensor, gt_tensor = next(gen)
        
                check_rank(prev_img, curr_img, mask_tensor, gt_tensor, c_rank=4)
        
                # pad everything to multiples of 64
                prev_img, curr_img, mask_tensor, gt_tensor = map(pad64, (prev_img, curr_img, mask_tensor, gt_tensor))
        
                # generate flow field and build new input stack
                img_stack = np.concatenate((prev_img, curr_img), axis=-1)
                flow_field = np.expand_dims(self.optical_flow_model.infer_from_image_stack(img_stack[0, ...]), axis=0)
                
                yield [curr_img, mask_tensor, flow_field], gt_tensor
        
        # create the log directory for this training session
        date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_directory = f'./logs/mr_training_{date_and_time}/'
        if not path.exists(log_directory):
            os.mkdir(log_directory)
        
        # create the file names for the training files for this session
        checkpoint_file = path.join(log_directory, 'davis_unet_weights__{epoch:02d}__{val_loss:.2f}.h5')
        history_file = path.join(log_directory, f'mr_history_{date_and_time}.csv')

        # training callbacks
        callbacks = [
            TensorBoard(
                log_dir=log_directory,
                histogram_freq=0,
                write_graph=True,
                write_images=False
            ),
            ReduceLROnPlateau(patience=5),
            ModelCheckpoint(
                checkpoint_file,
                verbose=0,
                save_weights_only=True
            ),
            CSVLogger(history_file)
        ]

        history = self._model.fit_generator(
            with_optical_flow(train_generator),
            validation_data=with_optical_flow(val_generator),
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks
        )

        return history

    def predict(self, *inputs):
        """
        Run inference for a set of inputs (batch size must be 1).
        
        Args:
            inputs: inputs to mask refine model (see below)

        Returns:
            refined mask of shape [1, h, w, 1]
        
        Inputs (in this order):
            IMAGE [1, h, w, 3]
            MASK  [1, h, w, 1]
            FLOW  [1, h, w, 2]
        """
        
        check_rank(*inputs, c_rank=4)

        return self._model.predict(inputs, batch_size=1)

    def evaluate(self, *inputs_and_outputs):
        check_rank(*inputs_and_outputs, c_rank=4)
        
        return self._model.evaluate(inputs_and_outputs[:-1], inputs_and_outputs[-1])

    @property
    def metrics(self):
        return self.metrics
    
    def __call__(self, *args):
        return self._model(*args)

