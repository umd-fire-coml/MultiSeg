"""
Top-level objects for the mask propagation module.

See mask_propagation_training_davis.py for details and an example use.
"""
import cv2
from datetime import datetime
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.backend import tf
from keras.losses import K
import matplotlib.pyplot as plt
from skimage import io

__all__ = ['pad_image', 'plot_prediction', 'MaskPropagation']


def pad_image(image):
    """
    Pads images to be multiples of 8. (Do before feeding into U-Net.)
    :param: image to pad
    """
    # for davis, optical flow output always maps (480, 854) -> (480, 864)
    # for UNet, both dimensions must be a multiple of 8
    return cv2.copyMakeBorder(image, 0, 0, 5, 5, cv2.BORDER_CONSTANT, value=0)


def plot_prediction(frame_pair, pred_mask):
    """
    :param frame_pair:
    :param pred_mask:
    :return:
    """
    fig, axes = plt.subplots(4, 2)
    fig.set_size_inches(32, 16)

    img_prev = pad_image(io.imread(frame_pair[0]))
    img_curr = pad_image(io.imread(frame_pair[1]))
    mask_prev = pad_image(io.imread(frame_pair[2]))
    mask_curr = pad_image(io.imread(frame_pair[3]))

    axes[0][0].set_title("Image Prev")
    axes[0][1].set_title("Image Curr")
    axes[1][0].set_title("Mask Prev")
    axes[1][1].set_title("Mask Curr")
    axes[2][0].set_title("In Prev not in Curr")
    axes[2][1].set_title("In Curr not in Prev")
    axes[3][0].set_title("Pred Mask Curr")
    axes[3][1].set_title("Difference Btwn Curr and Pred")

    axes[0][0].imshow(img_prev)
    axes[0][1].imshow(img_curr)
    axes[1][0].imshow(mask_prev)
    axes[1][1].imshow(mask_curr)
    axes[2][0].imshow((mask_prev == 255) & (mask_curr != 255))
    axes[2][1].imshow((mask_prev != 255) & (mask_curr == 255))
    axes[3][0].imshow(pred_mask)
    axes[3][1].imshow((pred_mask == 255) & (mask_curr != 255))


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Defines a binary focal loss for contrastive mask loss.
    :param gamma:
    :param alpha:
    :return: focal loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


class MaskPropagation:
    def __init__(self):
        self._build_model()
        self.load_weights()

    def _build_model(self, deconv_act=None):
        """
        Builds the U-Net for the mask propagation network, 5 levels deep.
        :param deconv_act: activation for the deconvolutions (transposed convolutions)

        Adapted by Shivam and Derek from https://github.com/ShawDa/unet-rgb/blob/master/unet.py. Adaptations
        include a binary focal loss, transposed convolutions, and varied activations.
        """
        inputs = Input((None, None, 3))

        # block 1 (down-1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # block 2 (down-2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # block 3 (down-3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # block 4 (down-4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # block 5 (down-5)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        # block 6 (up-4)
        up6 = Conv2DTranspose(1024, (2, 2), strides=(2, 2),
                              activation=deconv_act, name='up6_upconv')(drop5)
        merge6 = Concatenate(axis=3, name='merge6_concat')([drop4, up6])
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        # block 7 (up-3)
        up7 = Conv2DTranspose(512, (2, 2), strides=(2, 2),
                              activation=deconv_act, name='up7_upconv')(conv6)
        merge7 = Concatenate(axis=3, name='merge7_concat')([conv3, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        # block 8 (up-2)
        up8 = Conv2DTranspose(256, (2, 2), strides=(2, 2),
                              activation=deconv_act, name='up8_upconv')(conv7)
        merge8 = Concatenate(axis=3, name='merge8_concat')([conv2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        # block 9 (up-1)
        up9 = Conv2DTranspose(128, (2, 2), strides=(2, 2),
                              activation=deconv_act, name='up9_upconv')(conv8)
        merge9 = Concatenate(axis=3, name='merge9_concat')([conv1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        # block 10 (final outputs)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        # compile model
        optimizer = Adam(lr=1e-4)
        loss = binary_focal_loss()  # 'binary_crossentropy'
        metrics = {
            'acc': 'accuracy'
        }
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model

    def load_weights(self, weights_path='./mask_prop/davis_unet_weights.h5'):
        self.model.load_weights(weights_path)

    def train(self, train_generator, val_generator, epochs=30, steps_per_epoch=500, val_steps_per_epoch=100):
        history_file = "logs/NEW_VAL_FL_history_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"

        callbacks = [
            TensorBoard(
                log_dir="logs",
                histogram_freq=0,
                write_graph=True,
                write_images=False
            ),
            ModelCheckpoint(
                "logs/NEW_VAL_FL_davis_unet_weights__{epoch:02d}__{val_loss:.2f}.h5",
                verbose=0, save_weights_only=True
            ),
            CSVLogger(history_file)
        ]

        history = self.model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=val_steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks
        )

        return history

    def predict(self, inputs, **kwargs):
        return self.model.predict(inputs, kwargs)

