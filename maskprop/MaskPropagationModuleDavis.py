from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from keras.backend import tf


# adapted from https://github.com/ShawDa/unet-rgb/blob/master/unet.py
# changed by Derek
# added focal loss Tim & Shivam

def binary_focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        # bce = binary_crossentropy(y_true, y_pred) # binary cross entropy

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


def get_model(input_shape=(480, 864, 3)):
    deconv_act = None

    input_shape = input_shape
    inputs = Input(input_shape)

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

    model = Model(input=inputs, output=conv10)

    # compile model
    optimizer = Adam(lr=1e-4)
    loss = binary_focal_loss()  # 'binary_crossentropy'
    metrics = {
        'acc': 'accuracy'
    }
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model