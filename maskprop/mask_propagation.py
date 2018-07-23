"""
Top-level objects for the mask propagation module.
"""

from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.backend import tf
from keras.losses import K

from pwc_net.pytorch.pwc_net_wrapper import PWCNetWrapper

__all__ = ['MaskPropagationModule', 'MaskPropagationNetwork']


class MaskPropagationModule:
    def __init__(self, model_pathname=None):
        self._optical_flow_model = PWCNetWrapper() if model_pathname is None else PWCNetWrapper(model_pathname)
        self._mask_propagation_model = MaskPropagationNetwork()

    def infer_mask(self, img1, img2, masks):
        # obtain full flow field
        flow = self._optical_flow_model.infer_flow_field(img1, img2)

        # for each mask?, propagate to find current mask
        # TODO implement multi-mask propagation (prob with TimeDistributed layer)

        return flow  # temporary


class MaskPropagationNetwork:
    def __init__(self):
        self._build_model()

    @staticmethod
    def binary_focal_loss(gamma=2., alpha=.25):
        """
        Defines a binary focal loss for contrastive mask loss.
        :param gamma:
        :param alpha:
        :return: focal loss function
        """
        def focal_loss_fixed(y_true, y_pred):
            # bce = binary_crossentropy(y_true, y_pred) # binary cross entropy

            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
                (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

        return focal_loss_fixed

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

        model = Model(input=[inputs], output=[conv10])

        # compile model
        optimizer = Adam(lr=1e-4)
        loss = MaskPropagationNetwork.binary_focal_loss()  # 'binary_crossentropy'
        metrics = {
            'acc': 'accuracy'
        }
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model



'''
class MaskPropagationOld(object):

    def __init__(self, mode, config, pwc_net_weights_path, model_dir='./logs', debugging=False, isolated=False,
                 optimizer=tf.train.AdadeltaOptimizer(), loss_function=tf.losses.sigmoid_cross_entropy):
        """
        Creates and builds the mask propagation network.
        :param mode: either 'training' or 'inference'
        :param config: not used atm
        :param pwc_net_weights_path: path to the weights for pwc-net
        :param model_dir: directory to save/load logs and model checkpoints
        :param debugging: whether to include extra print operations
        :param isolated: whether this is the only network running
        :param optimizer: tf optimizer object, e.g. tf.train.AdadeltaOptimizer(), tf.train.AdamOptimizer()
        :param loss_function: tf function that computes the loss between the predicted and gt masks
        """
        self.name = 'maskprop'
        self.mode = mode
        self.config = config
        self.weights_path = pwc_net_weights_path
        self.model_dir = model_dir
        self.debugging = debugging

        self.saver = tf.train.Saver()

        self.sess = tf.Session(tf.ConfigProto()) if isolated else K.get_session()

        self.optimizer = optimizer
        self.loss_function = loss_function

        assert mode in ['training', 'inference']

        self._build()

    def _build(self):
        """
        Builds the computation graph for the mask propagation network.
        """

        # set up the input images for the optical flow pwc-net
        with tf.variable_scope(self.name):
            # set up image and inputs
            self.prev_images = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='prev_image')
            self.curr_images = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='curr_image')

            if self.mode == 'training':
                self.gt_masks = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='gt_masks')

            # scale images by 255
            prev = tf.divide(self.prev_images, 255)
            curr = tf.divide(self.curr_images, 255)

        # feed images into PWC-Net to get optical flow field
        x, _, _ = PWCNet()(prev, curr)

        # prepare input optical flow and input mask for the u-net
        with tf.variable_scope(self.name):
            self.flow_field = tf.image.resize_bilinear(x, tf.shape(prev)[1:3], name='flow_field')
            self.prev_masks = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='prev_masks')

            x = tf.concat([self.prev_masks, self.flow_field], axis=3, name='unet_inputs')

        # build the u-net and get the final propagated mask
        x = self._build_unet(x)
        self.propagated_masks = x

        # build training end of network
        with tf.variable_scope(self.name):
            if self.mode == 'training':
                self.loss = self.loss_function(self.gt_masks, self.propagated_masks)
                trainables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='unet')
                self.optimizable = self.optimizer.minimize(self.loss, var_list=trainables)

        # load weights for optical flow model from disk
        tf.global_variables_initializer().run(session=self.sess)

        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pwcnet'))

        saver.restore(self.sess, self.weights_path)

    def _build_unet(self, x, conv_act=tf.nn.relu6, deconv_act=None):
        """
        Builds the mask propagation network proper (based on the u-Net architecture).
        :param x: input tensor of the mask and flow field concatenated [batch, w, h, 1+2]
        :param conv_act: activation function for the convolution layers
        :param deconv_act: activation function for the transposed convolution layers
        :return: output tensor of the U-Net [batch, w, h, 1]
        As a side effect, two instance variables unet_left_wing and unet_right_wing are set with the final output tensors
        for each layer of the two halves of the U.
        """
        with tf.variable_scope('unet'):
            _input = x

            x = TL.conv2d(x, 64, (3, 3), activation=conv_act, name='L1_conv1')
            x = TL.conv2d(x, 64, (3, 3), activation=conv_act, name='L1_conv2')
            L1 = x

            x = TL.max_pooling2d(x, (2, 2), (2, 2), name='L2_pool')
            x = TL.conv2d(x, 128, (3, 3), activation=conv_act, name='L2_conv1')
            x = TL.conv2d(x, 128, (3, 3), activation=conv_act, name='L2_conv2')
            L2 = x

            x = TL.max_pooling2d(x, (2, 2), (2, 2), name='L3_pool')
            x = TL.conv2d(x, 256, (3, 3), activation=conv_act, name='L3_conv1')
            x = TL.conv2d(x, 256, (3, 3), activation=conv_act, name='L3_conv2')
            L3 = x

            x = TL.max_pooling2d(x, (2, 2), (2, 2), name='L4_pool')
            x = TL.conv2d(x, 512, (3, 3), activation=conv_act, name='L4_conv1')
            x = TL.conv2d(x, 512, (3, 3), activation=conv_act, name='L4_conv2')
            L4 = x

            x = TL.max_pooling2d(x, (2, 2), (2, 2), name='L5_pool')
            x = TL.conv2d(x, 1024, (3, 3), activation=conv_act, name='L5_conv1')
            x = TL.conv2d(x, 1024, (3, 3), activation=conv_act, name='L5_conv2')
            L5 = x

            x = TL.conv2d_transpose(x, 1024, (2, 2), strides=(2, 2), activation=deconv_act, name='P4_upconv')
            x = tf.concat([L4, tf.image.resize_images(x, tf.shape(L4)[1:3])],
                          axis=3, name='P4_concat')
            x = TL.conv2d(x, 512, (3, 3), activation=conv_act, name='P4_conv1')
            x = TL.conv2d(x, 512, (3, 3), activation=conv_act, name='P4_conv2')
            P4 = x

            x = TL.conv2d_transpose(x, 512, (2, 2), strides=(2, 2), activation=deconv_act, name='P3_upconv')
            x = tf.concat([L3, tf.image.resize_images(x, tf.shape(L3)[1:3])],
                          axis=3, name='P3_concat')
            x = TL.conv2d(x, 256, (3, 3), activation=conv_act, name='P3_conv1')
            x = TL.conv2d(x, 256, (3, 3), activation=conv_act, name='P3_conv2')
            P3 = x

            x = TL.conv2d_transpose(x, 256, (2, 2), strides=(2, 2), activation=deconv_act, name='P2_upconv')
            x = tf.concat([L2, tf.image.resize_images(x, tf.shape(L2)[1:3])],
                          axis=3, name='P2_concat')
            x = TL.conv2d(x, 128, (3, 3), activation=conv_act, name='P2_conv1')
            x = TL.conv2d(x, 128, (3, 3), activation=conv_act, name='P2_conv2')
            P2 = x

            x = TL.conv2d_transpose(x, 128, (2, 2), strides=(2, 2), activation=deconv_act, name='P1_upconv')
            x = tf.concat([L1, tf.image.resize_images(x, tf.shape(L1)[1:3])],
                          axis=3, name='P1_concat')
            x = TL.conv2d(x, 64, (3, 3), activation=conv_act, name='P1_conv1')
            x = TL.conv2d(x, 64, (3, 3), activation=conv_act, name='P1_conv2')
            P1 = x

            x = tf.image.resize_images(x, tf.shape(_input)[1:3])
            x = TL.conv2d(x, 1, (1, 1), activation=tf.sigmoid, name='P0_conv')

            self.unet_left_wing = [L1, L2, L3, L4, L5]
            self.unet_right_wing = [P4, P3, P2, P1]

            return x

    def train_batch(self, prev_images, curr_images, prev_masks, gt_masks):
        """
        Trains the mask propagation network on a single batch of inputs.
        :param prev_images: previous images at time t-1 [batch, w, h, 3]
        :param curr_images: current images at time t [batch, w, h, 3]
        :param prev_masks: the masks at time t [batch, w, h, 1]
        :param gt_masks: ground truth next masks at time t+1 [batch, w, h, 1]
        :return: batch loss of the predicted masks against the provided ground truths
        """
        assert self.mode == 'training'

        inputs = {self.prev_images: prev_images,
                  self.curr_images: curr_images,
                  self.prev_masks: prev_masks,
                  self.gt_masks: gt_masks}

        _, loss = self.sess.run([self.optimizable, self.loss], feed_dict=inputs)

        return loss

    #MAJOR WORK IN PROGRESS
    def train_multi_step(self, generator, steps, batch_size, output_types, output_shapes=None):
        """DO NOT USE. THIS IS VERY BAD IF YOU USE. DO NOT USE
        Trains the mask propagation network on multiple steps (batches). (Essentially an epoch.)
        :param train_dataset: Training Dataset object
        :param steps: Number of times to call the generator. (Number of steps in this "epoch".)
        :param batch_size: A tf.int64 scalar tf.Tensor, representing the number of consecutive elements of the generator to combine in a single batch.
        :param output_types: output_types: A nested structure of tf.DType objects corresponding to each component of an element yielded by generator.
        :param output_shapes: (Optional.) A nested structure of tf.TensorShape objects corresponding to each component of an element yielded by generator.
        :return: a list of batch losses of the predicted masks against the generated ground truths
        """
        assert self.mode == 'training'
        assert isinstance(generator, TensorflowDataGenerator)

        dataset = TD.Dataset().batch(batch_size).from_generator(generator,
                                                   output_types=output_types, 
                                                   output_shapes=output_shapes)
        _iter = dataset.make_initializable_iterator()
        element = _iter.get_next()
        self.sess.run(_iter.initializer)

        sliced_tensor = generator.slice_tensor(element)
        inputs = {self.prev_image: sliced_tensor['prev_image'],
                  self.curr_image: sliced_tensor['curr_image'],
                  self.prev_mask: sliced_tensor['prev_mask'],
                  self.gt_mask: sliced_tensor['gt_mask']}

        losses = [None] * steps

        for i in range(steps):
             _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=inputs)
             losses.append(loss)

        return losses

    def infer_flow_field(self, prev_image, curr_image):
        """
        Evaluates the model to get the flow field between the two images.
        :param prev_image: starting image for flow [w, h, 3]
        :param curr_image: ending image for flow [w, h, 3]
        :return: flow field for the images [batch, w, h, 2]
        """
        inputs = {self.prev_images: np.expand_dims(prev_image, 0),
                  self.curr_images: np.expand_dims(curr_image, 0)}

        mask = self.sess.run(self.flow_field, feed_dict=inputs)

        return mask

    def infer_propagated_mask(self, prev_image, curr_image, curr_mask):
        """
        Propagates the masks through the model to get the predicted mask.
        :param prev_image: starting image for flow at time t-1 [w, h, 3]
        :param curr_image: ending image for flow at time t [w, h, 3]
        :param curr_mask: the mask at time t [w, h, 1]
        :return: predicted/propagated mask at time t+1 [1, w, h, 1]
        """
        inputs = {self.prev_images: np.expand_dims(prev_image, 0),
                  self.curr_images: np.expand_dims(curr_image, 0),
                  self.prev_masks: np.expand_dims(curr_mask, 0)}

        mask = self.sess.run(self.propagated_masks, feed_dict=inputs)

        return mask

    def save_weights(self, filename):
        weights_pathname = os.path.join(self.model_dir, filename)

        # TODO implement saving all weights
        pass

    def load_weights(self, filename):
        weights_pathname = os.path.join(self.model_dir, filename)

        # TODO implement loading all weights
        pass


# test script
def test():
    root_dir = 'C:/Users/tmthy/Documents/prog/python3/coml/MaskTrack_RCNN/'
    model_path = os.path.join(root_dir, '/PWC-Net/model_3000epoch/model_3007.ckpt')
    mp = MaskPropagationOld('training', None, model_path, debugging=True, isolated=False)

    img1 = imageio.imread(os.path.join(root_dir, 'PWC-Net/test_images/frame1.jpg'))
    img2 = imageio.imread(os.path.join(root_dir, 'PWC-Net/test_images/frame2.jpg'))

    oflow = mp.infer_flow_field(img1, img2)
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(oflow[0, :, :, 0])
    plt.subplot(212)
    plt.imshow(oflow[0, :, :, 1])
    plt.show()

    mp.infer_propagated_mask(img1, img2, np.reshape(np.empty(img1.shape)[:, :, 0], (1080, 1349, 1)))

    mp.train_batch(img1, img2, np.empty((1080, 1349, 1)), np.empty((1080, 1349, 1)))
'''
