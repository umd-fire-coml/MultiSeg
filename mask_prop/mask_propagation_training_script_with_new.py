#!usr/bin/env python

# Imports
import skimage.io as io
from keras.layers import *
from keras.backend import tf

# Static GPU memory allocation for tensorflow (need some GPU for PyTorch Optical Flow)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Our own modules
from train.davis_wrapper_new import DavisDataset
from opt_flow.pwc_net_wrapper import PWCNetWrapper
from .mask_propagation import MaskPropagation

##########################################################################
#
# Load Dataset Wrapper, Optical Flow Model, and UNet
#
##########################################################################

dataset = DavisDataset("./mask_prop/DAVIS", "480p", val_videos=[
    "car-shadow", "breakdance", "camel", "scooter-black", "libby", "drift-straight"
])
optical_flow = PWCNetWrapper("./opt_flow/pwc_net.pth.tar")
model = MaskPropagation()

##########################################################################
#
# Data Preprocessing Methods
#
##########################################################################


def get_model_input(img_prev_p, img_curr_p, mask_prev_p, mask_curr_p):
    """
    Returns tensor that contains previous mask and optical flow, and also
    returns current mask as the ground truth value.
    """
    img_prev, img_curr = io.imread(img_prev_p), io.imread(img_curr_p)

    finalflow = optical_flow.infer_flow_field(img_prev, img_curr)
    finalflow_x, finalflow_y = finalflow[:, :, 0], finalflow[:, :, 1]
    finalflow[:, :, 0] = (finalflow_x - finalflow_x.mean()) / finalflow_x.std()
    finalflow[:, :, 1] = (finalflow_y - finalflow_y.mean()) / finalflow_y.std()

    mask_prev = io.imread(mask_prev_p) / 255
    mask_curr = io.imread(mask_curr_p) / 255

    model_input = np.stack([mask_prev, finalflow[:, :, 0], finalflow[:, :, 1]], axis=2)

    return model_input, mask_curr


##########################################################################
#
# Define Data Generators
#
##########################################################################

def create_data_generators(batch_size=4):
    train, val = dataset.get_train_val()

    print("train size: ", len(train))
    print("val size: ", len(val))

    train_generator = dataset.data_generator(train, get_model_input, batch_size=batch_size)
    val_generator = dataset.data_generator(val, get_model_input, batch_size=batch_size)

    return train_generator, val_generator


train_generator, val_generator = create_data_generators()

##########################################################################
#
# Train Model
#
##########################################################################

model.train(train_generator, val_generator)
