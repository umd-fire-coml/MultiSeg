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
from pwc_net.pwc_net_wrapper import PWCNetWrapper
from .mask_propagation import MaskPropagation

##########################################################################
#
# Load Dataset Wrapper, Optical Flow Model, and UNet
#
##########################################################################

dataset = DavisDataset("DAVIS", "480p", val_videos=[
    "car-shadow", "breakdance", "camel", "scooter-black", "libby", "drift-straight"
])
opticalflow = PWCNetWrapper("../MaskTrack_RCNN/pwc_net/pytorch/pwc_net.pth.tar")
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

    # Check 1
    if img_prev.shape != img_curr.shape:
        print("ERROR: img_prev.shape != img_curr.shape", img_prev_p, img_prev.shape, img_curr.shape)
        return None, None
    if img_prev.shape != (480, 864, 3):
        print("ERROR: img_prev.shape != (480, 864, 3)", img_prev_p, img_prev.shape)
        return None, None

    finalflow = opticalflow.infer_flow_field(img_prev, img_curr)
    finalflow_x, finalflow_y = finalflow[:, :, 0], finalflow[:, :, 1]
    finalflow[:, :, 0] = (finalflow_x - finalflow_x.mean()) / finalflow_x.std()
    finalflow[:, :, 1] = (finalflow_y - finalflow_y.mean()) / finalflow_y.std()

    # Check 2
    if finalflow.shape != (480, 864, 2):
        print("ERROR: finalflow.shape != (480, 864, 2)", img_prev_p, finalflow.shape)
        return None, None

    mask_prev = mask_prev_p / 255
    mask_curr = mask_curr_p / 255

    # Check 3
    if mask_prev.shape != mask_curr.shape:
        print("ERROR: mask_prev.shape != mask_curr.shape", img_prev_p, mask_prev.shape, mask_curr.shape)
        return None, None
    if mask_prev.shape != (480, 864):
        print("ERROR: mask_prev.shape != (480, 864)", img_prev_p, mask_prev.shape)
        return None, None

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
