#!usr/bin/env python

# Imports
import skimage.io as io
import cv2
from datetime import datetime

import keras
from keras.layers import *
from keras.backend import tf

# Static GPU memory allocation for tensorflow (need some GPU for PyTorch Optical Flow)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Our own modules
from data.davis_wrapper_new import DavisDataset
from pwc_net.pytorch.pwc_net_wrapper import PWCNetWrapper
from .MaskPropagationModuleDavis import get_model

# Command to run: python TRAIN_MASK_PROP_FocalLoss.py 2>&1 | tee training_logs_fl_1.txt

run_step_by_step = False

if run_step_by_step: input("Imports Done. Next: Load Optical Flow and UNet, Continue? (type anything): ")

##########################################################################
#
# Load Dataset Wrapper, Optical Flow Model, and UNet
#
##########################################################################

dataset = DavisDataset("DAVIS", "480p", val_videos=[
    "car-shadow", "breakdance", "camel", "scooter-black", "libby", "drift-straight"
])
opticalflow = PWCNetWrapper("../MaskTrack_RCNN/pwc_net/pytorch/pwc_net.pth.tar")
model = get_model(input_shape=(480, 864, 3))

if run_step_by_step: input("Loaded Optical Flow and UNet. Next: Load Dataset. Continue? (type anything): ")


##########################################################################
#
# Data Preprocessing Methods
#
##########################################################################


def pad_image(image):
    # for davis, opitcal flow output always maps (480, 854) -> (480, 864)
    # for UNet, both dimensions must be a multiple of 8
    return cv2.copyMakeBorder(image, 0, 0, 5, 5, cv2.BORDER_CONSTANT, value=0)


def get_model_input(img_prev_p, img_curr_p, mask_prev_p, mask_curr_p):
    """
    Returns tensor that contains previous mask and optical flow, and also
    returns current mask as the ground truth value.
    """
    img_prev, img_curr = io.imread(img_prev_p), io.imread(img_curr_p)
    img_prev, img_curr = pad_image(img_prev), pad_image(img_curr)

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

    mask_prev = pad_image(io.imread(mask_prev_p)) / 255
    mask_curr = pad_image(io.imread(mask_curr_p)) / 255

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


batch_size = 4
num_epochs = 2

train, val = dataset.get_train_val()

print("train size: ", len(train))
print("val size: ", len(val))

train_generator = dataset.data_generator(train, get_model_input, batch_size=batch_size)
val_generator = dataset.data_generator(val, get_model_input, batch_size=batch_size)

if run_step_by_step: input("Loaded Data. Next: Train Model. Continue? (type anything): ")

##########################################################################
#
# Train Model
#
##########################################################################

history_file = "log_dir/NEW_VAL_FL_history_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir="log_dir",
        histogram_freq=0,
        write_graph=True,
        write_images=False
    ),
    keras.callbacks.ModelCheckpoint(
        "log_dir/NEW_VAL_FL_davis_unet_weights__{epoch:02d}__{val_loss:.2f}.hdf5",
        verbose=0, save_weights_only=True
    ),
    keras.callbacks.CSVLogger(history_file)
]

model = get_model()

# New addition: load weights trained on simple binary crossentropy loss
model.load_weights("log_dir/NEW_davis_unet_weights__30__0.03.hdf5")

history = model.fit_generator(
    train_generator,
    steps_per_epoch=int(len(train) / batch_size),
    validation_data=val_generator,
    validation_steps=int(len(val) / batch_size),
    epochs=num_epochs,
    callbacks=callbacks
)
