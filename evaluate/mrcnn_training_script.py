#!usr/bin/env python

import glob
import math
import sys

##############################################################################
#
# Notes:
# 1. Current training script does NOT modify class weights or augment images.
# 2. Modified Mask RCNN library to save model history
#
# Files of interest after training:
# 1. The saved weights in ../INITIAL_TRAINING_LOGS
# 2. training_logs.txt
# 3. val_images_(timestamp).txt
#
# Training command: python TRAINING.py 2>&1 | tee training_logs.txt
# This command runs TRAINING.py and saves output to file as well as stdout
#
##############################################################################

MRCNN_DIR = '../'  # isn't working: not using local MRCNN library
MODEL_DIR = '../INITIAL_TRAINING_LOGS'
WEIGHTS_PATH = '../mask_rcnn_coco.h5'
DATA_DIR = '../../train/wad/'

num_images = len(glob.glob(DATA_DIR + "train_color/*"))
batch_size = 2
num_epochs = 10  # will define how many versions of the weights to save
val_size = 0.00505  # makes val set = 200
train_steps = math.ceil((num_images * (1 - val_size)) / (batch_size * num_epochs))
val_steps = math.ceil((num_images * val_size) / batch_size)

print("Number of train images: ", num_images)
print("Number of train steps: ", train_steps)
print("Number of validation steps: ", val_steps)
print("Size of train set: ", train_steps * batch_size * num_epochs)
print("Size of val set: ", val_steps * batch_size)

# to import local version of the library
sys.path.append(MRCNN_DIR)

import mrcnn.model as modellib

from train.wad_data import WadConfig, WadDataset

cfg = WadConfig()
cfg.IMAGES_PER_GPU = 2
cfg.STEPS_PER_EPOCH = train_steps
cfg.VALIDATION_STEPS = val_steps
cfg.display()

dataset_train = WadDataset()
dataset_val = dataset_train.load_data(DATA_DIR, "train", val_size=val_size)  # TODO fix with new splitting mechanism

dataset_train.prepare()
dataset_val.prepare()

##############################################################################
#
#  TRAINING
#
##############################################################################

we_should_train = True

if we_should_train:
    model = modellib.MaskRCNN(mode="training", config=cfg, model_dir=MODEL_DIR)

    model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    ##########################################################################
    #
    #  train Heads
    #
    ##########################################################################

    model.train(dataset_train, dataset_val,
                learning_rate=cfg.LEARNING_RATE,
                epochs=num_epochs,
                layers='heads')
