#!usr/bin/env python
"""
Mask Refine Training & Utilities Script

Commands Available:
    * train - run the training script for the mask refine module
    * augs - display the images of a dataset before and after augmentation
             (requires a matplotlib backend that supports display)
    * sizes - run each component of the module separate on fixed size inputs to
              verify expected input and output tensor shapes
              
Information about the command line arguments can be found by running the
command `python -m mask_refine.train -h` from the root directory of this project.
"""

import argparse
import imgaug.augmenters as iaa
import sys
from warnings import warn
import tensorflow as tf

from train.davis2017_dataset import *
from train.datautils import splitd
from mask_refine.config import Configuration

"""
Mapping from command description (key) to actual command line arguments (value).
Right now, they're all the same, but this dictionary is to provide
forward-compatibility in case they can in future.
"""
COMMANDS = {'train': 'train',
            'augs': 'augs',
            'sizes': 'sizes'
            }
COMMANDS_LIST = list(COMMANDS.values())

"""
Sequence of imgaug augmentations applied to ground truth masks to transform
them into input masks for (coarse) training.
"""
AUG_SEQ = iaa.Sequential([
        iaa.Multiply((0.25, 0.95)),
        iaa.ElasticTransformation(alpha=(2000, 10000), sigma=(20, 100)),
        iaa.GaussianBlur(sigma=(0, 20)),
        iaa.Sometimes(0.25, iaa.Multiply((0.5, 0.75))),
        iaa.MultiplyElementwise((0.8, 1.1)),
        iaa.Sometimes(0.05, iaa.GaussianBlur(sigma=(5, 100))),
        iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=(1, 15)))
    ])


# PARSE COMMAND LINE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('cmd', choices=COMMANDS_LIST,
                    default=COMMANDS['train'],
                    help='Display plots of the augmented masks used for training',
                    )

############################################################################

args = parser.parse_args()

cmd = args.cmd

config = Configuration()
config.summary()


def printd(string):
    """Debugging print wrapper."""
    
    if config.debugging:
        print(string)


def warn_if_debugging_without_prints(command):
    """
    Warn if running a debugging command (e.g. sizes) without debugging statements.
    If the user doesn't wish to continue, the system will exit.

    Args:
        command: name of the debugging command
    """
    
    # if printing debugs, then it's fine
    if config.debugging:
        return
    
    # warn when not printing debugs
    warn(f'\'{command}\' is a debugging command, but debugs are not printed. (Use -p or -print-debugs to output.)')
    response = input('Are you sure you want to continue? [y/n] ')
    if 'y' not in response.strip().lower():
        sys.exit()
    else:
        printd('Continuing...')


############################################################################

if cmd == COMMANDS['augs']:
    dataset = get_trainval(config.dataset_path)

    gen = dataset.paired_generator(AUG_SEQ)

    for X, y in gen:
        import matplotlib.pyplot as plt

        printd(f'X.shape: {X.shape}, y.shape: {y.shape}')

        plt.imshow(X[..., 6].astype(int))
        plt.show()
        plt.imshow(y[..., 0])
        plt.show()
elif cmd == COMMANDS['train']:
    from opt_flow.opt_flow import TensorFlowPWCNet
    from mask_refine.mask_refine import MaskRefineSubnet

    dataset = get_trainval(config.dataset_path)

    train, val = splitd(*config.splits, shuffle=False)
    train_gen, val_gen = train.paired_generator(AUG_SEQ), val.paired_generator(AUG_SEQ)

    pwc_net = TensorFlowPWCNet(dataset.size, model_pathname=config.optical_flow_path,
                               verbose=config.debugging, gpu=config.optical_flow_device)

    with tf.device(f'/device:GPU:{device + 1}'):
        mr_subnet = MaskRefineSubnet(pwc_net)

        if config.mask_refine_path is not None:
            mr_subnet.load_weights(config.mask_refine_path)

    printd('Starting MaskRefine training...')

    mr_subnet.train(train_gen, val_gen, epochs=config.epochs_per_run, steps_per_epoch=config.steps_per_epoch)
elif cmd == COMMANDS['sizes']:
    warn_if_debugging_without_prints("sizes")

    from opt_flow.opt_flow import TensorFlowPWCNet
    from mask_refine.mask_refine import MaskRefineSubnet
    import numpy as np

    dataset = get_trainval(config.dataset_path)

    with tf.device(config.model_device):
        pwc_net = TensorFlowPWCNet(dataset.size, model_pathname=config.optical_flow_path, verbose=config.debugging)
        
        mr_subnet = MaskRefineSubnet(pwc_net)
        
        input_stack = np.empty((1, 480, 854, 6))
        output = mr_subnet.predict(input_stack)

        printd('INPUT/OUTPUT INFERENCE TESTS')
        printd('MaskRefineSubnet:')
        printd(f'Input Shape:\t{input_stack.shape}')
        printd(f'Output Shape:\t{output.shape}')

        input_stack = np.empty((480, 854, 6))
        output = pwc_net.infer_from_image_stack(input_stack)

        printd('PWCNet:')
        printd(f'Input Shape:\t{input_stack.shape}')
        printd(f'Output Shape:\t{output.shape}')

        input_stack = np.empty((480, 854, 7))
        output = mr_subnet.predict(input_stack)

        printd('Entire Module:')
        printd(f'Input Shape:\t{input_stack.shape}')
        printd(f'Output Shape:\t{output.shape}')

