#!usr/bin/env python
"""
Mask Refine Training & Utilities Script

Commands Available:
    * train - run the training script for the mask refine module
    * augs - display the images of a dataset before and after augmentation
             (requires a matplotlib backend that supports display)
    * sizes - run each component of the module separate on fixed size inputs to
              verify input and output tensor sizes/shapes
    * infer - (WIP) predict an overall refined mask for an image pair from a
              dataset (with augmentation, not with image segmentation integration)
              
More information about the command line arguments can be found by running the
command `python -m train.mrefine -h` from the root directory of this project.
"""

import argparse
import imgaug.augmenters as iaa
import sys
from warnings import warn
import tensorflow as tf

from train.davis2017_dataset import *
from train.datautils import splitd

"""
Mapping from command description (key) to actual command line arguments (value).
Right now, they're all the same, but this dictionary is to provide
forward-compatibility in case they can in future.
"""
COMMANDS = {'train': 'train',
            'infer': 'infer',
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
parser.add_argument('-d', '--dataset', dest='dataset_path', type=str,
                    nargs=1,
                    default=['G:\\Team Drives\\COML-Fall-2018\\T0-VidSeg\\Data\\DAVIS'],
                    )
parser.add_argument('-o', '--optical-flow', dest='optical_flow_path', type=str,
                    nargs=1,
                    default=['./opt_flow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'])
parser.add_argument('-m', '--mask-refine', dest='mask_refine_path', type=str,
                    nargs=1,
                    default=[None])
parser.add_argument('-v', '--validation-split', dest='val_split', type=float,
                    nargs=1, default=[0.15])
parser.add_argument('-e', '-epochssteps', dest='epochssteps', type=int,
                    nargs=2, default=[200, 1000])
parser.add_argument('-p', '--print-debugs', dest='print_debugs', action='store_true')
parser.add_argument('--gpu', dest='device', type=int, nargs=1, default=[0])

############################################################################

args = parser.parse_args()

cmd = args.cmd
dataset_path = args.dataset_path[0]
optical_flow_path = args.optical_flow_path[0]
mask_refine_path = args.mask_refine_path[0]
val_split = args.val_split[0]
epochs = args.epochssteps[0]
steps = args.epochssteps[1]
print_debugs = args.print_debugs
device = args.device[0]

print('Arguments given to trainmaskrefine command: ')
print(f'\tcommand\t{cmd}')
print(f'\tdataset\t{dataset_path}')
print(f'\toptical\t{optical_flow_path}')
print(f'\tmrefine\t{mask_refine_path}')
print(f'\tv split\t{val_split}')
print(f'\tepochs\t{epochs}')
print(f'\tsteps\t{steps}')
print(f'\tdebugs\t{print_debugs}')
print(f'\tdevice\tGPU:{device}')
print()


def printd(string):
    """Debugging print wrapper."""
    
    if print_debugs:
        print(string)


def warn_if_debugging_without_prints(command):
    """
    Warn if running a debugging command (e.g. sizes) without debugging statements.
    If the user doesn't wish to continue, the system will exit.

    Args:
        command: name of the debugging command
    """
    
    # if printing debugs, then it's fine
    if print_debugs:
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
    dataset = get_trainval(dataset_path)

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

    dataset = get_trainval(dataset_path)

    train, val = splitd(dataset, 1 - val_split, val_split, shuffle=False)
    train_gen, val_gen = train.paired_generator(AUG_SEQ), val.paired_generator(AUG_SEQ)

    pwc_net = TensorFlowPWCNet(dataset.size, model_pathname=optical_flow_path,
                               verbose=print_debugs, gpu=device)
    
    with tf.device(f'/device:GPU:{device + 1}'):
        with pwc_net.graph.as_default():
            mr_subnet = MaskRefineSubnet(pwc_net)
    
            if mask_refine_path is not None:
                mr_subnet.load_weights(mask_refine_path)
    
            printd('Starting MaskRefine training...')
    
            mr_subnet.train(train_gen, val_gen, epochs=epochs, steps_per_epoch=steps)
elif cmd == COMMANDS['sizes']:
    warn_if_debugging_without_prints("sizes")

    from opt_flow.opt_flow import TensorFlowPWCNet
    from mask_refine.mask_refine import MaskRefineSubnet
    import numpy as np

    dataset = get_trainval(dataset_path)

    with tf.device(f'/device:GPU:{device}'):
        pwc_net = TensorFlowPWCNet(dataset.size, model_pathname=optical_flow_path, verbose=print_debugs)
        with pwc_net.graph.as_default():
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

