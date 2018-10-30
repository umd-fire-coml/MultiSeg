#!usr/bin/env python

################################################################################
#                   Mask Refine Training Script & Utilities                    #
#                                                                              #
# Commands Available:                                                          #
#   * train - run the training script for the mask refine module               #
#   * augs - display the images of a dataset before & after augmentation       #
#   * sizes - run each component of the module separately on fixed size inputs #
#             to verify input and output sizes                                 #
#                                                                              #
################################################################################

import argparse
import imgaug.augmenters as iaa
import sys
from warnings import warn

from train.davis2017_dataset import *
from train.datautils import splitd

# mapping from command description (key) to actual command line arg (value)
# (right now, they're all the same, but this provides forward compatibility in
# case we change the command names in future )
COMMANDS = {'train': 'train',
            'infer': 'infer',
            'augs': 'augs',
            'sizes': 'sizes'
            }
COMMANDS_LIST = list(COMMANDS.values())


def load_data_peripherals(dpath):
    dataset = get_trainval(dpath)

    seq = iaa.Sequential([
        iaa.Multiply((0.25, 0.95)),
        iaa.ElasticTransformation(alpha=(2000, 10000), sigma=(20, 100)),
        iaa.GaussianBlur(sigma=(0, 20)),
        iaa.Sometimes(0.25, iaa.Multiply((0.5, 0.75))),
        iaa.MultiplyElementwise((0.8, 1.1)),
        iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(5, 100))),
        iaa.AdditiveGaussianNoise(scale=(1, 25))
    ])

    return dataset, seq


def warn_if_debugging_without_prints(command):
    if not print_debugs:
        warn(f'"{command}" is a debugging command, but debugs are not printed. (Use -p or -print-debugs to output.)')
        response = input('Are you sure you want to continue? [y/n] ')
        if 'y' not in response.strip().lower():
            sys.exit()


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

print('Arguments given to trainmaskrefine command: ')
print(f'\tcommand\t{cmd}')
print(f'\tdataset\t{dataset_path}')
print(f'\toptical\t{optical_flow_path}')
print(f'\tmrefine\t{mask_refine_path}')
print(f'\tv split\t{val_split}')
print(f'\tepochs\t{epochs}')
print(f'\tsteps\t{steps}')
print(f'\tdebugs\t{print_debugs}')
print()


def printd(string):
    if print_debugs:
        print(string)


############################################################################

if cmd == COMMANDS['augs']:
    dataset, seq = load_data_peripherals(dataset_path)

    gen = dataset.paired_generator(seq)

    for X, y in gen:
        import matplotlib.pyplot as plt

        printd(f'X.shape: {X.shape}, y.shape: {y.shape}')

        plt.imshow(X[..., 6].astype(int))
        plt.show()
        plt.imshow(y[..., 0])
        plt.show()
elif cmd == COMMANDS['train']:
    from opt_flow.opt_flow import TensorFlowPWCNet
    from mask_refine.mask_refine import MaskRefineSubnet, MaskRefineModule

    dataset, seq = load_data_peripherals(dataset_path)

    train, val = splitd(dataset, 1 - val_split, val_split, shuffle=False)
    train_gen, val_gen = train.paired_generator(seq), val.paired_generator(seq)

    pwc_net = TensorFlowPWCNet(dataset.size, model_pathname=optical_flow_path, verbose=print_debugs)
    with pwc_net.graph.as_default():
        mr_subnet = MaskRefineSubnet()
        mr_module = MaskRefineModule(pwc_net, mr_subnet)

        if mask_refine_path is not None:
            mr_subnet.load_weights(mask_refine_path)

        printd('Starting MaskRefine training...')

        hist = mr_module.train(train_gen, val_gen, epochs=epochs, steps_per_epoch=steps)
        printd(hist)
elif cmd == COMMANDS['infer']:
    from opt_flow.opt_flow import TensorFlowPWCNet
    from mask_refine.mask_refine import MaskRefineSubnet, MaskRefineModule

    dataset, seq = load_data_peripherals(dataset_path)
    imgs = dataset.paired_generator(seq)

    pwc_net = TensorFlowPWCNet(dataset.size, model_pathname=optical_flow_path, verbose=print_debugs)
    with pwc_net.graph.as_default():
        mr_subnet = MaskRefineSubnet()
        mr_module = MaskRefineModule(pwc_net, mr_subnet)

        if mask_refine_path is not None:
            mr_subnet.load_weights(mask_refine_path)
    # TODO work in progress

elif cmd == COMMANDS['sizes']:
    warn_if_debugging_without_prints("sizes")

    from opt_flow.opt_flow import TensorFlowPWCNet
    from mask_refine.mask_refine import MaskRefineSubnet, MaskRefineModule
    import numpy as np

    dataset, seq = load_data_peripherals(dataset_path)

    pwc_net = TensorFlowPWCNet(dataset.size, model_pathname=optical_flow_path, verbose=print_debugs)
    with pwc_net.graph.as_default():
        mr_subnet = MaskRefineSubnet()
        mr_module = MaskRefineModule(pwc_net, mr_subnet)

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
        output = mr_module.refine_mask(input_stack)

        printd('Entire Module:')
        printd(f'Input Shape:\t{input_stack.shape}')
        printd(f'Output Shape:\t{output.shape}')

