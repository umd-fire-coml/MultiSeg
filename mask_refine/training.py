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
import numpy as np
import tensorflow as tf

from train.davis2017 import *
from train.datautils import splitd
from mask_refine.config import Configuration

"""
Mapping from command description (key) to actual command line arguments (value).
Right now, they're all the same, but this dictionary is to provide
forward-compatibility in case they can in future.
"""
COMMANDS = {'train': 'train',
            'augs': 'augs',
            'sizes': 'sizes',
            'null': 'null'
            }

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
parser.add_argument('cmd', choices=COMMANDS.keys(), default=COMMANDS['train'])
parser.add_argument('--config', '-c', action='store', default=None, dest='config_path',
                    help='path of the config file, defaults to \'./config.yaml\'')


args = parser.parse_args()
cmd = args.cmd
config_path = args.config_path

config = Configuration(config_path) if config_path is not None else Configuration()
config.summary()


############################################################################

if cmd == COMMANDS['augs']:
    from train.viz import vis_square

    for inputs in get_trainval(config.dataset_path).paired_generator(AUG_SEQ):
        assert len(inputs) == 4
        
        imgs = list(inputs)

        # strip the extra dimensions and scale to image-expected ranges
        imgs[0] = (imgs[0][0, ...] * 255).astype(np.uint8)
        imgs[1] = (imgs[1][0, ...] * 255).astype(np.uint8)
        imgs[2] = (imgs[2][0, ..., 0] * 255).astype(np.uint8)
        imgs[3] = (imgs[3][0, ..., 0] * 255).astype(np.uint8)

        vis_square(*imgs, titles=['prev', 'curr', 'aug', 'gt'])
        
elif cmd == COMMANDS['train']:
    from opt_flow.opt_flow import TensorFlowPWCNet
    from mask_refine.model import MaskRefineNetwork

    dataset = get_trainval(config.dataset_path)

    train, val = splitd(*config.splits, shuffle=False)
    train_gen, val_gen = train.paired_generator(AUG_SEQ), val.paired_generator(AUG_SEQ)

    pwc_net = TensorFlowPWCNet(dataset.size, model_pathname=config.optical_flow_path,
                               verbose=config.debugging, gpu=config.optical_flow_device)

    with tf.device(config.model_device):
        mr_subnet = MaskRefineNetwork(pwc_net)

        if config.mask_refine_path is not None:
            mr_subnet.load_weights(config.mask_refine_path)

    print('Starting MaskRefine training...')

    mr_subnet.train(train_gen, val_gen, config=config)
elif cmd == COMMANDS['sizes']:
    from opt_flow.opt_flow import TensorFlowPWCNet
    from mask_refine.model import MaskRefineNetwork
    import numpy as np
    
    def run_size_test(name, inputs, predictor):
        outputs = predictor(inputs)
        
        print(f'{name}:')
        print(f'Input Shape:\t{inputs.shape}')
        print(f'Output Shape:\t{outputs.shape}')

    dataset = get_trainval(config.dataset_path)

    with tf.device(config.model_device):
        pwc_net = TensorFlowPWCNet(dataset.size, model_pathname=config.optical_flow_path, verbose=config.debugging)
        mr_subnet = MaskRefineNetwork(pwc_net)
        
        run_size_test('MaskRefineSubnet', np.empty((1, 480, 854, 6)), mr_subnet.predict)
        run_size_test('PWCNet', np.empty((480, 854, 6)), pwc_net.infer_from_image_stack)
else:
    print('\nDebugging run completed (no training done).')

