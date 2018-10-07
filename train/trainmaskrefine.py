#!usr/bin/env python

import argparse
import imgaug.augmenters as iaa

from mask_refine.mask_refine import MaskRefineSubnet, MaskRefineModule
from train.davis2017_dataset import *
from train.datautils import splitd
from opt_flow.pwc_net_wrapper import *

commands = ['train', 'augs']

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train mask refine module')
    parser.add_argument('cmd', choices=commands,
                        default=commands[0],
                        help='Display plots of the augmented masks used for training',
                        )
    parser.add_argument('-d', '--dataset', dest='dataset_path', type=str,
                        nargs=2,
                        default=['-d', 'G:\\Team Drives\\COML-Fall-2018\\T0-VidSeg\Data\\DAVIS'],
                        )

    args = parser.parse_args()

    ##############################################################

    dataset = get_trainval(args.dataset_path[1])

    seq = iaa.Sequential([
        iaa.ElasticTransformation(alpha=(200, 1000), sigma=(20, 100)),
        iaa.GaussianBlur(sigma=(0.1, 7.5)),
        iaa.AdditiveGaussianNoise(scale=(1, 5))
    ])

    if args.cmd == 'augs':
        gen = dataset.paired_generator(seq)

        for X, y in gen:
            import matplotlib.pyplot as plt

            plt.imshow(X[..., 6].astype(int))
            plt.show()
            plt.imshow(y[..., 0])
            plt.show()
    elif args.cmd == 'train':
        train, val = splitd(dataset, 0.8, 0.2)  # TODO parametrize this
        train_gen, val_gen = train.paired_generator(), val.paired_generator()

        pwc_net = PWCNetWrapper()
        mr_subnet = MaskRefineSubnet()
        mr_module = MaskRefineModule(pwc_net, mr_subnet)

