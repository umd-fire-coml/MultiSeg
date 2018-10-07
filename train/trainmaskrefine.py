import argparse
import imgaug.augmenters as iaa

from train.davis2017_dataset import *

datasets = ['DAVIS2017', 'WAD']

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train mask refine module')
    # parser.add_argument('dataset')

    args = parser.parse_args()

    ##############################################################

    data_dir = 'G:\\Team Drives\\COML-Fall-2018\\T0-VidSeg\Data\\DAVIS'
    dataset = get_trainval(data_dir)

    seq = iaa.Sequential([
        iaa.ElasticTransformation(alpha=(10, 1000), sigma=(10, 100)),
        iaa.GaussianBlur(sigma=(0.1, 7.5)),
        iaa.AdditiveGaussianNoise(scale=10)

    ])

    gen = dataset.paired_generator(seq)

    for X, y in gen:
        import matplotlib.pyplot as plt

        plt.imshow(X[..., 6].astype(int))
        plt.show()
        plt.imshow(y.astype(int))
        plt.show()

