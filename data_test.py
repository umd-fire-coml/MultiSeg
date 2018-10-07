import imgaug.augmenters as iaa

from train.davis2017_dataset import *


if __name__ == '__main__':
    data_dir = 'G:\\Team Drives\\COML-Fall-2018\\T0-VidSeg\Data\\DAVIS'
    dataset = get_trainval(data_dir)

    seq = iaa.Sequential([
        iaa.MedianBlur(k=(11, 13))
    ])

    gen = dataset.paired_generator(augmentation=seq)

    X, y = next(gen)
    print(X.shape, y.shape)

    import matplotlib.pyplot as plt

    plt.imshow(X[..., :3].astype(int))
    plt.show()
    plt.imshow(X[..., 3:6].astype(int))
    plt.show()
    plt.imshow(X[..., 6].astype(int))
    plt.show()
    plt.imshow(y[..., 0])
    plt.show()

