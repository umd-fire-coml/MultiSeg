import os

from train.wad_dataset import WadDataset
from train.davis2017_dataset import *
from train.datautils import splitd

team_dir = 'G:/Team Drives/COML-Fall-2018/T0-VidSeg/'


def test_dataset_splitting():
    # when testing on own machine, change to correct directory
    root_dir = os.path.join(team_dir, 'Data/CVPR-WAD-2018/')

    dataset = WadDataset()
    dataset.load_data(root_dir, 'train')
    dataset.prepare()
    print(dataset.num_images, '\n~~~~~~~~~~')

    subsets = splitd(dataset, 0.25, 0.5, 0.25)

    for subset in subsets:
        print(subset.num_images)


def test_davis2017():
    root_dir = os.path.join(team_dir, 'Data/DAVIS/')

    total_dataset = get_trainval(root_dir)

    for img_id in total_dataset.image_ids:
        print(total_dataset.has_mask(img_id))
        print(total_dataset.load_mask(img_id).shape[2])

    print(total_dataset.image_info[1])

    import matplotlib.pyplot as plt

    mask = total_dataset.load_mask(1)
    print(mask.shape)

    plt.imshow(mask[..., 0])
    plt.show()
    plt.imshow(mask[..., 1])
    plt.show()
    plt.imshow(mask[..., 2])
    plt.show()


def test_multimask():
    root_dir = os.path.join(team_dir, 'Data/DAVIS/')

    dataset = Davis2017Dataset('trainval', '480p', data_dir=root_dir)

    dataset.load_video('schoolgirls')

    dataset.prepare()

    import matplotlib.pyplot as plt
    import numpy as np

    mask = dataset.load_mask(0)

    for i in range(7):
        plt.imshow(mask[..., i])
        plt.show()


if __name__ == '__main__':
    # test_dataset_splitting()

    # test_davis2017()

    test_multimask()

