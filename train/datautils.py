from copy import deepcopy
import numpy as np

from image_seg.utils import Dataset


def split_dataset(dataset: Dataset, *splits: float, shuffle=True) -> list:
    """
    Utility function that splits a *prepared* dataset object (that is, you've
    called dataset.prepare() ) into multiple datasets based on splits. If you
    pass in 'k' splits, you'll get back 'k+1' dataset objects.
    :param dataset: dataset to split off of
    :param splits: split ratios for each split
    :param shuffle: whether to first shuffle the images before splitting
    :return: list of datasets for each split
    """
    if sum(splits) >= 1.:
        raise ValueError('splits must total less than 1.0')

    def convert_to_partition(x):
        return round(x * dataset.num_images)

    partitions = list(map(convert_to_partition, splits))

    if shuffle:
        np.random.shuffle(dataset.image_info)

    split_datasets = []
    curr_index = 0
    for partition in reversed(partitions):
        left = min(curr_index, len(dataset.image_info))
        right = min(left + partition, len(dataset.image_info))
        split_info = dataset.image_info[left:right]
        curr_index += partition

        new_dataset = deepcopy(dataset)
        new_dataset.image_info = split_info
        new_dataset.prepare()

        split_datasets.append(new_dataset)

    dataset.image_info = dataset.image_info[min(curr_index, len(dataset.image_info)):]
    dataset.prepare()
    split_datasets.append(dataset)

    return list(reversed(split_datasets))

