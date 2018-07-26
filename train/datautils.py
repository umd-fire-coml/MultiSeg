from copy import deepcopy
import numpy as np

from image_seg.utils import Dataset


def split_dataset(dataset: Dataset, *splits: float, shuffle=True) -> list:
    """
    Utility function that splits a *prepared* dataset object (that is, you've
    called dataset.prepare() ) into multiple datasets based on splits.
    :param dataset: dataset to split
    :param splits: split ratios for each split
    :param shuffle: whether to first shuffle the images before splitting
    :return: list of datasets for each split
    In general, if you pass in 'k' splits, you'll get back 'k+1' dataset
    objects. If the splits sum to 1.0, then passing in 'k' splits will return
    'k' datasets.

    The intuition behind the two different use cases are as follow: if you pass
    in a set of splits that sum to 1.0, you want to clearly (and explicitly)
    define how much data each dataset will end up with (as a portion of the
    whole dataset). In contrast, passing in a set of splits that don't sum to
    1.0 follows the logic of taking a dataset and 'stripping off' certain
    amounts for other purposes.
    """
    # TODO since we no longer need to start from back, do everything non-reversed
    if sum(splits) > 1.:
        raise ValueError('splits can sum to at most 1.0')

    if sum(splits) == 1.:
        splits = splits[1:]

    # convert split percentages to actual numbers of images
    def convert_to_partition(x):
        return round(x * dataset.num_images)

    partitions = list(map(convert_to_partition, splits))

    # shuffles the images before splitting
    if shuffle:
        np.random.shuffle(dataset.image_info)

    # 'strip off' the last x number of images (starting from the back)
    split_datasets = []
    curr_index = 0
    for partition in reversed(partitions):
        # extract the desired image info elements
        left = min(curr_index, len(dataset.image_info))
        right = min(left + partition, len(dataset.image_info))
        split_info = dataset.image_info[left:right]
        curr_index += partition

        # load the chosen images into a new dataset
        new_dataset = deepcopy(dataset)
        new_dataset.image_info = split_info
        new_dataset.prepare()

        split_datasets.append(new_dataset)

    # include the original dataset with whatever's left
    dataset.image_info = dataset.image_info[min(curr_index, len(dataset.image_info)):]
    dataset.prepare()
    split_datasets.append(dataset)

    # reverse (since we starting splitting from the end) and return the datasets
    return list(reversed(split_datasets))

