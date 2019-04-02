"""
Wrapper class and associated information for the 2018 CVPR WAD dataset
primarily for use with mrcnn, but also includes utilities for working
with other networks in this project.

Usage:
train_dataset = Dataset(root_dir=<root directory for data>)

"""

import numpy as np
import os
import pickle
import re
import skimage.io

from os.path import join, isfile, exists

from image_seg import config, utils
from train import datautils

__all__ = ['WadConfig', 'WadDataset', 'class_names', 'classes_to_index', 'index_to_classes', 'index_to_class_names']

###############################################################################
#                              CLASS DICTIONARIES                             #
###############################################################################

class_names = {
    33: 'car',
    34: 'motorcycle',
    35: 'bicycle',
    36: 'person',
    37: 'rider',
    38: 'truck',
    39: 'bus',
    40: 'tricycle',
    0: 'others',
    1: 'rover',
    17: 'sky',
    161: 'car_groups',
    162: 'motorcycle_group',
    163: 'bicycle_group',
    164: 'person_group',
    165: 'rider_group',
    166: 'truck_group',
    167: 'bus_group',
    168: 'tricycle_group',
    49: 'road',
    50: 'sidewalk',
    65: 'traffic_cone',
    66: 'road_pile',
    67: 'fence',
    81: 'traffic_light',
    82: 'pole',
    83: 'traffic_sign',
    84: 'wall',
    85: 'dustbin',
    86: 'billboard',
    97: 'building',
    98: 'bridge',
    99: 'tunnel',
    100: 'overpass',
    113: 'vegetation'
}

classes_to_index = {e: i + 1 for i, e in enumerate(class_names.keys())}
index_to_classes = {v: k for k, v in classes_to_index.items()}
index_to_class_names = {v: class_names[k] for k, v in classes_to_index.items()}


###############################################################################
#                                CONFIGURATION                                #
###############################################################################

class WadConfig(config.Config):
    NAME = 'WAD'

    NUM_CLASSES = len(class_names) + 1

    BACKBONE = 'resnet101'


###############################################################################
#                                   DATASET                                   #
###############################################################################

# TODO cleanup

class WadDataset(datautils.Dataset):
    name = 'WAD'

    image_height = 2710
    image_width = 3384

    def __init__(self, random_state: int = None):
        super(self.__class__, self).__init__(self)

        # Add classes (35)
        for class_id, class_name in class_names.items():
            self.add_class(self.name, classes_to_index[class_id], class_name)

        # set internal configuration
        self.random_state = 42 if random_state is None else random_state

    def load_video(self, video_list_filename: str, labeled: bool = True, assume_match: bool = False):
        """Loads all the images from a particular video list into the dataset.
        :param video_list_filename: path of the file containing the list of images
        :param labeled: whether the images have a corresponding mask (and which
        file format to assume)
        :param assume_match: whether to assume the gt mask file exists
        """

        # Get list of images for this video
        video_file = open(video_list_filename, 'r')
        image_filenames = video_file.readlines()
        video_file.close()

        if image_filenames is None:
            print('No video list found at {}.'.format(video_list_filename))
            return

        # Generate images and masks
        for img_mask_paths in image_filenames:
            # Set paths and img_id
            if labeled:
                matches = re.search('^.*\\\\(.*\\.jpg).*\\\\(.*\\.png)', img_mask_paths)
                img_file, mask_file = matches.group(1, 2)
                img_id = img_file[:-4]
            else:
                matches = re.search('^([0-9a-zA-z]+)', img_mask_paths)
                img_id = matches.group(1)
                img_file = img_id + '.jpg'
                mask_file = None

            # Check if files exist
            if not isfile(join(self.root_dir + '_color', img_file)):
                continue
            if not assume_match and not isfile(join(self.root_dir + '_label', mask_file)):
                mask_file = None

            # Add the image to the dataset
            self.add_image(self.name, image_id=img_id, path=img_file, mask_path=mask_file)

    def _load_all_images(self, labeled: bool = True, assume_match: bool = False):
        """Load all images from the img_dir directory, with corresponding masks
        if doing training.
        :param labeled: whether the images have a corresponding mask
        :param assume_match: whether to assume all images have ground-truth masks (ignored if labeled is false)
        """

        # Retrieve list of all images in directory
        images = next(os.walk(self.root_dir + '_color'))[2]

        # Iterate through images and add to dataset
        for img_filename in images:
            img_id = img_filename[:-4]

            # If using masks, only add images to dataset that also have a mask
            if labeled:
                mask_filename = img_id + '_instanceIds.png'

                # Ignores the image (doesn't add) if no mask exists
                if not assume_match and not isfile(join(self.root_dir + '_label', mask_filename)):
                    continue
            else:
                mask_filename = None

            # Adds the image to the dataset
            self.add_image('WAD', img_id, img_filename, mask_path=mask_filename)

    def load_data(self, root_dir: str, subset: str,
                  labeled: bool = True, assume_match: bool = False, use_pickle: bool = True):
        """Load a subset of the WAD image segmentation dataset.
        :param root_dir: Root directory of the train
        :param subset: Which subset to load: images will be looked for in 'subset_color' and masks will
        be looked for in 'subset_label' (will look for pickle file subset.pkl first)
        :param labeled: Whether the images have ground-truth masks
        :param assume_match: whether to assume the mask files for all images exist (ignored if labeled
        is False)
        :param use_pickle: If False, forces a fresh load of the files
        """
        self.root_dir = join(root_dir, subset)

        pickle_path = self.root_dir + '.pkl'

        if use_pickle and isfile(pickle_path):
            self.load_data_from_file(pickle_path)
        else:
            # Check directories for existence
            assert exists(self.root_dir + '_color')
            if labeled:
                assert exists(self.root_dir + '_label')

            # blindly load all images
            self._load_all_images(labeled=labeled, assume_match=assume_match)

            # save list of images for future speed improvements
            self.save_data_to_file(pickle_path)

    def load_image(self, image_id: int) -> np.ndarray:
        """Load the specified image and return a [H,W,3] Numpy array.
        image_id: integer id of the image
        """

        info = self.image_info[image_id]

        # If not a WAD dataset image, delegate to parent class
        if info["source"] != 'WAD':
            return super(self.__class__, self).load_image(image_id)

        # Load image
        path = join(self.root_dir + '_color', info['path'])
        image = skimage.io.imread(path)

        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        return image

    def load_mask(self, image_id: int) -> (np.ndarray, np.ndarray):
        """Generate instance masks for an image.
        image_id: integer id of the image
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]

        # If not a WAD dataset image, delegate to parent class
        if info["source"] != "WAD":
            return super(self.__class__, self).load_mask(image_id)

        # Read the original mask image
        mask_path = join(self.root_dir + '_label', info['mask_path'])
        raw_mask = skimage.io.imread(mask_path)

        # unique is a sorted array of unique instances (including background)
        unique = np.unique(raw_mask)

        # section that removes/involves background
        index = np.searchsorted(unique, 255)
        unique = np.delete(unique, index, axis=0)

        # tensors!
        raw_mask = raw_mask.reshape(self.image_height, self.image_width, 1)

        # k = instance_count
        # broadcast (h, w, 1) x (k,) => (h, w, k) : bool array
        masks = raw_mask == unique

        # get the actual class id
        # int(PixelValue / 1000) is the label (class of object)
        unique = np.floor_divide(unique, 1000)
        class_ids = np.array([classes_to_index[e] for e in unique])

        # Return mask, and array of class IDs of each instance.
        return masks, class_ids

    def load_data_from_file(self, filename: str):
        """Load images from pickled file.
        filename: name of the pickle file
        """
        with open(filename, 'rb') as f:
            self.image_info = pickle.load(f)

    def save_data_to_file(self, filename: str):
        """Save loaded images to pickle file.
        filename: name of the pickle file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.image_info, f)

    def image_reference(self, image_id: int) -> str:
        """Return the image filename."""

        info = self.image_info[image_id]

        if info["source"] == self.name:
            # TODO return a meaningful and correct path
            return join(self.root_dir, info["path"])
        else:
            super(self.__class__, self).image_reference(image_id)

