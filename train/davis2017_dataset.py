import os
from os import path
import skimage
from warnings import warn

from PIL import Image
import numpy as np

from image_seg import utils


__all__ = ['Davis2017Dataset',
           'get_trainval', 'get_test_dev', 'get_test_challenge']

_subset_param_desc = "which subset of DAVIS 2017 to get (either 'trainval', " \
                      "'test-dev', or 'test-challenge')"

_quality_param_desc = "quality of the video (either '480p', '1080p', or 'fullres'); " \
                       "'1080p' and 'fullres' refer to the same image quality"

_selection_param_desc = "which objects to retrieve (either 'images', 'labels', or " \
                         "'videos')"


class Davis2017Dataset(utils.Dataset):
    """
    Class for representing a dataset of images from the DAVIS 2017 Dataset.

    Methods for Loading Images into Dataset:
     * load_subset - loads a particular subset of the data, as defined by the DAVIS 2017 Challenge
     * load_video - loads a particular video from a particular subset of the data
     * load_frame - loads a particular frame from a particular video from a particular subset of the data

    Methods for Actually Loading Images:
     * load_image - returns the image tensor
     * load_mask - returns the mask tensor and class array
    """
    name = 'DAVIS2017'

    def __init__(self, subset: str, quality: str, data_dir='./'):
        super(self.__class__, self).__init__(self)

        if not path.exists(data_dir):
            warn('Creating a Dataset with a root directory that does not exist on the current machine. Most '
                 'load methods will fail.')

        self.subset = subset
        self.quality = quality
        self.data_dir = data_dir

    @property
    def all_masked(self):
        for img_dict in self.image_info:
            if 'mask_path' not in img_dict:
                return False

        return True

    def load_subset(self, *videos):
        """
        Loads all the videos within this subset and video quality for the
        videos given. If a list of videos is given, only those videos will be
        loaded; otherwise, all available videos will be loaded.
        :param videos: an iterable of strings, containing the names of the videos to load
        """

        if len(videos) == 0:
            _, videos, _ = next(os.walk(self.build_absolute_path_to('images', '')))

        for vid in videos:
            self.load_video(vid)

    def load_video(self, video: str):
        """
        Loads all the frames from a specific video.
        :param video: name of the video to load from
        """

        _, _, frame_filenames = next(os.walk(self.build_absolute_path_to('images', video)))

        for fname in frame_filenames:
            self.load_frame(video, fname)

    def load_frame(self, video, img_filename):
        """
        Loads a single frame from the video specified.
        :param video: name of the video to load from
        :param img_filename: filename (with extension) for the image

        img_filename and mask_filename should be identical except for their extensions.
        """

        img_id = img_filename[:-4]
        img_path_to_store = path.join(video, img_filename)
        mask_path_to_store = path.join(video, f'{img_id}.png')

        if path.exists(self.build_absolute_path_to('labels', mask_path_to_store)):
            self.add_image(self.name, img_id, img_path_to_store, mask_path=mask_path_to_store)
        else:
            self.add_image(self.name, img_id, img_path_to_store)

    def load_image(self, image_id: int):
        info = self.image_info[image_id]

        if info['source'] != self.name:
            return super(self.__class__, self).load_image(image_id)

        img_path = self.build_absolute_path_to('images', info['path'])

        # Load image
        image = skimage.io.imread(img_path)

        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        return image

    def has_mask(self, image_id: int):
        """Returns whether the image/mask pair with image_id actually has a mask."""

        return 'mask_path' in self.image_info[image_id]

    def load_mask(self, image_id: int):
        info = self.image_info[image_id]

        if info['source'] != self.name:
            return super(self.__class__, self).load_mask(image_id)

        if not self.has_mask(image_id):
            raise ValueError('this image does not have a mask')

        mask_path = self.build_absolute_path_to('labels', info['mask_path'])

        # mask = skimage.io.imread(mask_path)

        mask = np.atleast_3d(Image.open(mask_path))[..., 0]
        mask = np.expand_dims(mask, axis=2)

        uniqs = np.expand_dims(np.expand_dims(np.unique(mask), axis=0), axis=0)
        uniqs = np.delete(uniqs, 0)

        mask_tensor = mask == uniqs

        # extra info from davis2017 official util function
        # np.array(im.getpalette()).reshape((-1, 3))

        # Return mask and class of mask
        return mask_tensor, uniqs

    def __str__(self):
        try:
            return f'<Davis 2017 Dataset: {len(self.image_ids)} images prepared >'
        except:
            return '<Davis 2017 Dataset (unprepared)>'

    def build_absolute_path_to(self, selection: str, video_and_filename: str) -> str:
        """
        Builds and returns an absolute path to the file specified by filename, given a selection.
        :param selection: either 'images', 'labels', or 'videos'
        :param video_and_filename: name of the file
        :return: full absolute path
        """
        return path.join(self.data_dir,
                         Davis2017Dataset.build_relative_path(self.subset, self.quality, selection),
                         video_and_filename)

    @staticmethod
    def build_relative_path(subset: str, quality: str, selection: str) -> str:
        f"""
        Builds a relative path to the directory containing the dataset with the
        parameters given.
        :param subset: {_subset_param_desc}
        :param quality: {_quality_param_desc}
        :param selection: {_selection_param_desc}
        :return: path to the desired selection
        """
        if subset not in ['trainval', 'test-dev', 'test-challenge']:
            raise ValueError(f'{subset} not a subset for DAVIS 2017')
        if quality not in ['480p', 'fullres', '1080p']:
            raise ValueError(f'{quality} not one of the available qualities')
        if selection not in ['images', 'labels', 'videos']:
            raise ValueError('selection must be either \'images\' or \'labels\'')

        if quality == 'fullres' or quality == '1080p':
            quality = 'Full-Resolution'

        base_path = f'DAVIS-2017-{subset}-{quality}/DAVIS/'

        if selection == 'images':
            selection = 'JPEGImages'
        elif selection == 'labels':
            selection = 'Annotations'

        if selection == 'videos':
            rel_path = 'ImageSets/2017/'
        else:
            rel_path = f'{selection}/{quality}/'

        return path.join(base_path, rel_path)


def _load_predefined(root_dir, subset, quality) -> Davis2017Dataset:
    dataset = Davis2017Dataset(subset, quality, data_dir=root_dir)
    dataset.load_subset()
    dataset.prepare()

    return dataset


def get_trainval(root_dir, quality='480p') -> Davis2017Dataset:
    """
    Returns the prepared trainval dataset as defined by DAVIS 2017.
    :param quality: {_quality_param_desc}
    :return: trainval dataset for quality requested
    """

    return _load_predefined(root_dir, 'trainval', quality)


def get_test_dev(root_dir, quality='480p') -> Davis2017Dataset:
    """
    Returns the prepared test-dev dataset as defined by DAVIS 2017.
    :param quality: {_quality_param_desc}
    :return: test-dev dataset for quality requested
    """

    return _load_predefined(root_dir, 'test-dev', quality)


def get_test_challenge(root_dir, quality='480p') -> Davis2017Dataset:
    """
    Returns the prepared test-challenge dataset as defined by DAVIS 2017.
    :param quality: {_quality_param_desc}
    :return: test-challenge dataset for quality requested
    """

    return _load_predefined(root_dir, 'test-challenge', quality)

