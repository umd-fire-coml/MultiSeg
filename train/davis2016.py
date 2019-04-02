import glob
import numpy as np
import re
import skimage.io

from image_seg import config, utils
from os.path import join, exists

###############################################################################
#                              CLASS DICTIONARIES                             #
###############################################################################

classes = {
    0: 'No Mask',
    1: 'Mask'
}

###############################################################################
#                                CONFIGURATION                                #
###############################################################################


class DAVISConfig(config.Config):
    NAME = 'DAVIS'
    NUM_CLASSES = len(classes) + 1
    BACKBONE = 'resnet101'


###############################################################################
#                                   DATASET                                   #
###############################################################################

class Davis2016Dataset(utils.Dataset):
    # TODO not working yet
    name = 'DAVIS'
    image_height = 480
    image_width = 854

    def __init__(self, root_dir=None, random_state=42):
        super(self.__class__, self).__init__(self)

        # Add classes (35)
        for class_id, class_name in classes.items():
            self.add_class(self.name, class_id, class_name)

        self.root_dir = root_dir
        self.random_state = random_state

    def load_data(self, root_dir, labeled=True, assume_match=False):
        """Load a subset of the DAVIS image segmentation dataset.
        :param root_dir: Root directory of the train
        :param subset: Which subset to load: images will be looked for in 'subset_color' and masks will
        be looked for in 'subset_label' (will look for pickle file subset.pkl first)
        :param labeled: Whether the images have ground-truth masks
        :param assume_match: Whether to assume all images have ground-truth masks (ignored if labeled
        is False)
        """
        self.root_dir = root_dir

        # Check directories for existence
        assert exists(join(self.root_dir)), 'the root directory doesn\'t exist'
        # TODO fill in (incomplete implementation)
        self.load_video(root_dir, labeled=labeled, assume_match=assume_match)

    def load_video(self, video_list_filename, labeled=True, assume_match=False):
        """Loads all the images from a particular video list into the dataset.
        :param video_list_filename: path of the file containing the list of images
        :param labeled: whether these images have corresponding labels
        :param assume_match: Whether to assume all images have ground-truth masks
        """
        # TODO add error checking everywhere
        # involve those unused parameters
        # Get list of images for this video
        video_file = open(video_list_filename, 'r')
        image_filename_pairs = video_file.readlines()
        video_file.close()

        for pair in image_filename_pairs:
            image_filename, mask_filename = pair.split(' ')
            image_id = re.match('some regex', image_filename)  # fill in regex
            self.add_image(self.name, image_id=image_id, path=image_filename, mask_path=mask_filename)

    def load_image(self, image_id: int):
        """Load the specified image and return a [H,W,3] Numpy array.
        :param image_id: integer id of the image (less than num_images)
        """

        info = self.image_info[image_id]

        # If not a DAVIS dataset image, delegate to parent class
        if info["source"] != self.name:
            return super(self.__class__, self).load_image( image_id)

        # Load image
        path = join(self.root_dir, info['path'])
        image = skimage.io.imread(path)

        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        return image

    def load_mask(self, image_id: int) -> (np.ndarray, int):
        """Generate instance masks for an image. If an image does not have any
        masks, a zero-ed out array will be returned with class_id of 0.
        :param image_id: integer id of the image
        :returns masks, class_id
        masks: bool array of shape [h, w, 1]
        class_id: id of the mask, either 0 or 1
        """
        info = self.image_info[image_id]

        # If not a DAVIS dataset image, delegate to parent class
        if info["source"] != "DAVIS":
            return super(self.__class__, self).load_mask(image_id)

        # Read the original mask image
        mask_path = join(self.root_dir, info['mask_path'])
        mask = skimage.io.imread(mask_path)

        # correct for weird instances of multi-channel masks
        if mask.ndims == 3 and mask.shape[-1] != 1:
            mask = mask[..., 0]

        assert mask.shape == [self.image_height, self.image_width, 1],\
            'loaded mask is not in the expected shape'

        # Return mask and class of mask
        return mask, int(np.sum(mask) != 0)

    def image_reference(self, image_id: int) -> str:
        """Return the image filename."""

        info = self.image_info[image_id]

        if info["source"] == "DAVIS":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class MaskPropDavisDataset(object):
    # TODO: in init include way to download dataset
    # include download link and expected directory structure

    def __init__(self, directory, quality, val_videos=()):
        """
        :param directory: root directory of the DAVIS dataset
        :param quality: video image quality (e.g. '480p')

        self.frame_pairs = an array of tuples of the form:
        (img_prev, img_curr, mask_prev, mask_curr) PATHS
        """

        # TODO currently fails silently

        # generate mask pairs

        self.trn_frame_pairs = []  # tuples of image and masks at t-1 and t
        self.val_frame_pairs = []

        image_dir = f"{directory}/JPEGImages/{quality}/"
        mask_dir = f"{directory}/Annotations/{quality}/"

        # CHANGE HERE: splitting into train_videos and val_videos
        videos = [x[len(image_dir):] for x in glob.glob(image_dir + "*")]
        self.trn_videos = list(set(videos) - set(val_videos))
        self.val_videos = val_videos

        # CHANGE HERE:
        for video in videos:

            frames = [x[len(image_dir) + len(video) + 1:-4] for x in glob.glob(image_dir + video + "/*")]
            frames.sort()

            for prev, curr in zip(frames[:-1], frames[1:]):

                image_prev = image_dir + video + "/" + prev + ".jpg"
                image_curr = image_dir + video + "/" + curr + ".jpg"
                mask_prev = mask_dir + video + "/" + prev + ".png"
                mask_curr = mask_dir + video + "/" + curr + ".png"

                if video in self.trn_videos:
                    self.trn_frame_pairs.append((image_prev, image_curr, mask_prev, mask_curr))
                else:
                    self.val_frame_pairs.append((image_prev, image_curr, mask_prev, mask_curr))

    def get_train_val(self, shuffle=True, random_state=42):
        if shuffle:
            np.random.seed(random_state)
            trn_frame_pairs = self.trn_frame_pairs.copy()
            val_frame_pairs = self.val_frame_pairs.copy()
            np.random.shuffle(trn_frame_pairs)
            np.random.shuffle(val_frame_pairs)
            return trn_frame_pairs, val_frame_pairs
        return self.trn_frame_pairs, self.val_frame_pairs

    def get_video_split(self):
        return self.trn_videos, self.val_videos

    def get_random_pair(self, val=True, random_state=42):
        # returns from training
        np.random.seed(random_state)
        if val:
            return self.val_frame_pairs[np.random.choice(range(len(self.val_frame_pairs)))]
        return self.trn_frame_pairs[np.random.choice(range(len(self.trn_frame_pairs)))]

    def data_generator(self, frame_pairs, get_model_input, batch_size=4, random_seed=None):
        """
        :param frame_pairs:
        :param get_model_input: function to get model input given a frame pair (i.e. apply Optical Flow)
        :param batch_size:
        :param random_seed:
        """

        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(frame_pairs)

        i = 0

        while True:
            batch_count = 0
            X = []
            y = []

            while batch_count < batch_size:
                # TODO first call doesn't take into account if frame_pairs is empty
                sample = frame_pairs[i]
                model_input, ground_truth = get_model_input(*sample)

                if model_input is not None and ground_truth is not None:
                    X.append(model_input)
                    # THIS PART IS WRONG: makes the output dimensions (batch_size, 480, 864, 1)
                    # But I don't think it impacts the output
                    y.append(np.expand_dims(ground_truth, axis=3))
                    batch_count = batch_count + 1

                i = i + 1  # keeps track of where we are in dataset

                # restart -> may cause repeats in samples within epoch or validation run
                if i >= len(frame_pairs):
                    i = 0  # go back to beginning
                    np.random.shuffle(frame_pairs)

            X = np.array(X)
            y = np.array(y)

            yield X, y

