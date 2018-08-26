import os
from os import path

from image_seg import utils


__all__ = ['Davis2017Dataset',
           'generate_datasets_from_videos',
           'generate_train_val_datasets',
           'generate_full_dataset']

_subset_param_desc = "which subset of DAVIS 2017 to get (either 'trainval', " \
                      "'test-dev', or 'test-challenge')"

_quality_param_desc = "quality of the video (either '480p', '1080p', or 'fullres'); " \
                       "'1080p' and 'fullres' refer to the same image quality"

_selection_param_desc = "which objects to retrieve (either 'images', 'labels', or " \
                         "'videos')"


class Davis2017Dataset(utils.Dataset):
    name = 'DAVIS2017'
    def __init__(self, subset, quality, data_dir='./'):
        super(self.__class__, self).__init__(self)

        self.subset = subset
        self.quality = quality
        self.data_dir = data_dir

    def load_all_data(self):
        pass

    def load_data(self, video: str, frame_ids, assume_match=True):
        try:
            iter(frame_ids)
        except TypeError:
            frame_ids = list(frame_ids)

        image_path = path.join(self.data_dir,
                               Davis2017Dataset.build_relative_path(self.subset, self.quality, 'images'),
                               video)
        label_path = path.join(self.data_dir,
                               Davis2017Dataset.build_relative_path(self.subset, self.quality, 'labels'),
                               video)

        for fid in frame_ids:
            img_path = path.join(image_path, f'{fid}.jpg')
            lab_path = path.join(image_path, f'{fid}.png')

            self.add_image(self.name, f'{video}{fid}', img_path, mask_path=lab_path)

    def load_image(self, image_id):
        pass

    def load_mask(self, image_id):
        pass

    def __str__(self):
        return '<Davis 2017 Dataset>'

    @staticmethod
    def build_relative_path(subset: str, quality: str, selection: str):
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

        if selection == 'images':
            selection = 'JPEGImages'
        elif selection == 'labels':
            selection = 'Annotations'

        base_path = f'DAVIS-2017-{subset}-{quality}/DAVIS/'
        if selection == 'videos':
            rel_path = 'ImageSets/2017/'
        else:
            rel_path = f'{selection}/{quality}/'

        return path.join(base_path, rel_path)


def generate_datasets_from_videos(root_dir: str, subset: str, quality: str) -> list:
    images_root_dir = path.join(root_dir, Davis2017Dataset.build_relative_path(subset, quality, 'images'))
    labels_root_dir = path.join(root_dir, Davis2017Dataset.build_relative_path(subset, quality, 'labels'))

    _, videos_from_images, _ = next(os.walk(images_root_dir))
    _, videos_from_labels, _ = next(os.walk(labels_root_dir))
    videos_from_images = set(videos_from_images)
    videos_from_labels = set(videos_from_labels)

    videos = videos_from_images & videos_from_labels

    datasets = []

    for video_filename in videos:
        new_dataset = Davis2017Dataset()

        image_dir = path.join(images_root_dir, video_filename)
        label_dir = path.join(labels_root_dir, video_filename)

        _, _, images = next(os.walk(image_dir))
        _, _, labels = next(os.walk(label_dir))
        images = set(images)
        labels = set(labels)

        print('from images', images)
        print('from labels', labels)

        images = images & labels

        datasets.append(new_dataset)

    return datasets


def generate_train_val_datasets(subset: str, quality: str) -> (Davis2017Dataset, Davis2017Dataset):
    f"""
    Generates the default DAVIS2017 train and validation datasets.
    :param subset: {_subset_param_desc}
    :param quality: {_quality_param_desc}
    :return: (train dataset, val dataset)
    """
    pass


def generate_full_dataset(subset: str, quality: str) -> Davis2017Dataset:
    dataset = Davis2017Dataset()
    dataset.load_data(subset, quality)

    return dataset


if __name__ == '__main__':
    lst = generate_datasets_from_videos('G:\Team Drives\COML-Summer-2018\Data\DAVIS', 'trainval', '480p')
    print(lst)

