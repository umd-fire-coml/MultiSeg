from os import path

from image_seg import utils


class Davis2017Dataset(utils.Dataset):

    def load_data(self, subset: str, quality: str):
        pass

    def load_image(self, image_id):
        pass

    def load_mask(self, image_id):
        pass

    @staticmethod
    def _build_relative_path(subset: str, quality: str, selection: str):
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


def generate_datasets_from_videos(subset: str, quality: str) -> list:
    pass


def generate_train_val_datasets(subset: str, quality: str) -> (Davis2017Dataset, Davis2017Dataset):
    pass


def generate_full_dataset(subset: str, quality: str) -> Davis2017Dataset:
    dataset = Davis2017Dataset()
    dataset.load_data(subset, quality)

    return dataset

