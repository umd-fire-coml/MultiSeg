from train.wad_dataset import WadDataset
from train.datautils import split_dataset

if __name__ == '__main__':
    root_dir = 'G:/Team Drives/COML-Summer-2018/Data/CVPR-WAD-2018/'

    dataset = WadDataset(root_dir)
    dataset.load_data(root_dir, 'train')
    dataset.prepare()
    print(dataset.num_images, '\n~~~~~~~~~~')

    subsets = split_dataset(dataset, 0.5, 0.25)

    for subset in subsets:
        print(subset.num_images)
