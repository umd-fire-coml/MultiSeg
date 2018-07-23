import os
import matplotlib.pyplot as plt
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("C:\\Users\\rmdu\\Image_Seg")
os.chdir(ROOT_DIR)
print('Project Directory: {}'.format(ROOT_DIR))

# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
print('Logs and Model Directory: {}'.format(LOGS_DIR))

from samples.balloon import balloon
config = balloon.BalloonConfig()
BALLOON_DIR = os.path.join(ROOT_DIR, "dataset\\balloon")

# Load dataset
# Get the dataset from the releases page
# https://github.com/matterport/Mask_RCNN/releases
dataset = balloon.BalloonDataset()
dataset.load_balloon(BALLOON_DIR, "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))

print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# Import Mask RCNN
from mrcnn import utils

# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    masks, class_ids = dataset.load_mask(image_id)
    plt.imshow(image)
    plt.show()
    for i in range(0, masks.shape[-1]):
        cropped_mask = utils.crop_image_by_mask(image , masks[:, :, i])
        plt.imshow(cropped_mask)
        plt.show()

