"""
This module contains a number of visualization routines to help visualize our
results (for images and masks).
"""

from math import sqrt, ceil, floor
import matplotlib.pyplot as plt
from typing import List


# BASIC VISUALIZATION PRIMITIVES
def vis_fill(*imgs, rows=2, cols=2, titles: List[str] = None, save_path=None, **kwargs):
    """
    Visualizes images in a grid of shape [rows, cols], as specified, filling in
    each spot with images provided in row-major order.
    
    Args:
        *imgs: images to show (in row-major order)
        rows: number of rows in the grid
        cols: number of cols in the grid
        titles: list of titles for each subplot
        save_path: file path to save the generated figure
        **kwargs: arguments to be passed to the figure

    Raises:
        ValueError if there are more images than spots
        
    If an image of None is passed in, that image will be skipped (the subplot
    will remain blank).
    """
    n = len(imgs)

    if n > rows*cols:
        raise ValueError('not enough spots to fill with all the images')

    fig, axes = plt.subplots(rows, cols, squeeze=False, **kwargs)

    # going in row-major order, fill the spots of the grid with images
    i = 0
    for row in range(rows):
        for col in range(cols):
            if imgs[i]:
                axes[row][col].imshow(imgs[i])
                if titles is not None and i < len(titles):
                    axes[row][col].set_title(titles[i])
            i += 1
            if i == n:
                break

    # save the figure
    if save_path is not None:
        fig.savefig(save_path)
    
    # show the figure
    plt.show()


def vis_plot(img, title: str = None, **kwargs):
    """
    Visualizes a single image.
    
    Args:
        img: image to show
        title: title of the plot
        **kwargs: arguments to be passed to the figure
    """
    
    vis_fill(img, rows=1, cols=1, titles=[title], **kwargs)


def vis_row(*imgs, titles: List[str] = None, **kwargs):
    """
    Visualizes a row of images.
    
    Args:
        *imgs: list of images to show
        titles: list of titles for each subplot
        **kwargs: arguments to be passed to the figure
    """
    vis_fill(*imgs, rows=1, cols=len(imgs), titles=titles, **kwargs)


def vis_col(*imgs, titles: List[str] = None, **kwargs):
    """
    Visualizes a column of images.

    Args:
        *imgs: list of images to show
        titles: list of titles for each subplot
        **kwargs: arguments to be passed to the figure
    """
    
    vis_fill(*imgs, rows=len(imgs), cols=1, titles=titles, **kwargs)


def vis_fix_col(*imgs, cols=2, titles: List[str] = None, **kwargs):
    """
    Visualizes a set of images in a grid-like format, with a fixed number of
    columns, as specified.
    
    Args:
        *imgs: list of images to show (processed in row-major order)
        cols: number of columns in the grid
        titles: list of titles for each subplot
        **kwargs: arguments to be passed to the figure
    """
    n = len(imgs)
    rows = ceil(n / cols)

    vis_fill(*imgs, rows=rows, cols=cols, titles=titles, **kwargs)


def vis_fix_row(*imgs, rows=2, titles: List[str] = None, **kwargs):
    """
    Visualizes a set of images in a grid-like format, with a fixed number of
    rows, as specified.

    Args:
        *imgs: list of images to show (processed in row-major order)
        rows: number of rows in the grid
        titles: list of titles for each subplot
        **kwargs: arguments to be passed to the figure
    """
    n = len(imgs)
    cols = ceil(n / rows)

    vis_fill(*imgs, rows=rows, cols=cols, titles=titles, **kwargs)


def vis_square(*imgs, titles: List[str] = None, **kwargs):
    """
    Visualizes the images in a grid that is as close to a square as possible.
    
    Args:
        *imgs: list of images to show (processed in row-major order)
        titles: list of titles for each subplot
        **kwargs: arguments to be passed to the figure
    """
    n = len(imgs)
    rows, cols = floor(sqrt(n)), ceil(sqrt(n))

    vis_fill(*imgs, rows=rows, cols=cols, titles=titles, **kwargs)


# KERAS MODEL VISUALIZATION
def save_model_graph_plot(model, filename):
    from keras.utils import plot_model
    
    plot_model(model, to_file=filename, show_shapes=True)


_metric_names = {
    'acc': 'Accuracy',
    'loss': 'Loss',
    'binary_crossentropy': 'Binary Cross Entropy',
    'mask_binary_crossentropy': 'Mask-Normalized Binary Cross Entropy'
}


def plot_history_metric(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'Model {_metric_names[metric]}')
    plt.ylabel(_metric_names[metric])
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_history(history):
    plot_history_metric(history, 'acc')
    plot_history_metric(history, 'loss')

