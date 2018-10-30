from math import sqrt, ceil, floor
import matplotlib.pyplot as plt
from typing import List


# BASIC VISUALIZATION PRIMITIVES
def vis_fill(*imgs, rows=2, cols=2, titles: List[str] = None, **kwargs):
    n = len(imgs)

    if n > rows*cols:
        raise ValueError('not enough spots to fill with all the images')

    _, axes = plt.subplots(rows, cols, squeeze=False, **kwargs)

    i = 0
    for row in range(rows):
        for col in range(cols):
            # TODO add better support for blank plots in the middle
            axes[row][col].imshow(imgs[i])
            if titles is not None and i < len(titles):
                axes[row][col].set_title(titles[i])
            i += 1
            if i == n:
                break

    plt.show()


def vis_plot(img, title: str = None, **kwargs):
    vis_fill(img, rows=1, cols=1, titles=[title], **kwargs)


def vis_row(*imgs, titles: List[str] = None, **kwargs):
    vis_fill(*imgs, rows=1, cols=len(imgs), titles=titles, **kwargs)


def vis_col(*imgs, titles: List[str] = None, **kwargs):
    vis_fill(*imgs, rows=len(imgs), cols=1, titles=titles, **kwargs)


def vis_fix_col(*imgs, cols=2, titles: List[str] = None, **kwargs):
    n = len(imgs)
    rows = ceil(n / cols)

    vis_fill(*imgs, rows=rows, cols=cols, titles=titles, **kwargs)


def vis_fix_row(*imgs, rows=2, titles: List[str] = None, **kwargs):
    n = len(imgs)
    cols = ceil(n / rows)

    vis_fill(*imgs, rows=rows, cols=cols, titles=titles, **kwargs)


def vis_square(*imgs, titles: List[str] = None, **kwargs):
    n = len(imgs)
    rows, cols = floor(sqrt(n)), ceil(sqrt(n))

    vis_fill(*imgs, rows=rows, cols=cols, titles=titles, **kwargs)

