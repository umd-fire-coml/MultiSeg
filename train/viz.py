from math import sqrt, ceil, floor
import matplotlib.pyplot as plt
from typing import List


# BASIC VISUALIZATION PRIMITIVES
def vis_plot(img, title: str = None):
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


def vis_row(*imgs, titles: List[str] = None):
    n = len(imgs)

    if n == 1:
        vis_plot(imgs[0])
        return

    _, axes = plt.subplots(1, n)

    for i in range(n):
        axes[i].imshow(imgs[i])
        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i])

    plt.show()


def vis_col(*imgs, titles: List[str] = None):
    n = len(imgs)

    if n == 1:
        vis_plot(imgs[0])
        return

    _, axes = plt.subplots(n, 1)

    for i in range(n):
        axes[i].imshow(imgs[i])
        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i])

    plt.show()


def vis_fill(*imgs, rows=2, cols=2, titles: List[str] = None):
    n = len(imgs)

    if n > rows*cols:
        raise ValueError('not enough spots to fill with all the images')

    if rows == 1:
        vis_row(*imgs, titles=titles)
    elif cols == 1:
        vis_col(*imgs, titles=titles)
    else:
        _, axes = plt.subplots(rows, cols)

        i = 0
        for row in range(rows):
            for col in range(cols):
                axes[row][col].imshow(imgs[i])
                if titles is not None and i < len(titles):
                    axes[row][col].set_title(titles[i])
                i += 1
                if i == n:
                    break

        plt.show()


def vis_fix_col(*imgs, cols=2, titles: List[str] = None):
    n = len(imgs)
    rows = ceil(n / cols)

    vis_fill(*imgs, rows=rows, cols=cols, titles=titles)


def vis_fix_row(*imgs, rows=2, titles: List[str] = None):
    n = len(imgs)
    cols = ceil(n / rows)

    vis_fill(*imgs, rows=rows, cols=cols, titles=titles)


def vis_square(*imgs, titles: List[str] = None):
    n = len(imgs)
    rows, cols = floor(sqrt(n)), ceil(sqrt(n))

    vis_fill(*imgs, rows=rows, cols=cols, titles=titles)

