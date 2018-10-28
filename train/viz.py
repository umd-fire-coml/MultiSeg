from math import sqrt, ceil, floor
import matplotlib.pyplot as plt
import numpy as np


# BASIC VISUALIZATION PRIMITIVES
def vis_plot(img):
    plt.imshow(img)
    plt.show()


def vis_row(*imgs):
    n = len(imgs)

    if n == 1:
        vis_plot(imgs[0])
        return

    _, axes = plt.subplots(1, n)

    for i in range(n):
        axes[i].imshow(imgs[i])

    plt.show()


def vis_col(*imgs):
    n = len(imgs)

    if n == 1:
        vis_plot(imgs[0])
        return

    _, axes = plt.subplots(n, 1)

    for i in range(n):
        axes[i].imshow(imgs[i])

    plt.show()


def vis_fill(*imgs, rows=2, cols=2):
    n = len(imgs)

    if n > rows*cols:
        raise ValueError('not enough spots to fill with all the images')

    if rows == 1:
        vis_row(*imgs)
    elif cols == 1:
        vis_col(*imgs)
    else:
        _, axes = plt.subplots(rows, cols)

        i = 0
        for row in range(rows):
            for col in range(cols):
                axes[row][col].imshow(imgs[i])
                i += 1
                if i == n:
                    break

        plt.show()


def vis_fix_col(*imgs, cols=2):
    n = len(imgs)
    rows = ceil(n / cols)

    vis_fill(*imgs, rows=rows, cols=cols)


def vis_fix_row(*imgs, rows=2):
    n = len(imgs)
    cols = ceil(n / rows)

    vis_fill(*imgs, rows=rows, cols=cols)


def vis_square(*imgs):
    n = len(imgs)
    rows, cols = floor(sqrt(n)), ceil(sqrt(n))

    vis_fill(*imgs, rows=rows, cols=cols)

