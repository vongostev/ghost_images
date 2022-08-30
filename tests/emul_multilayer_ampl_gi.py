# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:30:30 2022

@author: Pavel Gostev
"""

import __init__
import numpy as np
from gi import GIEmulator, filter_scale, multilayer_object
from lightprop2d import square_hole, random_wave

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_3D_array_slices(imgdata, area_size):
    _, n_x, n_y = imgdata.shape
    # X, Y = np.mgrid[-n_x // 2:n_x // 2, -n_y // 2:n_y // 2] / n_x * area_size
    X, Y = np.mgrid[-n_x // 2:n_x // 2, -n_y // 2:n_y // 2] + n_x // 2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, T in enumerate(imgdata[::-1]):
        ax.contourf(X, Y, T + i + 1, cmap='rainbow_alpha', levels=64)

    # ax.set_xticks(np.linspace(-area_size / 2, area_size / 2, 10))
    # ax.set_yticks(np.linspace(-area_size / 2, area_size / 2, 10))
    # ax.set_zlim(ax.get_zlim()[::-1])

    ax.set_xticks(np.arange(0, 129, 32))
    ax.set_yticks(np.arange(0, 129, 32))
    ax.set_zticks(np.arange(1, 10, 2))
    ax.set_zticklabels(list(map(str, np.arange(1, 10, 2)[::-1])))
    ax.view_init(elev=15.,azim=45)
    # ax.set_xlabel('пиксели')
    # ax.set_ylabel('пиксели')
    #ax.set_ylabel('мкм')
    ax.set_zlabel('Слои', rotation=90)
    # plt.colorbar(ax=ax)
    plt.tight_layout()
    plt.savefig('img/oktaeder.png', dpi=300, bbox_inches='tight')
    plt.show()


# get colormap
ncolors = 256
color_array = plt.get_cmap('gist_rainbow')(range(ncolors))

# change alpha values
color_array[:, -1] = np.linspace(0, 1, ncolors)

# create a colormap object
map_object = LinearSegmentedColormap.from_list(
    name='rainbow_alpha', colors=color_array)

# register this new colormap with matplotlib
plt.register_cmap(cmap=map_object)

npoints = 128
wl0 = 0.632
nimgs = 1024 * 8
area_size = 1024

scale = 0.05
# Oktaeder
o1 = filter_scale(square_hole, scale)
# o2 = filter_scale(filter_from_img('img/alum.png', npoints), scale)
filters = [o1] * 9
fargs = [(x,) for x in range(128, 128 * 6, 128)]  # (100, 500, 125, 0)]
fargs += fargs[-2::-1]
layers = multilayer_object(filters, fargs)


tests = []
for layer in layers:
    tests.append(
        GIEmulator(area_size, npoints, wl0, nimgs=nimgs,
                   init_field_gen=random_wave,
                   init_gen_args=(8,),
                   object_gen=layer,
                   use_gpu=False,
                   use_cupy=False,
                   use_dask=False
                   )
    )
    tests[-1].calculate_ghostimage()


imgdata = np.concatenate(
    [[t.ghost_data] for t in tests], axis=0)

for i in range(len(imgdata)):
    imgdata[i] = tests[i].ghost_data / \
        np.prod([(1 - tests[k].ghost_data) ** 2 for k in range(i)], axis=0)
    imgdata[i] /= imgdata[i].max()
    fig, axes = plt.subplots(1, 2)
    im1 = axes[0].imshow(tests[i].ghost_data / tests[i].ghost_data.max())
    axes[0].set_title(f'Слой {i+1}')
    im2 = axes[1].imshow(imgdata[i])
    axes[1].set_title(f'Слой {i+1} с коррекцией')
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.show()

# imgdata = np.concatenate(
#     [[t.ghost_data / t.ghost_data.max()] for t in tests], axis=0)

# imgdata[imgdata < 0.1] = 0
plot_3D_array_slices(imgdata, area_size)
