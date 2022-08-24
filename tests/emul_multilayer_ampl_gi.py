# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:30:30 2022

@author: Pavel Gostev
"""

import __init__
import numpy as np
from gi import GIEmulator, filter_scale, multilayer_object, filter_from_img
from lightprop2d import rectangle_hole, random_wave


import matplotlib.pyplot as plt


npoints = 128
wl0 = 0.632
nimgs = 1024 * 8
area_size = 1000

scale = 0.5
o1 = filter_scale(rectangle_hole, scale)
o2 = filter_scale(filter_from_img('img/alum.png', npoints), scale)
filters = [o2, o1, o2]
fargs = (500, 100)  # (100, 500, 125, 0)]
layers = multilayer_object(filters, fargs)


tests = []
for layer in layers:
    tests.append(
        GIEmulator(area_size, npoints, wl0, nimgs=nimgs,
                   init_field_gen=random_wave,
                   init_gen_args=(4,),
                   object_gen=layer,
                   use_gpu=True,
                   use_cupy=True,
                   use_dask=False
                   )
    )
    tests[-1].calculate_ghostimage()


for i in range(len(tests)):
    gimg = tests[i].ghost_data / \
        np.prod([(1 - tests[k].ghost_data) ** 2 for k in range(i)], axis=0)

    fig, axes = plt.subplots(1, 2)
    im1 = axes[0].imshow(tests[i].ghost_data / tests[i].ghost_data.max())
    axes[0].set_title(f'Слой {i+1}')
    im2 = axes[1].imshow(gimg / gimg.max())
    axes[1].set_title(f'Слой {i+1} с коррекцией')
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.show()
