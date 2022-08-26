# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:12:24 2022

@author: Pavel Gostev
"""
import __init__
from gi import GIEmulator, filter_from_img
from lightprop2d import rectangle_hole, random_wave

import matplotlib.pyplot as plt

npoints = 32
wl0 = 0.632
nimgs = npoints ** 2 
area_size = 1000

test_objects = [
    (rectangle_hole, (500, 100)),
    (filter_from_img('img/alum.png', npoints), ())
]

for i, (obj, args) in enumerate(test_objects):
    test = GIEmulator(area_size, npoints, wl0, nimgs=nimgs,
                      init_field_gen=random_wave,
                      init_gen_args=(1,),
                      object_gen=obj,
                      object_gen_args=args,
                      use_gpu=False,
                      use_cupy=False,
                      use_dask=False
                      )
    test.calculate_all()
    test.calculate_xycorr_widths(nx=16, ny=16, window_points=16)

    test.timecorr_data
    test.xycorr_data
    test.ghost_data
    test.xycorr_widths_data
    test.contrast_data
    test.g2_data

    print(test.g2)
    print(test.contrast)
    print(test.xycorr_width)
    print(test.timecorr_width)

    plt.imshow(test.ghost_data)
    plt.colorbar()
    plt.show()

    plt.semilogy(test.xycorr_data[test.npoints // 2])
    plt.show()

    plt.imshow(test.xycorr_widths_data.mean(axis=0))
    plt.colorbar()
    plt.show()
