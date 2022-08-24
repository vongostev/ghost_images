# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:53:05 2022

@author: vonGostev
"""

import __init__
import numpy as np
import matplotlib.pyplot as plt

from lightprop2d import plane_wave, square_hole, rectangle_hole, gaussian_beam, FilterComposer
from lightprop2d import Beam2D, mm, um

# XY grid dimensions
npoints = 512
# All input data are in cm
# XY grid widening
area_size = 2 * mm
# Wavelength in cm
wl0 = 0.532 * um

# Round hole radius
R = 0.5 * mm


def test_filter_composer():

    try:
        FilterComposer(None, ())
    except ValueError as E:
        print(E)
    try:
        FilterComposer([], ())
    except ValueError as E:
        print(E)
    print(FilterComposer(plane_wave, ()))
    print(FilterComposer(plane_wave, ()))
    print(FilterComposer(square_hole, (1, )))
    #print(FilterComposer([plane_wave] * 2, ()).f)
    print(FilterComposer([square_hole] * 2, (1, )))
    print(FilterComposer([square_hole, rectangle_hole], ((1, ), (1, 0.5))))


def test_spatial_filter(*, f_gen=None, f_init=None, f_args=()):

    beam = Beam2D(area_size, npoints, wl0, init_field_gen=gaussian_beam,
                  init_gen_args=(1, R), use_gpu=False)
    # beam = Beam2D(area_size, npoints, wl0, init_field_gen=plane_wave, use_gpu=1)
    # plt.imshow(beam.iprofile)
    # plt.show()

    # Spatial filtering
    beam.coordinate_filter(f_init, f_gen, f_args)

    plt.imshow(beam.iprofile)
    plt.show()


d = 0.25 * area_size
test_filter_composer()

fs = [
    FilterComposer(plane_wave, ()).f,
    FilterComposer(plane_wave, ()).f,
    FilterComposer(square_hole, (d, )).f,
    #FilterComposer([plane_wave] * 2, ()).f,
    FilterComposer([square_hole] * 2, (d, )).f,
    FilterComposer([square_hole, rectangle_hole],
                   ((d, ), (d * 2, d * 0.5))).f,
]

for f in fs:
    test_spatial_filter(f_gen=f)
