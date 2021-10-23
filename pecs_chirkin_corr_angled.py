# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 23:18:02 2021

@author: vonGostev
"""
import __init__
from gi import ImgEmulator
from gi.slm import slm_phaseprofile, slm_expand
from lightprop2d import Beam2D, gaussian_beam, mm, cm, um, rectangle_hole
import numpy as np
import matplotlib.pyplot as plt

npoints = 2 ** 9
area_size = 0.2*cm
beam_radius = area_size / 4
wl = 0.632*um  # nm

img_number = 100

rng = np.random.default_rng()


def slm_random_gaussbeam(x, y):
    ibeam = Beam2D(area_size, npoints,
                   wl, init_field_gen=gaussian_beam,
                   init_gen_args=(1, beam_radius), use_gpu=1, unsafe_fft=True)

    slm = slm_phaseprofile(area_size / npoints,
                           rng.uniform(0, 2 * np.pi, size=(50, 80)),
                           pixel_size=100e-4, pixel_gap=0, angle=np.pi / 3)
    slm = slm_expand(slm, npoints)
    ibeam.coordinate_filter(f_init=slm)
    return ibeam.field


emulator = ImgEmulator(area_size*um, npoints,
                       wl*um, imgs_number=img_number,
                       init_field_gen=slm_random_gaussbeam,
                       object_gen=rectangle_hole,
                       object_gen_args=(10*um, 50*um),
                       use_gpu=1,
                       z_ref=1, z_obj=1,
                       parallel_njobs=-1
                       )

emulator.calculate_xycorr()
plt.imshow(emulator.xycorr_data)
