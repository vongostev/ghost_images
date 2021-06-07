# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 20:14:57 2021

@author: Дима
"""
import __init__
import matplotlib.pyplot as plt
from lightprop2d import Beam2D, random_round_hole
from gi.emulation import ImgEmulator


# All input data are in cm
# XY grid dimensions
npoints = 128
# XY grid widening
beam_radius = 25e-4  # 25 um
area_size = 200e-4  # 200 um
# Wavelength in cm
wl0 = 632e-7

test = ImgEmulator(area_size, npoints, wl0, 100,
                   init_field_gen=random_round_hole,
                   init_gen_args=(25e-4,),
                   z_ref=500e-4)
test.spatial_coherence()

plt.plot(test.ghost_data[npoints // 2])
plt.show()

fake_beam = Beam2D(area_size, npoints, wl0, init_field=test.ghost_data)
print(fake_beam.gaussian_fwhm)
