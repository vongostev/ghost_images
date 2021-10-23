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
npoints = 256
# XY grid widening
beam_radius = 25e-4  # 25 um
area_size = 200e-4  # 200 um
# Wavelength in cm
wl0 = 632e-7

# test = ImgEmulator(area_size, npoints, wl0, 1000,
#                    init_field_gen=random_round_hole,
#                    init_gen_args=(25e-4,),
#                    z_ref=500e-4)
test = ImgEmulator(area_size, npoints, wl0, 100,
                   expdata_dir=r'H:\SciData\GI\23_05_H_40\23_05_H_40',
                   use_expdata=True,
                   z_ref=0e-4)
test.calculate_xycorr()

plt.plot(test.xycorr_data[test.npoints // 2])
plt.show()

fake_beam = Beam2D(area_size, test.npoints, wl0, init_field=test.xycorr_data)
print(fake_beam.D4sigma)
