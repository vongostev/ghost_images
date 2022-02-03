# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 20:14:57 2021

@author: Дима
"""
import __init__
import matplotlib.pyplot as plt
from lightprop2d import round_hole
from gi.emulation import GIEmulator


# All input data are in cm
# XY grid dimensions
# ref_dir = r'H:\SciData\GI\211221_computational\patterns'
ref_dir = r'H:\SciData\GI\23_05_H_40\23_05_H_40'
# ref_crop = (300, 812, 0, 512)
ref_crop = (150, 360,	70,	280)
npoints = 256
# XY grid widening
area_size = 210 * 6.45  # 200 um
# Wavelength in cm
wl0 = 632

test = GIEmulator(area_size, npoints, wl0, 2000,
                  object_gen=round_hole,
                  object_gen_args=(area_size / 4,),
                  binning_order=2,
                  expdata_dir=ref_dir,
                  expdata_crop=ref_crop,
                  img_prefix='Pattern',
                  use_expdata=True,
                  use_cupy=True)
test.calculate_all()
test.calculate_xycorr_widths(nx=20, ny=20, window_points=32)

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

plt.plot(test.times, test.timecorr_data)
plt.show()
plt.imshow(test.ghost_data)
plt.show()

plt.imshow(test.xycorr_data)
plt.show()

plt.imshow(test.xycorr_widths_data.mean(axis=0))
plt.colorbar()
plt.show()
