# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:12:24 2022

@author: von.gostev
"""
import __init__
from gi import GIEmulator
from lightprop2d import rectangle_hole, random_wave

import matplotlib.pyplot as plt

npoints = 256
wl0 = 0.632
nimg = 1000
area_size = 1000

test = GIEmulator(area_size, npoints, wl0, nimg,
                  init_field_gen=random_wave,
                  init_gen_args=(16,),
                  object_gen=rectangle_hole,
                  object_gen_args=(500, 100),
                  parallel_njobs=1,
                  use_gpu=True,
                  use_cupy=True
                  )
test.calculate_all()
test.calculate_xycorr_widths(nx=20, ny=20, window_points=32)
print(test.g2)
# print(test.timecorr_data)
# print(test.xycorr_data)
# print(test.ghost_data)
# test.xycorr_widths_data
# print(test.contrast_data)
print(test.contrast)
print(test.xycorr_width)
print(test.timecorr_width)

plt.semilogy(test.xycorr_data[test.npoints // 2])
plt.show()

plt.imshow(test.xycorr_widths_data.mean(axis=0))
plt.colorbar()
plt.show()
