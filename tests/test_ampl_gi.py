# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:12:24 2022

@author: von.gostev
"""
import __init__
from gi import GIEmulator
from lightprop2d import Beam2D, rectangle_hole, random_wave, gaussian_beam

import matplotlib.pyplot as plt

npoints = 128
wl0 = 0.632
nimg = 1000
area_size = 1000

test = GIEmulator(area_size, npoints, wl0, nimg,
                  init_field_gen=random_wave,
                  init_gen_args=(8,),
                  object_gen=rectangle_hole,
                  object_gen_args=(500, 100),
                  parallel_njobs=1,
                  use_gpu=True,
                  use_cupy=True
                  )
test.calculate_all()
# test.calculate_xycorr_widths()
print(test.g2)
print(test.timecorr_data)
print(test.xycorr_data)
print(test.ghost_data)
# test.xycorr_widths_data
print(test.contrast_data)
print(test.contrast)
print(test.xycorr_width)
print(test.timecorr_width)

plt.semilogy(test.xycorr_data[test.npoints // 2])
plt.show()
