# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:21:02 2021

@author: Pavel Gostev
"""
import __init__
import matplotlib.pyplot as plt
from gi.experiment import GIExpDataProcessor

"""
settings_file -- путь к файлу с настройками эксперимента в формате json
"""
# settings_file = r'H:\SciData\GI\17_04_2019_obj_v\17_04_2019_obj_v.txt'
# settings_file = r'C:\Users\von.gostev\Downloads\17_04_2019_obj_v\17_04_2019_obj_v.txt'
# settings_file = r'H:\SciData\GI\110921\110921.txt'
# settings_file = r'H:\SciData\GI\211221_computational\ghost_proector_12_15_scatt.txt'
settings_file = r'H:\SciData\GI\170222\ghost_proector_12_23.txt'

test = GIExpDataProcessor(
    settings_file, n_images=3000, parallel_njobs=-2,
    parallel_reading=1, binning_order=1, use_cupy=False)
test.calculate_all()
test.calculate_xycorr_widths(nx=10, ny=10, window_points=64)

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
