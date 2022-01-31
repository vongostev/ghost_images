# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:21:02 2021

@author: Pavel Gostev
"""
import __init__
import matplotlib.pyplot as plt
from gi.experiment import GIExpDataProcessor, ImgViewer

"""
settings_file -- путь к файлу с настройками эксперимента в формате json
"""
# settings_file = r'H:\SciData\GI\17_04_2019_obj_v\17_04_2019_obj_v.txt'
# settings_file = r'C:\Users\von.gostev\Downloads\17_04_2019_obj_v\17_04_2019_obj_v.txt'
# settings_file = r'H:\SciData\GI\110921\110921.txt'
settings_file = r'H:\SciData\GI\211221_computational\ghost_proector_12_15_scatt.txt'

analyser = GIExpDataProcessor(
    settings_file, n_images=3000, parallel_njobs=-2,
    parallel_reading=1, binning_order=5, use_cupy=False)
analyser.calculate_all()
# analyser.calculate_ghostimage()
# analyser.calculate_contrast()
# analyser.calculate_xycorr()
# analyser.calculate_timecorr()
# analyser.calculate_xycorr_widths(nx=5, ny=5)
# print(analyser.information)

# viewer = ImgViewer(analyser.ghost_data)
# viewer.accumulate(analyser.contrast_data)
# viewer.accumulate(analyser.xycorr_data)
# viewer.accumulate(analyser.ref_data[0])
# viewer.show(1)

plt.plot(analyser.times, analyser.timecorr_data)
plt.show()
plt.imshow(analyser.ghost_data)
plt.show()

plt.imshow(analyser.xycorr_data)
plt.show()
