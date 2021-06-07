# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:45:41 2021

@author: vonGostev
"""
import time


from gi.emulation import ImgEmulator
from gi.experiment import ImgViewer


"""
Программа запускается из Spyder через F5 без дополнительных аргументов
Здесь

settings_file -- путь к файлу с настройками эксперимента в формате json
"""

t = time.time()


analyser = ImgEmulator(100, 128, 128)
analyser.correlate()
# analyser.contrast()

analyser.spatial_coherence()
# analyser.time_coherence()
print(time.time() - t)

# print(analyser.information())
# analyser.save_pictures()
# analyser.save_information()

viewer = ImgViewer(analyser.ghost_data)
# viewer.accumulate(analyser.cd)

# viewer.accumulate(analyser.ref_data[0])
viewer.show(1)
