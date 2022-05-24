# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:49:16 2022

@author: von.gostev
"""
import numpy as np
import matplotlib.pyplot as plt


simdata = np.load('fs_simdata_200121.npz', allow_pickle=True)
for method in simdata:
    print(method)
    mdata = simdata[method].tolist()
    for z in mdata:
        #print(z)
        if int(z) > 100:
            continue
        data = mdata[z]
        for nimg in data:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(data[nimg]['ip'])
            axes[0].set_title(f'Intensity Profile on z={z} um')
            axes[1].imshow(data[nimg]['cs'])
            axes[1].set_title(f'CorrFun on z={z} um')
            axes[2].imshow(data[nimg]['gi'])
            axes[2].set_title(f'Test GI on z={z} um')
            plt.savefig(f'dataz{z}um_170522.png', dpi=300)
            plt.show()