# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:22:29 2021

@author: von.gostev
"""
import __init__
import numpy as np
from lightprop2d import Beam2D, um, cm, rectangle_hole, gaussian_beam

import matplotlib.pyplot as plt
from gi.emulation import ImgEmulator
import numba as nb
import cupy as cp

npoints = 1024
area_size = 400
wl0 = 0.632


def get_builders(area_size, npoints, a: float = 1.,
                 dims: int = 4,
                 layers: int = 1,
                 core_pitch: float = 5,
                 central_core_radius: float = 0):

    dh = 1.*area_size/(npoints-1.)

    # A grid for layers
    lgrid = np.arange(layers + 1)
    # Angles for positions of cores in one layer
    pos_angles = [
        np.arange(0, 2 * np.pi, 2 * np.pi / dims / lnum) for lnum in lgrid]
    # Radius-vector modules of cores centrums
    pos_radiuses = lgrid * int((core_pitch + 2 * a) // dh)
    pos_radiuses[1:] += int((central_core_radius - a) // dh)
    # Coordinates of cores as all combinations of radiuses and angles
    cores_coords = [[
        [npoints // 2 + int(r * np.sin(t)),
         npoints // 2 + int(r * np.cos(t))]
        for t in _a] for r, _a in zip(pos_radiuses, pos_angles)]
    cores_coords = sum(cores_coords, [])
    # cores_radiuses = [central_core_radius] + [a] * (len(cores_coords) - 1)
    return np.array(cores_coords), a


def random_fbundle(X, Y, cores_num, cores_coords, core_radius, method='a'):
    if method == 'a':
        amplitudes = cp.random.randint(
            0, 256, size=(cores_num,)).astype(np.float64)
        phases = cp.zeros(cores_num)
    elif method == 'p':
        amplitudes = cp.ones(cores_num)
        phases = cp.random.uniform(0, 2*np.pi, size=(cores_num,))
    elif method == 'ap':
        amplitudes = cp.random.randint(
            0, 256, size=(cores_num,)).astype(np.float64)
        phases = cp.random.uniform(0, 2*np.pi, size=(cores_num,))

    n = cp.zeros((X.size, Y.size), dtype=np.complex128)
    k = 0
    _n = X.size // 2
    _nh = _n // 4
    gaussian = gaussian_beam(
        X[_n - _nh:_n + _nh],
        Y[_n - _nh:_n + _nh], 1, core_radius / 4)

    for indxs in cores_coords:
        i, j = indxs
        n[i - _nh:i + _nh, j - _nh:j + _nh] += amplitudes[k] * \
            np.exp(1j * phases[k]) * gaussian
        k += 1

    return n


z_refs = [0]
simdata = {}

nimgs = [1000]
methods = ['a']


if __name__ == "__main__":
    builders = get_builders(
        area_size,
        npoints,
        a=4.2,
        core_pitch=0.1,
        dims=6,
        layers=16,
        central_core_radius=4.2)

    cores_num = len(builders[0])

    for method in methods:
        simdata[method] = {}
        for z in z_refs:
            simdata[method][z] = {}
            for nimg in nimgs:
                simdata[method][z][nimg] = {}
                test = ImgEmulator(area_size, npoints, wl0, nimg,
                                   init_field_gen=random_fbundle,
                                   init_gen_args=(cores_num, *builders),
                                   z_ref=z,
                                   object_gen=rectangle_hole,
                                   object_gen_args=(50, 200),
                                   parallel_njobs=1,
                                   use_gpu=True,
                                   fast_corr=False)
                test.calculate_xycorr()
                test.calculate_ghostimage()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(test.ref_data[0])
                axes[0].set_title(f'Intensity Profile on z={z} um')
                axes[1].imshow(test.xycorr_data)
                axes[1].set_title(f'CorrFun on z={z} um')
                axes[2].imshow(test.ghost_data)
                axes[2].set_title(f'Test GI on z={z} um')
                plt.savefig(
                    f'data_z{z}um_nimg{nimg}_method_{method}.png', dpi=300)
                plt.show()

                simdata[method][z][nimg]['ip'] = test.ref_data[0]
                simdata[method][z][nimg]['cs'] = test.xycorr_data
                simdata[method][z][nimg]['gi'] = test.ghost_data

    # np.savez_compressed('fs_simdata_190121_a', **simdata)
