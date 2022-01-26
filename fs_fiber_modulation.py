# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:22:29 2021

@author: von.gostev
"""
import __init__
import numpy as np
from lightprop2d import Beam2D, um, cm, rectangle_hole, gaussian_beam, round_hole

import matplotlib.pyplot as plt
from gi.emulation import GIEmulator
import cupy as cp

npoints = 512
area_size = 150
wl0 = 0.532


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
    return np.unique(cores_coords, axis=0), a


def random_fbundle(X, Y, cores_num, cores_coords, core_radius, method='a', backend=cp):
    if method == 'a':
        amplitudes = backend.random.uniform(
            0, 1, size=(cores_num,), dtype=np.float32)
        phases = backend.zeros(cores_num)
    elif method == 'p':
        amplitudes = 1.
        phases = backend.random.uniform(
            0, np.pi, size=(cores_num,), dtype=np.float32)
    elif method == 'ap':
        amplitudes = backend.random.uniform(
            0, 1, size=(cores_num,), dtype=np.float32)
        phases = backend.random.uniform(
            0, np.pi, size=(cores_num,), dtype=np.float32)
    else:
        raise ValueError(f'Unknowm method `{method}`')
    n = backend.zeros((X.size, Y.size), dtype=np.complex64)
    k = 0
    _n = X.size // 2
    _nh = 32
    x = X[_n - _nh:_n + _nh]
    y = Y[_n - _nh:_n + _nh]
    gaussian = gaussian_beam(
        x, y, 1, 0.5829260426150318) * round_hole(x, y, 1.5 * core_radius)
    mod = amplitudes * backend.exp(1j * phases)
    # mod[backend.abs(mod) ** 2 > 1] = 1
    gauss_profiles = backend.tensordot(mod, gaussian, axes=0)
    gauss_profiles = gauss_profiles.reshape((cores_num, 2 * _nh, 2 * _nh))
    for indxs in cores_coords:
        i, j = indxs
        n[i - _nh:i + _nh, j - _nh:j + _nh] += gauss_profiles[k]
        k += 1
    return n


z_refs = [0]
simdata = {}

nimgs = [100]
methods = ['a']


if __name__ == "__main__":
    builders = get_builders(
        area_size,
        npoints,
        a=1.5,
        core_pitch=0.5,
        dims=6,
        layers=14,
        central_core_radius=1.5)

    cores_num = len(builders[0])

    for method in methods:
        simdata[method] = {}
        for z in z_refs:
            simdata[method][z] = {}
            for nimg in nimgs:
                simdata[method][z][nimg] = {}
                test = GIEmulator(area_size, npoints, wl0, nimg,
                                  init_field_gen=random_fbundle,
                                  init_gen_args=(
                                      cores_num, *builders, method, cp),
                                  z_ref=z,
                                  object_gen=rectangle_hole,
                                  object_gen_args=(40, 40),
                                  parallel_njobs=1,
                                  use_gpu=True,
                                  use_cupy=True)
                test.calculate_timecorr()
                # test.calculate_xycorr()
                # test.calculate_xycorr_widths(nx=50, ny=50, window_points=10)
                # test.calculate_ghostimage()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(test.ref_data[0].get())
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

                # del test

    # np.savez_compressed('fs_simdata_200121', **simdata)
