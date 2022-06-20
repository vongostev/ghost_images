# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:22:29 2021

@author: von.gostev
"""
import __init__
import numpy as np
import cv2
from lightprop2d import rectangle_hole, gaussian_beam, round_hole

import matplotlib.pyplot as plt
from gi.emulation import GIEmulator, log
import cupy as cp

from lightprop2d import Beam2D
from fiber_bundle import get_randomized_center_square_structure, get_radial_structure, mira_mask


def get_indxs_and_profile(
        cores_coords, core_radius, area_size, npoints, wl0, backend=cp):
    b = Beam2D(area_size, npoints, wl0,
               init_field=backend.zeros((npoints, npoints)),
               use_gpu=True, complex_bits=64, numpy_output=False)
    _n = b.X.size // 2
    _nh = 32
    x = b.X[_n - _nh:_n + _nh]
    y = b.Y[_n - _nh:_n + _nh]
    gaussian = round_hole(x, y, core_radius) * gaussian_beam(
        x, y, 1, 0.5829260426150318)

    _cc = np.repeat(cores_coords, 2, axis=1)
    _cc[:, ::2] -= _nh
    _cc[:, 1::2] += _nh

    return _cc, gaussian


def random_fbundle(X, Y, cores_num, cores_coords, core_radius, cc, gaussian,
                   noised=False, method='a', backend=cp):
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

    if noised:
        n = backend.random.uniform(
            0, 1e-5, (X.size, Y.size), dtype=backend.float32)
        n = n.astype(backend.complex64)
    else:
        n = backend.zeros((X.size, Y.size), dtype=backend.complex64)

    mod = amplitudes * (backend.cos(phases) + 1j * backend.sin(phases))

    for k in range(len(cc)):
        li, ti, lj, tj = cc[k]
        n[li:ti, lj:tj] += gaussian * mod[k]

    return n


z_refs = [10, 20, 30] #[*np.arange(0, 51, 2), 100, 150, 200, 250][:1]
xyc_widths = []
simdata = {}

nimgs = [1024]
methods = ['ap']
npoints = 1024
area_size = 160
wl0 = 0.83

if __name__ == "__main__":
    # builders = get_radial_structure(
    #     area_size,
    #     npoints,
    #     a=1.5,
    #     core_pitch=0.25,
    #     dims=6,
    #     layers=7,
    #     central_core_radius=1.5)
    builders = get_randomized_center_square_structure(
        area_size, npoints, a=2.5, layers=20, core_pitch=0.25)
    log.info("Fiber structure created")
    pre_calcs = get_indxs_and_profile(
        *builders, area_size, npoints, wl0, backend=cp)
    log.info("Pre-calculations completed")

    cores_num = len(builders[0])

    for method in methods:
        simdata[method] = {}
        for z in z_refs:
            simdata[method][z] = {}
            for nimg in nimgs:
                simdata[method][z][nimg] = {}
                test = GIEmulator(area_size, npoints, wl0, nimgs=nimg,
                                  init_field_gen=random_fbundle,
                                  init_gen_args=(
                                      cores_num, *builders, *pre_calcs, False, method, cp),
                                  z_ref=z,
                                  object_gen=mira_mask,
                                  parallel_njobs=1,
                                  binning_order=16,
                                  use_gpu=True,
                                  use_cupy=True,
                                  use_dask=False)
                # test.calculate_timecorr()
                test.calculate_xycorr()  # window_points=128)
                test.calculate_ghostimage()

                fig, axes = plt.subplots(1, 3, figsize=(16, 6))
                axes[0].imshow(test._np(test.ref_data[0]))
                axes[0].set_title(
                    f'Intensity Profile on z={z} um', fontsize=16)
                axes[1].imshow(test.xycorr_data)
                axes[1].set_title(f'CorrFun on z={z} um', fontsize=16)
                axes[2].imshow(test.ghost_data)
                axes[2].set_title(f'Test GI on z={z} um')
                plt.savefig(
                    f'data_z{z}um_nimg{nimg}_method_{method}_radial.png', dpi=300)
                plt.show()

                xyc_widths.append(test.xycorr_width)

                # simdata[method][z][nimg]['ip'] = test.ref_data[0]
                # simdata[method][z][nimg]['cs'] = test.xycorr_data
                # simdata[method][z][nimg]['gi'] = test.ghost_data

                # test.calculate_xycorr_widths(nx=50, ny=50, window_points=128)

                del test

    # np.savez_compressed('fs_simdata_200121', **simdata)
    z_refs[0] = 1
    plt.plot(z_refs, [x.get()[0] for x in xyc_widths], 'o-')
    plt.semilogx(z_refs, [x.get()[1] for x in xyc_widths], 'o-')
    plt.xlabel("Distance, um")
    plt.ylabel("CF width")
    plt.show()
