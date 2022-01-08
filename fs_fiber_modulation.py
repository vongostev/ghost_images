# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:22:29 2021

@author: von.gostev
"""
import __init__
import numpy as np
from lightprop2d import Beam2D, um, cm

import matplotlib.pyplot as plt
from gi.emulation import ImgEmulator
import numba as nb

npoints = 1024
area_size = 400
wl0 = 0.632


@nb.njit(fastmath=True, nogil=True, cache=True)
def gaussian_beam(x, y, A0, rho0, x0=0, y0=0):
    x -= x0
    y -= y0
    return A0 * np.exp(- (x ** 2 + y ** 2) / 2 / rho0 ** 2)


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


@nb.njit(fastmath=True, nogil=True, cache=True, parallel=True)
def random_fbundle(X, Y, amplitudes, phases, cores_coords, core_radius):
    cores_num = len(cores_coords)
    amplitudes = np.ones((cores_num,))#np.random.uniform(0, 1, size=(cores_num,))
    phases = np.random.uniform(0, 2*np.pi, size=(cores_num,))#np.zeros(cores_num)
    n = np.zeros((X.size, Y.size), dtype=np.complex128)
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


z_refs = [0, 100, 500, 1000]
simdata = {}

nimg = 1000

if __name__ == "__main__":
    builders = get_builders(
        area_size,
        npoints,
        a=4.2,
        core_pitch=1,
        dims=6,
        layers=14,
        central_core_radius=4.2)

    # x = np.linspace(-area_size/2, area_size/2, npoints)
    cores_num = len(builders[0])

    # iprofile = random_fbundle(x, x.reshape(
    #     (-1, 1)), amplitudes, phases, *builders)
    amplitudes = np.random.uniform(0, 1, size=(cores_num,))
    phases = np.random.uniform(-np.pi, np.pi, size=(cores_num,))

    for z in z_refs:
        # beam = Beam2D(area_size, npoints, wl0,
        #               init_field_gen=random_fbundle,
        #               init_gen_args=(amplitudes, phases, *builders))
        # beam.propagate(z)
        # plt.imshow(beam.iprofile)
        # plt.show()

        test = ImgEmulator(area_size, npoints, wl0, nimg,
                           init_field_gen=random_fbundle,
                           init_gen_args=(amplitudes, phases, *builders),
                           z_ref=z,
                           parallel_njobs=-1,
                           fast_corr=0)
        test.calculate_xycorr()

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(test.ref_data[0])
        axes[1].imshow(test.xycorr_data)
        plt.show()

        z = f"{z}"
        simdata[z] = {}
        simdata[z]['ip'] = test.ref_data[0]
        simdata[z]['cs'] = test.xycorr_data

np.savez_compressed('fs_simdata_phase_random_stable_i', **simdata)
