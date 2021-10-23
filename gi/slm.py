# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 22:50:20 2021

@author: von.gostev
"""
import numpy as np
import numba as nb


@nb.njit(parallel=True)
def scale(mask, kx, ky):     # fill A with B scaled by k
    Y = mask.shape[0] * ky
    X = mask.shape[1] * kx
    scaled = np.empty((Y, X))
    for y in nb.prange(0, ky):
        for x in range(0, kx):
            scaled[y:Y:ky, x:X:kx] = mask
    return scaled


@nb.njit(parallel=True, nogil=True)
def slm_grid(dL, pixel_size=10e-4, pixel_gap=4e-5, shape=(1440, 1050),
             angle=0):
    ngap = int(pixel_gap // dL)
    npixel = int(pixel_size // dL)

    def angled(x): return int(x * np.cos(angle))

    grid_y = np.zeros((shape[0] * (npixel + ngap) - ngap, 1))
    grid_x = np.zeros(shape[1] * angled(npixel + ngap) - angled(ngap))
    for i in nb.prange(max(shape)):
        pixel_offset_y = (npixel + ngap) * i
        grid_y[pixel_offset_y:pixel_offset_y + npixel, :] = 1
        # grid_y[pixel_offset, :] = 0.5
        # grid_y[pixel_offset + npixel, :] = 0.5
        pixel_offset_x = int((npixel + ngap) * np.cos(angle)) * i
        grid_x[pixel_offset_x:pixel_offset_x + angled(npixel)] = 1
        # grid_x[pixel_offset] = 0.5
        # grid_x[pixel_offset + npixel] = 0.5

    return grid_y * grid_x


def slm_phaseprofile(dL, pixel_map, pixel_size=10e-4, pixel_gap=4e-5,
                     angle=0):
    shape = pixel_map.shape
    ngap = int(pixel_gap // dL)
    npixel = int(pixel_size // dL)

    grid = slm_grid(
        dL, pixel_size, pixel_gap, shape, angle).astype(np.complex128)
    phase_mask = scale(pixel_map, int(max(1, npixel + ngap) * np.cos(angle)),
                       max(1, npixel + ngap))
    if ngap > 0:
        phase_mask = phase_mask[:-ngap, :]
    phase_mask = np.cos(phase_mask) + 1j * np.sin(phase_mask)
    grid *= phase_mask
    grid[grid == 0] = 1
    return grid


def slm_expand(phase_profile, npoints):
    py, px = phase_profile.shape
    xo = (npoints - px) // 2
    yo = (npoints - py) // 2
    if px >= npoints and py >= npoints:
        yo = abs(yo)
        xo = abs(yo)
        return phase_profile[yo:yo+npoints, xo:xo+npoints]

    _profile = np.ones((npoints, npoints), dtype=np.complex128)
    _profile[yo:yo+npoints, xo:xo+npoints] = phase_profile
    return _profile
