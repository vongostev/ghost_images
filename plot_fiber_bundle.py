# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:29:23 2022

@author: vonGostev
"""

import __init__
import numpy as np
import numba as nb
from lightprop2d import round_hole

import matplotlib.pyplot as plt

from lightprop2d import Beam2D
import time

npoints = 1024
area_size = 120
wl0 = 0.532


def get_radial_structure(area_size: float, npoints: int, a: float = 1.,
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
    cores_coords = []
    for r, _a in zip(pos_radiuses, pos_angles):
        cores_coords.extend([
            [npoints // 2 + int(r * np.sin(t)),
             npoints // 2 + int(r * np.cos(t))]
            for t in _a])
    cores_coords = np.unique(np.array(cores_coords, dtype=np.int32), axis=0)
    return cores_coords, a


def get_randomized_radial_structure(area_size: float, npoints: int, a: float = 1.,
                                    dims: int = 4,
                                    layers: int = 1,
                                    core_pitch: float = 5,
                                    central_core_radius: float = 0):
    cc, a = get_radial_structure(
        area_size, npoints, a, dims, layers, core_pitch, central_core_radius)
    return cc + np.random.randint(-4, 5, size=cc.shape), a


def get_rectangle_structure(
        area_size: float, npoints: int, a: float,
        layers: tuple, core_pitch: float):

    dh = 1.*area_size/(npoints-1.)
    dn = int((2 * a + core_pitch) // dh)
    size_x = int(layers[0] * dn // 2)
    size_y = int(layers[1] * dn // 2)
    _n = npoints // 2

    coord_x = np.arange(
        _n - size_x, _n + size_x, dn, dtype=np.int32)
    coord_y = np.arange(
        _n - size_y, _n + size_y, dn, dtype=np.int32)
    cores_coords = np.array(
        [[y, x] for x in coord_x for y in coord_y])
    return cores_coords, a


@nb.njit(fastmath=True, nogil=True, cache=True)
def block_centers_and_dims(offset, npoints, layers, core_diameter, nblocks):
    block_layers = int(layers // nblocks)
    block_halfsize = int((npoints - 2 * offset + 2 * core_diameter) // (nblocks * 2))

    coords = np.arange(offset + block_halfsize, npoints -
                       offset, block_halfsize * 2)
    return np.array([[y, x] for x in coords for y in coords]), block_layers


@nb.njit(fastmath=True, nogil=True, cache=True, parallel=True)
def get_randomized_square_block(
        indx_center, core_diameter, core_pitch, layers):

    N = layers ** 2
    dn = core_diameter + core_pitch
    size_x = size_y = int((layers) * dn) + core_diameter
    cores_coords = np.zeros((N, 2), dtype=np.float32)
    i = 0

    for i in range(N):
        can_added = False
        while not can_added:
            x = np.random.randint(- size_x // 2 + core_pitch, size_x // 2 - core_pitch)
            y = np.random.randint(- size_y // 2 + core_pitch, size_y // 2 - core_pitch)
            p = np.asarray([x, y])
            can_added = True
            # pf = p.astype(np.float32)
            for p1 in cores_coords[:i+1]:
                if np.linalg.norm(p - p1) < core_diameter:
                    can_added = False
                    break
            if can_added:
                cores_coords[i] = p

    return cores_coords + indx_center


@nb.njit(fastmath=True, nogil=True, cache=True, parallel=True)
def get_randomized_square_structure(
        area_size: float, npoints: int, a: float,
        layers: int, core_pitch: float, nblocks: int = 1):

    N = layers ** 2
    dh = 1.*area_size/(npoints-1.)
    core_diameter = 2 * a // dh
    core_pitch = core_pitch // dh
    offset = int((npoints - (core_diameter + core_pitch) * layers - 2 *core_pitch) // 2)
    centers_block, layers_block = block_centers_and_dims(
        offset, npoints, layers, core_diameter, nblocks)
    n_block = layers_block ** 2
    cores_coords = np.zeros((N, 2), dtype=np.float32)
    for i, indx_center in enumerate(centers_block):
        coords_block = get_randomized_square_block(
             indx_center, core_diameter, core_pitch, layers_block)
        cores_coords[i * n_block: (i + 1) * n_block] = coords_block

    return cores_coords.astype(np.int64), a


def get_indxs_and_profile(
        cores_coords, core_radius, area_size, npoints, wl0):
    b = Beam2D(area_size, npoints, wl0,
               init_field=np.zeros((npoints, npoints)),
               use_gpu=False, complex_bits=64)
    _n = b.X.size // 2
    _nh = 32
    x = b.X[_n - _nh:_n + _nh]
    y = b.Y[_n - _nh:_n + _nh]
    core_profile = round_hole(x, y, core_radius)

    _cc = np.repeat(cores_coords, 2, axis=1)
    _cc[:, ::2] -= _nh
    _cc[:, 1::2] += _nh

    return b.X, b.Y, _cc, core_profile


@ nb.njit(fastmath=True, nogil=True, cache=True, parallel=True)
def fiber_bundle(X, Y, cc, profile, cores_coords):
    n = np.zeros((X.size, Y.size), dtype=np.complex64)
    N = len(cc)
    for k in range(N):
        li, ti, lj, tj = cc[k]
        n[li:ti, lj:tj] += profile

    return n


t = time.time()

# builders = get_radial_structure(
#     area_size,
#     npoints,
#     a=1.5,
#     core_pitch=0.25,
#     dims=6,
#     layers=14,
#     central_core_radius=1.5)
builders = get_randomized_square_structure(
    area_size, npoints, a=1.5, layers=20, core_pitch=0.25, nblocks=1)
pre_calcs = get_indxs_and_profile(
    *builders, area_size, npoints, wl0)
fbprofile = fiber_bundle(*pre_calcs, builders[0])
print(time.time() - t)

img = plt.imshow(np.abs(fbprofile))
img.set_cmap('gray')
plt.tight_layout()
plt.axis('off')
plt.show()
plt.savefig('fiber_bundle.png', dpi=300)
