# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 00:03:40 2021

@author: vonGostev
"""

import __init__
import pyMMF
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from lightprop2d import Beam2D, random_round_hole, rectangle_hole, round_hole, um
from gi import ImgEmulator

# Parameters
NA = 0.2
radius = 25  # in microns
n1 = 1.45
wl = 0.632  # wavelength in microns

# calculate the field on an area larger than the diameter of the fiber
area_size = 5*radius
npoints = 2**8  # resolution of the window
fiber_length = 50e4  # um


def imshow(arr):
    plt.imshow(arr, extent=[-area_size / 2, area_size / 2] * 2)
    plt.xlabel(r'x, $\mu m$')
    plt.ylabel(r'y, $\mu m$')
    plt.show()


def generate_beams(area_size, npoints, wl,
                   init_field, init_field_gen, init_gen_args,
                   object_gen, object_gen_args,
                   z_obj, z_ref, use_gpu,
                   modes_profiles, modes_matrix_t, modes_matrix_dot_t,
                   fiber_matrix, ):
    obj = Beam2D(area_size, npoints, wl,
                 init_field=init_field,
                 init_field_gen=init_field_gen,
                 init_gen_args=init_gen_args, use_gpu=use_gpu)

    modes_coeffs = obj.fast_deconstruct_by_modes(
        modes_matrix_t, modes_matrix_dot_t)
    obj.construct_by_modes(modes_profiles, fiber_matrix @ modes_coeffs)

    ref = Beam2D(area_size, npoints, wl, init_field=obj.field, use_gpu=use_gpu)

    obj.propagate(z_obj)
    ref.propagate(z_ref)

    if object_gen is not None:
        obj.coordinate_filter(f_gen=object_gen, fargs=object_gen_args)

    return ref.iprofile.get(), obj.iprofile.get()


# Create the fiber object
profile = pyMMF.IndexProfile(npoints=npoints, areaSize=area_size)
# Initialize the index profile
profile.initStepIndex(n1=n1, a=radius, NA=NA)
# Instantiate the solver
solver = pyMMF.propagationModeSolver()
# Set the profile to the solver
solver.setIndexProfile(profile)
# Set the wavelength
solver.setWL(wl)

# Estimate the number of modes for a graded index fiber
Nmodes_estim = pyMMF.estimateNumModesSI(wl, radius, NA, pola=1)

xp = cp

try:
    with np.load("fiber_properties.npz") as data:
        fiber_matrix = xp.array(data["fiber_matrix"])
        modes_list = np.array(data["modes_list"])
except FileNotFoundError:
    modes = solver.solve(mode='SI', curvature=None)
    # modes_eig = solver.solve(nmodesMax=500, boundary='close',
    #                          mode='eig', curvature=None, propag_only=True)
    modes_list = np.array(modes.profiles)[np.argsort(modes.betas)[::-1]]
    fiber_matrix = xp.array(modes.getPropagationMatrix(fiber_length))
    np.savez_compressed("fiber_properties",
                        fiber_matrix=modes.getPropagationMatrix(fiber_length),
                        modes_list=modes_list)


modes_matrix = xp.array(np.vstack(modes_list).T)
modes_matrix_t = modes_matrix.T
modes_matrix_dot_t = modes_matrix.T.dot(modes_matrix)
emulator = ImgEmulator(area_size*um, npoints,
                       wl*um, imgs_number=1000,
                       init_field_gen=random_round_hole,
                       init_gen_args=((radius - 1)*um,),
                       object_gen=rectangle_hole,
                       object_gen_args=(10*um, 50*um),
                       use_gpu=1
                       )

emulator.calculate_xycorr()
corr_before_fiber = emulator.xycorr_data

emulator = ImgEmulator(area_size*um, npoints,
                       wl*um, imgs_number=1000,
                       init_field_gen=random_round_hole,
                       init_gen_args=((radius - 1)*um,),
                       iprofiles_gen=generate_beams,
                       iprofiles_gen_args=(
                           modes_list, modes_matrix_t,
                           modes_matrix_dot_t, fiber_matrix),
                       object_gen=rectangle_hole,
                       object_gen_args=(10*um, 50*um),
                       use_gpu=1
                       )

emulator.calculate_xycorr()
corr_after_fiber = emulator.xycorr_data

fig, ax = plt.subplots(1, 2)
ax[0].imshow(corr_before_fiber, extent=(-62.5,
             62.5, -62.5, 62.5), cmap=plt.cm.Greys_r)
ax[0].set_xlabel('x, um')
ax[0].set_ylabel('y, um')
ax[1].set_xlabel('x, um')
ax[1].imshow(corr_after_fiber, extent=(-62.5,
             62.5, -62.5, 62.5), cmap=plt.cm.Greys_r)
plt.tight_layout()
plt.show()
