# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:34:32 2021

@author: vonGostev
"""
import __init__
import pyMMF
import numpy as np
import matplotlib.pyplot as plt

from lightprop2d import Beam2D, random_round_hole, rectangle_hole
from gi import ImgEmulator

# Parameters
NA = 0.27
radius = 25  # in microns
n1 = 1.45
wl = 0.6328  # wavelength in microns

# calculate the field on an area larger than the diameter of the fiber
area_size = 3.5*radius
npoints = 2**7  # resolution of the window
fiber_length = 50e4  # um


def imshow(arr):
    plt.imshow(arr, extent=[-area_size / 2, area_size / 2] * 2)
    plt.xlabel(r'x, $\mu m$')
    plt.ylabel(r'y, $\mu m$')
    plt.show()


def generate_beams(area_size, npoints, wl,
                   init_field, init_field_gen, init_gen_args,
                   object_gen, object_gen_args,
                   z_obj, z_ref,
                   modes_profiles, fiber_matrix):

    obj = Beam2D(area_size, npoints, wl,
                 init_field=init_field,
                 init_field_gen=init_field_gen,
                 init_gen_args=init_gen_args)

    modes_coeffs = obj.deconstruct_by_modes(modes_profiles)
    obj.construct_by_modes(modes_profiles, fiber_matrix @ modes_coeffs)

    ref = Beam2D(area_size, npoints, wl, init_field=obj.xyfprofile)

    obj.propagate(z_obj)
    ref.propagate(z_ref)

    if object_gen is not None:
        obj.coordinate_filter(
            lambda x, y: object_gen(x, y, *object_gen_args))

    return ref.iprofile, obj.iprofile


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

modes_semianalytical = solver.solve(mode='SI', curvature=None)
# modes_eig = solver.solve(nmodesMax=500, boundary='close',
#                          mode='eig', curvature=None, propag_only=True)
modes_list = np.array(modes_semianalytical.profiles)[
    np.argsort(modes_semianalytical.betas)[::-1]]

fiber_matrix = modes_semianalytical.getPropagationMatrix(fiber_length)

emulator = ImgEmulator(area_size * 1e-4, npoints,
                       wl * 1e-4, imgs_number=1000, init_field_gen=random_round_hole,
                       init_gen_args=((radius - 1) * 1e-4,),
                       iprofiles_gen=generate_beams,
                       iprofiles_gen_args=(modes_list, fiber_matrix),
                       object_gen=rectangle_hole,
                       object_gen_args=(10e-4, 50e-4))
emulator.calculate_ghostimage()
emulator.calculate_xycorr()

imshow(emulator.ghost_data)
imshow(emulator.xycorr_data)
