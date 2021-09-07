# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:51:06 2021

@author: von.gostev
"""
import __init__
from gi.slm import slm_phaseprofile, slm_expand
from lightprop2d import Beam2D, gaussian_beam, mm, cm, um, round_hole
import pyMMF
import numpy as np
import matplotlib.pyplot as plt

npoints = 2 ** 9
area_size = 0.5*cm
beam_radius = 1*mm
wl = 0.632*um  # nm

# fiber
NA = 0.27
radius = 25  # um
n1 = 1.472


ibeam = Beam2D(area_size, npoints,
               wl, init_field_gen=gaussian_beam,
               init_gen_args=(1, beam_radius), use_gpu=1, unsafe_fft=True)

slm = slm_phaseprofile(area_size / npoints,
                       2 * np.pi * np.random.random(size=(100, 100)),
                       pixel_size=40e-4, pixel_gap=10e-4)
slm = slm_expand(slm, npoints)

# Is.append(ibeam.iprofile.get()[npoints // 2 - 6])
ibeam.coordinate_filter(f_init=slm)
plt.imshow(ibeam._np(ibeam.iprofile),
           extent=[-ibeam.area_size / 2 / um, ibeam.area_size / 2 / um]*2)
plt.show()
for i in range(3):
    ibeam.lens_image(10*mm, 40*mm, 10*mm)
    ibeam.crop(ibeam.area_size / 4)
    plt.imshow(ibeam._np(ibeam.iprofile),
               extent=[-ibeam.area_size / 2 / um, ibeam.area_size / 2 / um]*2)
    plt.show()

ibeam.coordinate_filter(f_gen=round_hole, fargs=(radius * um, ))
fbeam = Beam2D(ibeam.area_size, npoints,
               wl, init_field=ibeam._np(ibeam.field),
               use_gpu=0, unsafe_fft=True)

# Create the fiber object
profile = pyMMF.IndexProfile(npoints=npoints, areaSize=ibeam.area_size / um)
# Initialize the index profile
profile.initStepIndex(n1=n1, a=radius, NA=NA)
# Instantiate the solver
solver = pyMMF.propagationModeSolver()
# Set the profile to the solver
solver.setIndexProfile(profile)
# Set the wavelength
solver.setWL(wl / um)

modes_semianalytical = solver.solve(mode='SI', curvature=None,
                                    storeData=False, n_jobs=-2)
modes_list = np.array(modes_semianalytical.profiles)[
    np.argsort(modes_semianalytical.betas)[::-1]]

fiber_length = 50e4  # um
fiber_matrix = fbeam._xp(
    modes_semianalytical.getPropagationMatrix(fiber_length))
modes_matrix = fbeam._xp(np.vstack(modes_list).T)
modes_matrix_t = modes_matrix.T
modes_matrix_dot_t = modes_matrix.T.dot(modes_matrix)
modes_coeffs = fbeam.fast_deconstruct_by_modes(
    modes_matrix_t, modes_matrix_dot_t)
fbeam.construct_by_modes(modes_list, modes_coeffs)
plt.imshow(fbeam._np(fbeam.iprofile),
           extent=[-fbeam.area_size / 2 / um, fbeam.area_size / 2 / um]*2)
plt.show()
fbeam.construct_by_modes(modes_list, fiber_matrix @ modes_coeffs)
plt.imshow(fbeam._np(fbeam.iprofile),
           extent=[-fbeam.area_size / 2 / um, fbeam.area_size / 2 / um]*2)
plt.show()
