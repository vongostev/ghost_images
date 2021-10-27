# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 23:18:02 2021

@author: vonGostev
"""
import __init__
from gi.slm import slm_phaseprofile, slm_expand
from lightprop2d import Beam2D, gaussian_beam, mm, cm, um, rectangle_hole, round_hole
import numpy as np
import matplotlib.pyplot as plt
import pyMMF
import cupy as cp

npoints = 2 ** 10
area_size = 3000*um
beam_radius = area_size / 8
wl = 0.632*um  # nm
fiber_length = 50e4  # um

img_number = 10
z = 1

angles = np.pi * np.arange(0, 3) / 6

rng = np.random.default_rng(seed=1)

# Parameters
NA = 0.2
radius = 25*um  # in microns
n1 = 1.45
farea_size = radius * 3.5


def slm_random_gaussbeam(angle, z, farea_size, fradius):
    ibeam = Beam2D(area_size, npoints,
                   wl, init_field_gen=gaussian_beam,
                   init_gen_args=(1, beam_radius), use_gpu=1)

    slm = slm_phaseprofile(area_size / npoints,
                           np.pi * 0.999 * rng.integers(0, 2, size=(400, 400)),
                           # rng.uniform(0, 2 * np.pi, size=(400, 400)),
                           pixel_size=20e-4, pixel_gap=0, angle=angle)
    slm = slm_expand(slm, npoints)
    ibeam.coordinate_filter(f_init=slm)
    ibeam.propagate(z)
    # ibeam.lens_image(1, 4, 1)
    # ibeam.crop(farea_size)
    # ibeam.coordinate_filter(f_gen=round_hole, fargs=(fradius,))
    return ibeam


speckles = [slm_random_gaussbeam(
    angle, z, farea_size, radius) for angle in angles]

x_extent = [-area_size / 2 * 10, area_size / 2 * 10]

fig, ax = plt.subplots(1, len(speckles), figsize=(10, 4))
for i, sp in enumerate(speckles):
    ax[i].imshow(sp.iprofile.get(), origin='lower',
                 extent=x_extent * 2)
    ax[i].set_title(f'{np.rad2deg(angles[i]):.0f} deg.')
    ax[i].set_xticks(np.linspace(*x_extent, 7))
    ax[i].set_xlabel('x, mm')
    ax[i].set_ylabel('y, mm')
fig.suptitle(
    'Профиль интенсивности в зависимости от угла падения на SLM', fontsize=16)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, len(speckles), figsize=(10, 4))
for i, sp in enumerate(speckles):
    ax[i].imshow(sp.phiprofile.get(), origin='lower',
                 extent=x_extent * 2)
    ax[i].set_title(f'{np.rad2deg(angles[i]):.0f} deg.')
    ax[i].set_xticks(np.linspace(*x_extent, 7))
    ax[i].set_xlabel('x, mm')
    ax[i].set_ylabel('y, mm')
fig.suptitle('Профиль фазы в зависимости от угла падения на SLM', fontsize=16)
plt.tight_layout()
plt.show()

# # Create the fiber object
# profile = pyMMF.IndexProfile(npoints=npoints, areaSize=area_size)
# # Initialize the index profile
# profile.initStepIndex(n1=n1, a=radius, NA=NA)
# # Instantiate the solver
# solver = pyMMF.propagationModeSolver()
# # Set the profile to the solver
# solver.setIndexProfile(profile)
# # Set the wavelength
# solver.setWL(wl)

# # Estimate the number of modes for a graded index fiber
# Nmodes_estim = pyMMF.estimateNumModesSI(wl, radius, NA, pola=1)

# xp = cp

# try:
#     with np.load("fiber_properties.npz") as data:
#         fiber_matrix = xp.array(data["fiber_matrix"])
#         modes_list = np.array(data["modes_list"])
# except FileNotFoundError:
#     modes = solver.solve(mode='SI', curvature=None)
#     # modes_eig = solver.solve(nmodesMax=500, boundary='close',
#     #                          mode='eig', curvature=None, propag_only=True)
#     modes_list = np.array(modes.profiles)[np.argsort(modes.betas)[::-1]]
#     fiber_matrix = xp.array(modes.getPropagationMatrix(fiber_length))
#     np.savez_compressed("fiber_properties",
#                         fiber_matrix=modes.getPropagationMatrix(fiber_length),
#                         modes_list=modes_list)


# modes_matrix = xp.array(np.vstack(modes_list).T)
# modes_matrix_t = modes_matrix.T
# modes_matrix_dot_t = modes_matrix.T.dot(modes_matrix)

# for i, sp in enumerate(speckles):
#     modes_coeffs = sp.fast_deconstruct_by_modes(modes_matrix_t, modes_matrix_dot_t)
#     sp.construct_by_modes(modes_list, fiber_matrix @ modes_coeffs)

# fig, ax = plt.subplots(1, len(speckles), figsize=(10, 4))
# for i, sp in enumerate(speckles):
#     ax[i].imshow(sp.iprofile.get())
#     ax[i].set_title(f'{np.rad2deg(angles[i]):.0f} deg.')
#     ax[i].set_xticks(np.arange(0, npoints + 1, 100))
# fig.suptitle('Speckles after fiber by angle of incidence', fontsize=16)
# plt.show()
