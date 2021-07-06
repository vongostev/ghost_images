# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 16:47:15 2021

@author: vonGostev
"""
import __init__
import pyMMF
import numpy as np
import matplotlib.pyplot as plt

# Parameters
NA = 0.27
radius = 10  # in microns
n1 = 1.45
wl = 0.6328  # wavelength in microns

# calculate the field on an area larger than the diameter of the fiber
area_size = 3.5*radius
npoints = 2**7  # resolution of the window


def plot_i(ibeam):
    area_size = ibeam.area_size
    plt.imshow(ibeam.iprofile,
               extent=[-area_size / 2e-4, area_size / 2e-4] * 2)
    plt.xlabel(r'x, $\mu m$')
    plt.ylabel(r'y, $\mu m$')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_modes(modes_coeffs):
    plt.plot(np.real(modes_coeffs))
    plt.plot(np.imag(modes_coeffs))
    plt.xlabel('Mode number')
    plt.ylabel('Coefficient')
    plt.title('Modes series before the fiber')
    plt.tight_layout()
    plt.show()


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
# modes_eig = solver.solve(nmodesMax=501, boundary='close',
#                          mode='eig', curvature=None, propag_only=True)
modes_list2 = np.array(modes_semianalytical.profiles)[
    np.argsort(modes_semianalytical.betas)[::-1]]

# fiber_length = 50e4  # um
# fiber_matrix = modes_semianalytical.getPropagationMatrix(fiber_length)

# fig, axes = plt.subplots(4, 9, figsize=(20, 10))
# for i, ax in enumerate(np.ravel(axes)):
#     ax.imshow(np.abs(modes_list[i]).reshape((npoints, npoints)))
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
# plt.show()
