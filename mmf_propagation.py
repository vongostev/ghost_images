# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 16:47:15 2021

@author: vonGostev
"""
from lightprop2d import Beam2D, random_wave
import __init__
import pyMMF
import numpy as np
import matplotlib.pyplot as plt

# Parameters
NA = 0.4
radius = 25  # in microns
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
profile.initStepIndexMicrostructured(n1=1.45, a=2.5, core_pitch=8, NA=NA,
                                     dims=6, layers=2)
# profile.initStepIndexConcentric(core_offset=15, layers=1, a=5)
# profile.initStepIndex(n1=n1, a=5, NA=NA)
# Instantiate the solver
solver = pyMMF.propagationModeSolver()
# Set the profile to the solver
solver.setIndexProfile(profile)
# Set the wavelength
solver.setWL(wl)

plt.imshow(profile.n.reshape((npoints, npoints)))
plt.show()
# # Estimate the number of modes for a graded index fiber
Nmodes_estim = pyMMF.estimateNumModesSI(wl, 25, NA, pola=1)

r_max = 3.8*radius
# modes_radial = solver.solve(mode='radial',
#                             curvature=None,
#                             # max radius to calculate (and first try for large radial boundary condition)
#                             r_max=r_max,
#                             dh=profile.dh,  # radial resolution during the computation
#                             min_radius_bc=1.5,  # min large radial boundary condition
#                             change_bc_radius_step=0.99,  # change of the large radial boundary condition if fails
#                             N_beta_coarse=1000,  # number of steps of the initial coarse scan
#                             degenerate_mode='sin',
#                             field_limit_tol=0.01
#                             )
# modes_semianalytical = solver.solve(mode='SI', curvature=None)
modes_eig = solver.solve(nmodesMax=Nmodes_estim,
                         boundary='close',
                         mode='eig',
                         curvature=None,
                         propag_only=True)

# Sort the modes
modes = {}
# idx = np.flip(np.argsort(modes_semianalytical.betas), axis=0)
# modes['SA'] = {'betas': np.array(modes_semianalytical.betas)[idx], 'profiles': [
#     modes_semianalytical.profiles[i] for i in idx]}
idx = np.flip(np.argsort(modes_eig.betas), axis=0)
modes['eig'] = {'betas': np.array(modes_eig.betas)[idx], 'profiles': [
    modes_eig.profiles[i] for i in idx]}
# idx = np.flip(np.argsort(modes_radial.betas), axis=0)
# modes['radial'] = {'betas': np.array(modes_radial.betas)[idx], 'profiles': [
#     modes_radial.profiles[i] for i in idx]}


def sort(a):
    return np.flip(np.sort(a), axis=0)


# plt.figure()
# plt.plot(sort(np.real(modes_eig.betas)),
#          label='Numerical simulations (eigenvalue solver)',
#          linewidth=2.)
# plt.plot(sort(np.real(modes_semianalytical.betas)),
#           'r--',
#           label='Semi-analytical',
#           linewidth=2.)
# plt.plot(sort(np.real(modes_radial.betas)),
#          'go',
#          label='Numerical simulations (radial solver)',
#          linewidth=2.)

# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.title(r'Semi-analytical VS numerical simulations', fontsize=30)
# plt.ylabel(r'Propagation constant $\beta$ (in $\mu$m$^{-1}$)', fontsize=25)
# plt.xlabel(r'Mode index', fontsize=25)
# plt.legend(fontsize=22, loc='upper right')
# plt.show()

# imode = 15
# plt.figure()
# plt.subplot(121)
# plt.imshow(np.abs(modes['SA']['profiles'][imode].reshape([npoints]*2)))
# plt.gca().set_title("Ideal LP mode", fontsize=25)
# plt.axis('off')

# plt.subplot(122)
# plt.imshow(np.abs(modes['radial']['profiles'][imode].reshape([npoints]*2)))
# plt.gca().set_title("Numerical simulations", fontsize=25)
# plt.axis('off')

for solver_type in modes:
    fig, axes = plt.subplots(2*3, 6, figsize=(6, 3*3))
    for i, ax in enumerate(np.ravel(axes)):
        try:
            ax.imshow(np.abs(modes[solver_type]
                      ['profiles'][i]).reshape([npoints]*2))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        except:
            continue
    plt.show()

ibeam = Beam2D(area_size=area_size, npoints=npoints, wl=wl,
               init_field_gen=random_wave, unsafe_fft=0)
fiber_length = 50e4  # um
fiber_matrix = ibeam._xp(modes_eig.getPropagationMatrix(fiber_length))
modes_list = np.array(modes_eig.profiles)[np.argsort(modes_eig.betas)[::-1]]
modes_matrix = ibeam.xp.array(np.vstack(modes_list).T)
modes_matrix_t = modes_matrix.T
modes_matrix_dot_t = modes_matrix.T.dot(modes_matrix)
modes_coeffs = ibeam.fast_deconstruct_by_modes(
    modes_matrix_t, modes_matrix_dot_t)
plt.imshow(ibeam.iprofile)
plt.show()
ibeam.construct_by_modes(modes_list, fiber_matrix @ modes_coeffs)
plt.imshow(ibeam.iprofile)
plt.show()
ibeam.propagate(0.01)
plt.imshow(ibeam.iprofile)
plt.show()
ibeam.propagate(0.1)
plt.imshow(ibeam.iprofile)
plt.show()

ibeam.propagate(1)
plt.imshow(ibeam.iprofile)
plt.show()
