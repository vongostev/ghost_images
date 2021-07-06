# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:40:57 2021

@author: vonGostev
"""
import __init__
import pyMMF
import types
import numpy as np
import matplotlib.pyplot as plt

# Parameters
NA = 0.2
radius = 25.  # in microns
# calculate the field on an area larger than the diameter of the fiber
areaSize = 3.5*radius
npoints = 2**8  # resolution of the window
n1 = 1.472
wl = 0.678  # wavelength in microns


# profile = pyMMF.IndexProfile(npoints=npoints, areaSize=areaSize)

# # Initialize the index profile
# # profile.initStepIndex(n1=n1, a=radius, NA=NA)
# profile.initParabolicGRIN(n1=n1, a=radius, NA=NA)
# # Instantiate the solver
# solver = pyMMF.propagationModeSolver()
# # Set the profile to the solver
# solver.setIndexProfile(profile)
# # Set the wavelength
# solver.setWL(wl)
# # Estimate the number of modes for a graded index fiber
# Nmodes_estim = pyMMF.estimateNumModesSI(wl, radius, NA, pola=1)

# print(f"Estimated number of modes using the V number = {Nmodes_estim}")

# modes = solver.solve(mode='radial',
#                      propag_only=True,
#                      N_beta_coarse=int(1e3),
#                      curvature=None)

# fig, axes = plt.subplots(5, 5, figsize=(10, 10))
# for i, ax in enumerate(np.ravel(axes)):
#     try:
#         field = modes.profiles[i].reshape([npoints]*2)
#         ax.imshow(np.log10(np.abs(field)))
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#     except:
#         continue
# plt.show()
