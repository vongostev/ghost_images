# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:52:13 2021

@author: von.gostev
"""
import time
import os, sys
sys.path.append(os.path.abspath('../..'))

import matplotlib.pyplot as plt
import Code.pyMMF.pyMMF as new
import pyMMF as old
import numpy as np

npoints = 2 ** 8
wl = 0.632

# fiber
NA = 0.27
n1 = 1.472

perf_data = {old: {'name': 'wavefrontshaping/pyMMF.git'}, 
             new: {'name': 'vongostev/pyMMF.git'}}
for pyMMF in [new, old]:
    perf_data[pyMMF]['modes'] = []
    perf_data[pyMMF]['times'] = []
    perf_data[pyMMF]['profiles'] = []
    for radius in range(5, 15):
        area_size = 3.5 * radius
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
        t = time.time()
        modes_semianalytical = solver.solve(
            mode='SI', curvature=None, n_jobs=-1)
        t = time.time() - t
        perf_data[pyMMF]['modes'].append(modes_semianalytical.number)
        perf_data[pyMMF]['times'].append(t)
        perf_data[pyMMF]['profiles'].append(modes_semianalytical.profiles)
     

plt.subplots(1, 1, figsize=(8,6))
for pyMMF in [new, old]:
    plt.plot(perf_data[pyMMF]['modes'], perf_data[pyMMF]['times'], 'o-',
             label=perf_data[pyMMF]['name'])
plt.legend(frameon=False)
plt.ylabel('Calculation time, s')
plt.xlabel('Modes number')
plt.show()

for i in range(len(perf_data[old]['profiles'])):
    oldp = perf_data[old]['profiles'][i]
    newp = perf_data[new]['profiles'][i]
    pn = len(oldp)
    print(i + 5, 'um', 'Profiles match:', 
          np.sum([np.allclose(newp[k], oldp[k]) for k in range(pn)]),
          'from', pn)
