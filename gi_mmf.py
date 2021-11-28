# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:34:32 2021

@author: vonGostev
"""
import __init__
import numpy as np
import matplotlib.pyplot as plt

from lightprop2d import Beam2D, random_round_hole_phase, random_round_hole, rectangle_hole, um
from gi import ImgEmulator
from scipy.linalg import expm
from joblib import Parallel, delayed

# Parameters
radius = 31.25  # in microns
n1 = 1.45
wl = 0.632  # wavelength in microns

# calculate the field on an area larger than the diameter of the fiber
area_size = 3.5 * radius
npoints = 2**8  # resolution of the window
xp = np


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

    if z_obj > 0:
        obj.propagate(z_obj)
    if z_ref > 0:
        ref.propagate(z_ref)

    if object_gen is not None:
        obj.coordinate_filter(f_gen=object_gen, fargs=object_gen_args)

    return ref.iprofile, obj.iprofile


def calc_gi(fiber_props, ifgen):
    with np.load(fiber_props) as data:
        fiber_op = data["fiber_op"]
        modes = xp.array(data["modes_list"])

    fiber_len = 10 / um
    fiber_matrix = expm(1j * fiber_op * fiber_len)
    modes_matrix = xp.array(np.vstack(modes).T)
    modes_matrix_t = modes_matrix.T
    modes_matrix_dot_t = modes_matrix.T.dot(modes_matrix)
    # emulator = ImgEmulator(area_size*um, npoints,
    #                        wl*um, imgs_number=1000,
    #                        init_field_gen=random_round_hole,
    #                        init_gen_args=((radius - 1)*um,),
    #                        object_gen=rectangle_hole,
    #                        object_gen_args=(10*um, 50*um),
    #                        use_gpu=1
    #                        )

    # emulator.calculate_xycorr()
    # corr_before_fiber = emulator.xycorr_data

    emulator = ImgEmulator(area_size*um, npoints,
                           wl*um, imgs_number=10,
                           init_field_gen=ifgen,
                           init_gen_args=(radius*um,),
                           iprofiles_gen=generate_beams,
                           iprofiles_gen_args=(
                               modes, modes_matrix_t,
                               modes_matrix_dot_t, fiber_matrix),
                           object_gen=rectangle_hole,
                           object_gen_args=(10*um, 40*um),
                           use_gpu=0
                           )

    emulator.calculate_ghostimage()
    emulator.calculate_xycorr()
    return {'gi': emulator.ghost_data, 'sc': emulator.xycorr_data}


fiber_props_list = ["../rsf_report_1/mmf_SI_50_properties.npz",
                    "../rsf_report_1/mmf_GRIN_62.5_properties.npz"]
ifgen_list = [random_round_hole_phase, random_round_hole]
params_keys = ['SI__slm', 'GRIN__slm', 'SI__dmd', 'GRIN__dmd']

params = np.array(np.meshgrid(fiber_props_list, ifgen_list)).reshape((2, -1)).T

_fiber_data = Parallel(n_jobs=2)(delayed(calc_gi)(*p) for p in params)
fiber_data = {k: v for k, v in zip(params_keys, _fiber_data)}
np.savez_compressed('gi_data_grin_si.npz', fiber_data)
