# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:12:13 2021

@author: Pavel Gostev
"""
import sys
import time
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np
from collections import namedtuple

from lightprop2d import Beam2D
from .experiment import (
    find_images, get_ref_imgnum, GIExpDataProcessor,
    crop_shape, _using_dask, low_res)

from logging import Logger, StreamHandler, Formatter

log = Logger('EMUL')

handler = StreamHandler(sys.stdout)
handler.setLevel(10)
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

try:
    import cupy as cp
    _using_cupy = True
except ImportError as E:
    _using_cupy = False
    log.warn(
        f"{E}, 'use_cupy' and 'use_gpu' keys are meaningless.")

cached_ref_obj = {'ref': None, 'obj': None}


def generate_beams(area_size, npoints, wl,
                   init_field, init_field_gen, init_gen_args,
                   object_gen, object_gen_args,
                   z_obj, z_ref, use_gpu, use_cupy, binning_order,
                   *args, **kwargs):

    if cached_ref_obj['ref'] is None:
        ref = Beam2D(area_size, npoints, wl,
                     init_field=init_field,
                     init_field_gen=init_field_gen,
                     init_gen_args=init_gen_args, use_gpu=use_gpu,
                     complex_bits=64,
                     numpy_output=not use_cupy)
        cached_ref_obj['ref'] = ref
    else:
        ref = cached_ref_obj['ref']
        ref.z = 0
        if init_field_gen is not None:
            field = init_field_gen(ref.X, ref.Y, *init_gen_args)
        if init_field is not None:
            field = init_field.copy()
        ref._update_obj(field)

    if cached_ref_obj['obj'] is None:
        obj = Beam2D(area_size, npoints, wl, init_field=ref.field.copy(),
                     init_spectrum=ref.spectrum.copy(), use_gpu=use_gpu,
                     complex_bits=64,
                     numpy_output=not use_cupy)
        cached_ref_obj['obj'] = obj
    else:
        obj = cached_ref_obj['obj']
        obj.z = 0
        obj._update_obj(ref.field.copy(), ref.spectrum.copy())

    obj.propagate(z_obj)
    ref.propagate(z_ref)

    if object_gen is not None:
        obj.coordinate_filter(f_gen=object_gen, fargs=object_gen_args)

    return low_res(ref.iprofile, binning_order, obj.xp), low_res(obj.iprofile, binning_order, obj.xp)


def generate_data(self, i: int):
    ref_img, obj_img = \
        self.iprofiles_gen(self.area_size, self.npoints, self.wl,
                           self.init_field,
                           self.init_field_gen, self.init_gen_args,
                           self.object_gen, self.object_gen_args,
                           self.z_obj, self.z_ref,
                           self.use_gpu, self.use_cupy,
                           self.binning_order,
                           *self.iprofiles_gen_args)
    if self.use_backet:
        obj_data = self.backend.sum(obj_img)
    else:
        obj_data = \
            obj_img[self.npoints // 2, self.npoints // 2]
    self.ref_data[i] = ref_img
    self.obj_data[i] = obj_data


def generate_data_exp(self, i, path):
    init_field = get_ref_imgnum(path, self.settings)
    npoints = init_field.shape[0]
    ref_img, obj_img = \
        self.iprofiles_gen(self.area_size, npoints, self.wl,
                           init_field, None, (),
                           self.object_gen, self.object_gen_args,
                           self.z_obj, self.z_ref,
                           self.use_gpu, self.use_cupy,
                           self.binning_order,
                           *self.iprofiles_gen_args)
    if self.use_backet:
        obj_data = self.backend.sum(obj_img)
    else:
        obj_data = \
            obj_img[self.npoints // 2, self.npoints // 2]
    self.ref_data[i] = ref_img
    self.obj_data[i] = obj_data


@dataclass
class __GIEmulatorDefault:
    # Monkey patch to add non-default values
    area_size: float
    npoints: int
    wl: float


@dataclass
class GIEmulator(GIExpDataProcessor, __GIEmulatorDefault):
    # Monkey patch to supress GIExpDataProcessor non-default value
    settings_file: str = None
    # End of the monkey patch
    init_field: np.ndarray = None
    init_field_gen: object = None
    init_gen_args: tuple = ()

    z_obj: float = 0
    z_ref: float = 0

    object_gen: object = None
    object_gen_args: tuple = ()

    iprofiles_gen: object = generate_beams
    iprofiles_gen_args: tuple = ()

    use_backet: bool = True
    use_gpu: bool = False
    use_expdata: bool = False
    use_cupy: bool = False
    use_dask: bool = False

    expdata_dir: str = ''
    expdata_format: str = 'bmp'
    expdata_crop: list = (150, 360,	70,	280)
    binning_order: int = 1
    img_prefix: str = ''

    backend: object = np
    tcpoints: int = 10

    def __post_init__(self):
        """
        Клаcс предназначен для эмуляции эксперимента
        по наблюдению фантомных изображений в квазитепловом свете

        """
        if self.nimgs <= 0:
            raise ValueError(
                'Non-positive images number `nimgs` is prohibited in the emulation. Set `nimgs` > 0')

        self._set_backend()

        if self.backend == cp and not self.use_gpu:
            log.warn(
                f'{type(self).__name__}.use_cupy is True, {type(self).__name__}.use_gpu set to True')
            self.use_gpu = True

        SETS = namedtuple('settings',
                          ['TCPOINTS', 'REF_CROP', 'BINNING'])
        self.settings = SETS(
            TCPOINTS=self.tcpoints,
            REF_CROP=self.expdata_crop,
            BINNING=self.binning_order)

        if self.use_expdata:
            ny, nx = (crop_shape(
                self.settings.REF_CROP) // self.binning_order).astype(int)
            if nx != ny:
                raise ValueError(
                    f'Experimental speckles crop must be quadratic, not {nx}x{ny}')
            if nx != self.npoints:
                log.warn(
                    f'{type(self).__name__}.npoints is redefined from REF_CROP. npoints = {nx}')
                self.npoints = nx
            self.Ny = self.Nx = self.npoints
        else:
            self.Ny = self.Nx = self.npoints // self.binning_order

        self._allocate_data()
        self.ref_data = self.ref_data.astype(np.float32)

        log.info('Generating obj and ref data')
        log.info(f'Ref propagation distance is {self.z_ref} cm')
        log.info(f'Obj propagation distance is {self.z_obj} cm')
        t = time.time()

        if self.use_expdata:
            log.info(f'Using experimental profiles from {self.expdata_dir}')
            imgs_paths = find_images(
                self.expdata_dir, self.nimgs,
                self.expdata_format, self.img_prefix)
            for i in tqdm(range(len(imgs_paths)), position=0, leave=True):
                generate_data_exp(self, i, imgs_paths[i])
        else:
            log.info(
                f'Using profiles generated by `{self.init_field_gen.__name__}`')
            for i in tqdm(range(self.nimgs), position=0, leave=True):
                generate_data(self, i)
        print()
        log.info(
            f'Obj and ref data generated. Elapsed time {(time.time() - t):.3f} s')

        self.times = np.arange(self.tcpoints)
        self._allocate_outputs()
        self._make_blocked_ref_data()
