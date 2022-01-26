# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:12:13 2021

@author: vonGostev
"""
import sys
import time
from tqdm import tqdm
from dataclasses import dataclass
from joblib import Parallel, delayed, wrap_non_picklable_objects
import numpy as np
import cupy as cp
from collections import namedtuple

from lightprop2d import Beam2D
from .experiment import find_images, get_ref_imgnum, GIExpDataProcessor

from logging import Logger, StreamHandler, Formatter

log = Logger('EMUL')

handler = StreamHandler(sys.stdout)
handler.setLevel(10)
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)


def generate_beams(area_size, npoints, wl,
                   init_field, init_field_gen, init_gen_args,
                   object_gen, object_gen_args,
                   z_obj, z_ref, use_gpu, use_cupy, *args, **kwargs):

    ref = Beam2D(area_size, npoints, wl,
                 init_field=init_field,
                 init_field_gen=init_field_gen,
                 init_gen_args=init_gen_args, use_gpu=use_gpu,
                 complex_bits=64,
                 numpy_output=not use_cupy)
    obj = Beam2D(area_size, npoints, wl, init_field=ref.field.copy(),
                 init_spectrum=ref.spectrum.copy(), use_gpu=use_gpu,
                 complex_bits=64,
                 numpy_output=not use_cupy)
    if object_gen is not None:
        obj.coordinate_filter(
            f_gen=lambda x, y: object_gen(x, y, *object_gen_args))

    obj.propagate(z_obj)
    ref.propagate(z_ref)

    return ref.iprofile, obj.iprofile


@wrap_non_picklable_objects
def generate_data(self, i: int):
    ref_img, obj_img = \
        self.iprofiles_gen(self.area_size, self.npoints, self.wl,
                           self.init_field, self.init_field_gen, self.init_gen_args,
                           self.object_gen, self.object_gen_args,
                           self.z_obj, self.z_ref,
                           self.use_gpu, self.use_cupy, *self.iprofiles_gen_args)
    if self.use_backet:
        obj_data = self.xp.sum(obj_img)
    else:
        obj_data = \
            obj_img[self.npoints // 2, self.npoints // 2]
    self.ref_data[i] = ref_img
    self.obj_data[i] = obj_data


@wrap_non_picklable_objects
def generate_data_exp(self, i, path, crop):
    _settings = namedtuple('settings', ['REF_CROP'])
    settings = _settings(REF_CROP=crop)
    init_field = get_ref_imgnum(path, settings)
    npoints = init_field.shape[0]
    ref_img, obj_img = \
        self.iprofiles_gen(self.area_size, npoints, self.wl,
                           init_field, None, (),
                           self.object_gen, self.object_gen_args,
                           self.z_obj, self.z_ref,
                           self.use_gpu, *self.iprofiles_gen_args)
    if self.use_backet:
        obj_data = self.xp.sum(obj_img)
    else:
        obj_data = \
            obj_img[self.npoints // 2, self.npoints // 2]
    self.ref_data[i] = ref_img
    self.obj_data[i] = obj_data


@dataclass
class GIEmulator(GIExpDataProcessor):

    area_size: float
    npoints: int
    wl: float

    imgs_number: int

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

    expdata_dir: str = ''
    expdata_format: str = 'bmp'
    expdata_crop: list = (150, 360,	70,	280)

    parallel_njobs: int = 4
    suppress_log: bool = False
    log_file: str = ''

    xp: object = np
    tcpoints: int = 10

    def __post_init__(self):
        """
        Клаcс предназначен для эмуляции эксперимента
        по наблюдению фантомных изображений в квазитепловом свете

        """
        if self.use_cupy:
            self.xp = cp

        self.obj_data = self.xp.empty(self.imgs_number, dtype=np.float32)
        self.ref_data = self.xp.empty(
            (self.imgs_number, self.npoints, self.npoints), dtype=np.float32)
        self.settings = namedtuple('settings', ['TCPOINTS'])
        self.settings.TCPOINTS = self.tcpoints
        """
        Здесь создается список объектных и референсных изображений
        self.obj_data -- изображения объекта
        self.ref_data -- изображения референсного пучка
        """
        log.info('Generating obj and ref data')
        log.info(f'Ref propagation distance is {self.z_ref} cm')
        log.info(f'Obj propagation distance is {self.z_obj} cm')
        t = time.time()

        if self.use_expdata:
            log.info(f'Using experimental profiles from {self.expdata_dir}')
            Parallel(n_jobs=self.parallel_njobs)(
                delayed(generate_data_exp)(self, i, path, self.expdata_crop)
                for i, path in tqdm(
                    enumerate(find_images(
                        self.expdata_dir,
                        self.imgs_number,
                        self.expdata_format)), position=0, leave=True))
        else:
            log.info(
                f'Using profiles generated by `{self.init_field_gen.__name__}`')
            try:
                Parallel(n_jobs=self.parallel_njobs)(
                    delayed(generate_data)(self, i) for i in tqdm(
                        range(self.imgs_number), position=0, leave=True))
            except Exception as E:
                log.warn(f'Processes-based parallelization failed: {E}')
                log.info('Trying to use threading parallelization')
                Parallel(n_jobs=self.parallel_njobs,
                         backend="threading")(
                    delayed(generate_data)(self, i) for i in tqdm(
                        range(self.imgs_number), position=0, leave=True))
        print()
        log.info(
            f'Obj and ref data generated. Elapsed time {(time.time() - t):.3f} s')
        self.gi = self.xp.zeros_like(self.ref_data[0])
        self.Nx = self.Ny = self.ghost_data.shape[0]
        self.sc = self.xp.zeros((self.Ny, self.Nx), dtype=np.float32)
        self.tc = self.xp.ones(self.settings.TCPOINTS)
        self.cd = self.xp.zeros((self.Ny, self.Nx), dtype=np.float32)
        self.times = np.arange(self.npoints)
