# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:12:13 2021

@author: vonGostev
"""
import sys
import time

from dataclasses import dataclass
from joblib import Parallel, delayed, wrap_non_picklable_objects
import numpy as np
from collections import namedtuple

from lightprop2d import Beam2D
from .experiment import find_images, get_ref_imgnum

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
                   z_obj, z_ref, use_gpu, *args, **kwargs):

    obj = Beam2D(area_size, npoints, wl,
                 init_field=init_field,
                 init_field_gen=init_field_gen,
                 init_gen_args=init_gen_args, use_gpu=use_gpu)
    ref = Beam2D(area_size, npoints, wl, init_field=obj.field, use_gpu=use_gpu)

    if object_gen is not None:
        obj.coordinate_filter(
            f_gen=lambda x, y: object_gen(x, y, *object_gen_args))

    obj.propagate(z_obj)
    ref.propagate(z_ref)
    refprofile = (ref.iprofile / np.max(ref.iprofile) * 255).astype(np.uint8)
    objprofile = (obj.iprofile / np.max(obj.iprofile) * 255).astype(np.uint8)

    if not use_gpu:
        return refprofile, objprofile
    else:
        return refprofile.get(), objprofile.get()


@wrap_non_picklable_objects
def generate_data(self):
    ref_img, obj_img = \
        self.iprofiles_gen(self.area_size, self.npoints, self.wl,
                           self.init_field, self.init_field_gen, self.init_gen_args,
                           self.object_gen, self.object_gen_args,
                           self.z_obj, self.z_ref,
                           self.use_gpu, *self.iprofiles_gen_args)
    if self.use_backet:
        obj_data = np.sum(obj_img)
    else:
        obj_data = \
            obj_img[self.npoints // 2, self.npoints // 2]
    return ref_img, obj_data


@wrap_non_picklable_objects
def generate_data_exp(self, path, crop):
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
        obj_data = np.sum(obj_img)
    else:
        obj_data = \
            obj_img[self.npoints // 2, self.npoints // 2]
    return ref_img, obj_data


def data_correlation(obj_data, ref_data, parallel_njobs=-1):
    def gi(pixel_data):
        return np.nan_to_num(np.corrcoef(obj_data, pixel_data))[0, 1]

    npoints = ref_data.shape[-1]
    ref_data = ref_data.reshape(ref_data.shape[0], -1).T
    corr_data = Parallel(n_jobs=parallel_njobs)(delayed(gi)(s)
                                                for s in ref_data)
    return np.asarray(corr_data).reshape((npoints, npoints))
    # return np.apply_along_axis(gi, 0, ref_data)


@dataclass
class ImgEmulator:

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

    expdata_dir: str = ''
    expdata_format: str = 'bmp'
    expdata_crop: list = (150, 360,	70,	280)

    parallel_njobs: int = 4
    suppress_log: bool = False
    log_file: str = ''

    def __post_init__(self):
        """
        Клаcс предназначен для эмуляции эксперимента
        по наблюдению фантомных изображенийв квазитепловом свете

        """

        self.data = np.empty(self.imgs_number,
                             dtype=[('ref', 'O'), ('obj', '<i4')])

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
            self.data[:] = Parallel(n_jobs=self.parallel_njobs)(
                delayed(generate_data_exp)(self, path, self.expdata_crop)
                for path in find_images(
                    self.expdata_dir,
                    self.imgs_number,
                    self.expdata_format))
        else:
            log.info(
                f'Using profiles generated by `{self.init_field_gen.__name__}`')
            self.data[:] = Parallel(n_jobs=self.parallel_njobs,
                                    backend="threading")(
                delayed(generate_data)(self) for _ in range(self.imgs_number))

        self.obj_data = self.data['obj']
        self.ref_data = np.stack(self.data['ref'])
        log.info(
            f'Obj and ref data generated. Elapsed time {(time.time() - t):.3f} s')
        self.ghost_data = np.zeros_like(self.ref_data[0])
        self.xycorr_data = np.zeros_like(self.ref_data[0])
        self.npoints = self.ghost_data.shape[0]
        del self.data

        # self.obj_data /= np.max(self.obj_data)

    def calculate_ghostimage(self):
        """
        Расчет корреляции между последовательностью суммарных сигналов в объектном плече
        и поточечными последовательностями сигналов в референсном плече
        """
        log.info('Calculating ghost image')
        t = time.time()
        self.ghost_data = data_correlation(self.obj_data, self.ref_data,
                                           self.parallel_njobs)
        log.info(
            f'Ghost image calculated. Elapsed time {(time.time() - t):.3f} s')

    def calculate_xycorr(self):
        """
        Расчет функции когерентности или поперечной корреляции
        """
        log.info('Calculating spatial correlation function')
        t = time.time()
        central_point_data = \
            self.ref_data[:, self.npoints // 2, self.npoints // 2]
        self.xycorr_data = data_correlation(central_point_data, self.ref_data,
                                            self.parallel_njobs)
        log.info(
            f'Spatial correlation function calculated. Elapsed time {(time.time() - t):.3f} s')
