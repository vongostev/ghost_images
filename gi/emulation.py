# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:12:13 2021

@author: vonGostev
"""
from dataclasses import dataclass
import numpy as np

from lightprop2d import Beam2D


def generate_beams(area_size, npoints, wl,
                   init_field, init_field_gen, init_gen_args,
                   object_gen, object_gen_args,
                   z_obj, z_ref):

    obj = Beam2D(area_size, npoints, wl,
                 init_field=init_field,
                 init_field_gen=init_field_gen,
                 init_gen_args=init_gen_args)
    if object_gen is not None:
        obj.coordinate_filter(
            lambda x, y: object_gen(x, y, *object_gen_args))
    obj.propagate(z_obj)

    ref = Beam2D(area_size, npoints, wl, init_field=obj.xyprofile)
    ref.propagate(z_ref)

    return ref.iprofile, obj.iprofile


def data_correlation(obj_data, ref_data):
    imgs_number, img_height, img_width = ref_data.shape

    def gi(pixel_data):
        return np.nan_to_num(np.corrcoef(obj_data, pixel_data))[0, 1]
    return np.apply_along_axis(gi, 0, ref_data)


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

    use_backet: bool = True

    def __post_init__(self):
        """
        Клаcс предназначен для эмуляции эксперимента
        по наблюдению фантомных изображенийв квазитепловом свете

        """

        self.obj_data = np.zeros(self.imgs_number)
        self.ref_data = np.empty(
            (self.imgs_number, self.npoints, self.npoints))
        self.ghost_data = np.zeros((self.npoints, self.npoints))

        """
        Здесь создается список объектных и референсных изображений
        self.obj_data -- изображения объекта
        self.ref_data -- изображения референсного пучка
        """
        for i in range(self.imgs_number):
            ref_img, obj_img = \
                generate_beams(self.area_size, self.npoints, self.wl,
                               self.init_field, self.init_field_gen, self.init_gen_args,
                               self.object_gen, self.object_gen_args,
                               self.z_obj, self.z_ref)
            if self.use_backet:
                self.obj_data[i] = np.sum(obj_img)
            else:
                self.obj_data[i] = \
                    obj_img[self.npoints // 2, self.npoints // 2]

            self.ref_data[i, :, :] = ref_img

        self.obj_data /= np.max(self.obj_data)

    def correlate(self):
        """
        Расчет корреляции между последовательностью суммарных сигналов в объектном плече
        и поточечными последовательностями сигналов в референсном плече
        """
        self.ghost_data = data_correlation(self.obj_data, self.ref_data)

    def spatial_coherence(self):
        """
        Расчет функции когерентности или поперечной корреляции
        """
        central_point_data = \
            self.ref_data[:, self.npoints // 2, self.npoints // 2]
        self.ghost_data = data_correlation(central_point_data, self.ref_data)
