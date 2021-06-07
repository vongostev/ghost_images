# -*- coding: utf-8 -*-
'''
Created on Mon Jun  7 17:40:40 2021

@author: von.gostev
'''
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import numba as nb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import UnivariateSpline

import json
from skimage.transform import downscale_local_mean


def low_res(img, n):
    return downscale_local_mean(img, (n, n))


def crop(x, c):
    return x[c[2]:c[3], c[0]:c[1]]


def crop_shape(c):
    return (c[3] - c[2], c[1] - c[0])


class GISettings:

    def __init__(self, path):
        with open(path, 'r') as f:
            settings = json.load(f)
        for attr in settings:
            setattr(self, attr, settings[attr])


def ImgFinder(settings):
    dir_name = settings.DIR
    yield from [join(dir_name, f) for f in listdir(dir_name)[:settings.N]
                if isfile(join(dir_name, f)) and f.endswith(settings.EXT)]


n_coh = 1
m_coh = 1


def data_correlation(obj_data, ref_data):
    def gi(pixel_data):
        return np.nan_to_num(np.corrcoef(obj_data, pixel_data))[0, 1]
    return np.apply_along_axis(gi, 0, ref_data)


class ImgAnalyser:

    def __init__(self, settings_file):
        '''
        ARGUMENTS
        ---------

        settings_file -- путь к файлу с настройками эксперимента в формате json
        '''

        self.settings = GISettings(settings_file)
        print(self.settings)

        self.N = self.settings.N
        self.Ny, self.Nx = crop_shape(self.settings.REF_CROP)
        print(self.Ny, self.Nx)

        self.obj_data = np.zeros(self.N)
        self.ref_data = np.zeros((self.N, self.Ny, self.Nx), dtype=np.int16)
        self.ghost_data = np.zeros((self.Ny, self.Nx), dtype=np.float32)

        self.sc = np.zeros((self.Ny, self.Nx), dtype=np.float32)
        self.times = np.linspace(
            0, self.settings.TCPOINTS / self.settings.FREQ,
            self.settings.TCPOINTS)
        self.time_corr = np.zeros(self.settings.TCPOINTS)
        self.cd = np.zeros((self.Ny, self.Nx), dtype=np.float32)

        self._create_data()

        if len(self.ref_data) == 0:
            raise IOError(
                'Не найдено изображений в выбранной папке: %s' % self.settings.DIR)

    def _create_data(self):
        '''
        Здесь создается список объектных и референсных изображений
        self.obj_data -- изображения объекта
        self.ref_data -- изображения референсного пучка
        '''
        i = 0

        for path in ImgFinder(self.settings):
            img = cv2.imread(path, 0)

            ref_img = crop(img, self.settings.REF_CROP)
            obj_img = crop(img, self.settings.OBJ_CROP)

            if not self.settings.DIFF:
                self.ref_data[i, :, :] = ref_img
                if self.settings.BACKET:
                    self.obj_data[i] = np.sum(obj_img)
                else:
                    self.obj_data[i] = \
                        obj_img[self.settings.Y_CORR, self.settings.X_CORR]
            else:
                self.ref_data = self._get_diff_data(ref_img, obj_img)

            i += 1

    def _get_diff_data(self, ref_img, obj_img):
        if not (self.Ny, self.Nx) == crop_shape(self.settings.OBJ_CROP):
            y2, x2 = crop_shape(self.settings.OBJ_CROP)
            x = np.min(self.Nx, x2)
            y = np.min(self.Ny, y2)
            return ref_img[:y, :x] - obj_img[:y, :x]

        else:
            return ref_img - obj_img

    def correlate(self):
        '''
        Расчет корреляции между последовательностью суммарных сигналов в объектном плече
        и поточечными последовательностями сигналов в референсном плече
        '''
        self.ghost_data = data_correlation(self.obj_data, self.ref_data)

    @property
    def diff(self):
        '''
        Расчет разности между последовательностями объектных и референсных изображений
        '''
        return np.mean(self.ref_data, axis=0)

    def spatial_coherence(self, x=0, y=0):
        '''
        Расчет функции когерентности или поперечной корреляции
        '''
        if x == 0:
            x = self.Nx // 2
        if y == 0:
            y = self.Ny // 2
        point_data = self.ref_data[:, y, x]
        self.sc = data_correlation(point_data, self.ref_data)

    def time_coherence(self):
        def cf1d(data, i):
            return np.nan_to_num(np.corrcoef(data[:-i], data[i:])[0, 1])

        ravel_data = self.ref_data.reshape((self.N, self.Nx * self.Ny))
        self.time_corr = \
            np.array([1] +
                     [np.mean(np.apply_along_axis(cf1d, 0, ravel_data, i))
                         for i in range(1, self.settings.TCPOINTS)])

    @property
    def contrast(self):
        with self.ghost_data as gi:
            self.cd = (gi - np.mean(gi)) / gi
        self.cd[np.abs(self.cd) > 1] = 0
        return np.mean(self.cd)

    @property
    def sc_width(self):
        d = self.sc[self.Ny // 2, :]
        spline = UnivariateSpline(np.arange(self.Nx), d - np.max(d) / 2, s=0)
        r = spline.roots()
        print(r)
        if not len(r):
            return 0
        return np.abs(r[1] - r[0])

    @property
    def tc_width(self):
        X, Y = self.tc
        spline = UnivariateSpline(
            X, Y - np.max(Y) / 2, s=0)
        r = spline.roots()
        print(r)
        return np.abs(r[0])

    @property
    def information(self):
        self.global_settings = GISettings(self.settings.GSFILE)

        with self.settings as sets:
            self.text_global = [getattr(sets, 'INFO', '') + '\n']
            with self.global_settings as gsets:
                self.text_global += [
                    'Условия проведения эксперимента\n',
                    'Источник теплового света основан на He-Ne лазере, ' +
                    'излучение которого проходит через матовый диск, ' +
                    f'вращающийся с угловой скоростью {sets.DISKV} град/с.',
                    f'Диаметр лазерного пучка на матовом диске составляет {gsets.BEAMD:.2f} см.',
                    f'Диск находится на расстоянии {gsets.AL1} см от линзы L1 ' +
                    f'с фокусным расстоянием {gsets.F1} см, и в сопряженной ' +
                    f'оптической плоскости на расстоянии {gsets.BL1} см от линзы ' +
                    'строится изображение поверхности диска.',
                    'Это изображение передается в объектную плоскость системы линзой L2 ' +
                    f'с фокусным расстоянием {gsets.F2} см, ' +
                    'стоящей в {gsets.AL2} см от диска и в {gsets.BL2} см от объекта.',
                    'В объектном плече стоит линза L3 с фокусным расстоянием {gsets.F3} см в {gsets.AL3} см ' +
                    f'от объекта и в {gsets.BL3} см от CCD-камеры',
                    'В опорном плече плоскость, идентичная объектной передается на CCD-камеру ' +
                    f'линзой L4 с фокусным расстоянием {gsets.F4} см, ' +
                    'расположенной в {gsets.AL4} см от передаваемой плоскости и в {gsets.BL4} см от CCD-камеры']
            self.text_ccd = [
                'CCD-камера работает в режиме программного запуска с выдержкой {sets.EXCERT} мс ' +
                f'и усилением {sets.AMPL}',
                f'Частота съемки составляет {sets.FREQ} Гц.',
                'Область регистрации объектного пучка составляет {0:d} на {1:d} точек'.format(
                    *crop_shape(sets.OBJ_CROP)),
                f'Область регистрации опорного пучка составляет {self.Nx:d} на {self.Ny:d} точек',
                f'Всего обрабатывается {self.N:d} изображений.']
        self.text_cf = [
            f'Ширина пространственной автокорреляционной функции в опорном пучке составляет {self.sc_width:f} точек',
            f'Ширина временной корреляционной функции в опорном пучке составляет {self.tc_width:f} c.']
        self.text_gi = [
            f'Средняя контрастность фантомного изображения составляет {self.contrast:.2%}',
            f'Видность фантомного изображения составляет {np.mean(self.ghost_data) / np.max(self.ghost_data):.2%}']

        return self.text_global + self.text_ccd + self.text_cf + self.text_gi

    def save_information(self):
        with open(join(self.settings.DIR, 'info.txt'), 'w') as f:
            f.write('\n'.join(self.information))

    def save_pictures(self, img_ext='png', thresh_mean=False):
        # np.mean(self.ref_data)*np.mean(self.obj_data)
        if thresh_mean:
            noise = np.mean(self.ghost_data)
            self.ghost_data[np.where(self.ghost_data <= noise)] = 0

        img_dir = self.settings.DIR
        plt.imsave(
            join(img_dir, f'gi{self.N}.{img_ext}'),
            self.ghost_data, format=img_ext, dpi=150)
        plt.imsave(
            join(img_dir, f'sc{self.N}.{img_ext}'),
            self.sc, format=img_ext, dpi=150)
        plt.imsave(join(img_dir, f'rd.{img_ext}'),
                   self.ref_data[0], format=img_ext, dpi=150)
        plt.imsave(join(img_dir, f'cd{self.N}.{img_ext}'),
                   self.cd, format=img_ext, dpi=150)

        plt.figure()
        plt.plot(*self.tc)
        plt.savefig(join(img_dir, f'tc.{img_ext}'),
                    format=img_ext, dpi=150)

        plt.figure()
        # plt.xticks(fontsize=21)
        # plt.yticks(fontsize=21)
        # ax = plt.subplot(111)
        # ax.set_xlabel('Номер пикселя', fontsize=21)
        # ax.set_ylabel(r'Нормированная корреляционная функция', fontsize=21)

        plt.plot(np.arange(1, self.Ny + 1), self.sc[:, self.Nx // 2])
        plt.savefig(join(img_dir, f'sc{self.N}.{img_ext}'),
                    format=img_ext, dpi=150)


class ImgViewer:

    def __init__(self, init):
        '''
        Инициализация изображения по пути к изображению 
            или принятому массиву пикселей numpy.ndarray
        '''
        if type(init) == str:
            self.path = init
            self.data = [cv2.imread(self.path, 0)]
        elif type(init) == np.ndarray:
            self.path = 'generated_img'
            self.data = [np.abs(init)]

    def accumulate(self, data):
        '''
        Добавление изображений в виде массивов numpy.ndarray или list в список отображения
        '''
        if type(data) == list:
            data = np.array(data)
        elif type(data) != np.ndarray:
            raise ValueError(
                f'Additional data must be "list" or "ndarray", not {type(data)}')
        if data.shape != self.data[0].shape:
            raise ValueError('Additional data must be in the same dimensions')
        self.data.append(data)

    def show3d(self):
        fig = plt.figure()

        ax = Axes3D(fig)

        H, W = self.data[0].shape
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        print(X.shape, Y.shape, self.data[0].shape)
        ax.plot_surface(X, Y, np.abs(self.data[0]), rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.view_init(50, 80)
        plt.show()

    def show(self, size):
        for d in self.data:
            d = low_res(d, size)
            plt.figure()
            plt.imshow(d)
            plt.colorbar()
        plt.show()
