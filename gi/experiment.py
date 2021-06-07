# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:40:40 2021

@author: von.gostev
"""
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


def crop(x, c): return x[c[2]:c[3], c[0]:c[1]]


def crop_shape(c): return (c[3] - c[2], c[1] - c[0])


def load_settings(path):
    with open(path, 'r') as f:
        return json.load(f)


def ImgFinder(settings):
    dir_name = settings['DIR']
    yield from [join(dir_name, f) for f in listdir(dir_name)[:settings['N']]
                if isfile(join(dir_name, f)) and f.endswith(settings['EXT'])]


n_coh = 1
m_coh = 1


def get_corrcoef(obj_data, ref_data):
    return np.nan_to_num(np.corrcoef(obj_data**n_coh, ref_data**m_coh))[0, 1]


def GetGhostData(ghost_data, obj_data, ref_data):
    W, H = ghost_data.shape

    for i in nb.prange(W):
        for j in nb.prange(H):
            ghost_data[i, j] = get_corrcoef(obj_data, ref_data[:, i, j])


class ImgAnalyser:

    def __init__(self, settings_file):
        """
        ARGUMENTS
        ---------

        settings_file -- путь к файлу с настройками эксперимента в формате json
        """

        self.settings = load_settings(settings_file)
        print(self.settings)

        self.N = self.settings['N']
        self.Ny, self.Nx = crop_shape(self.settings['REF_CROP'])
        print(self.Ny, self.Nx)

        self.obj_data = np.zeros(self.N)
        self.ref_data = np.zeros((self.N, self.Ny, self.Nx), dtype=np.int16)
        self.ghost_data = np.zeros((self.Ny, self.Nx), dtype=np.float32)

        self.sc = np.zeros((self.Ny, self.Nx), dtype=np.float32)
        self.tc = [
            np.linspace(
                0, 1 / self.settings['FREQ'] * self.settings['TCPOINTS'],
                self.settings['TCPOINTS'])]
        self.cd = np.zeros((self.Ny, self.Nx), dtype=np.float32)

        self._create_data()

        if len(self.ref_data) == 0:
            raise IOError(
                "Не найдено изображений в выбранной папке: %s" % self.settings['DIR'])

    def _create_data(self):
        """
        Здесь создается список объектных и референсных изображений
        self.obj_data -- изображения объекта
        self.ref_data -- изображения референсного пучка
        """
        i = 0

        for path in ImgFinder(self.settings):
            img = cv2.imread(path, 0)

            ref_img = crop(img, self.settings['REF_CROP'])
            obj_img = crop(img, self.settings['OBJ_CROP'])
            # plt.imshow(obj_img)
            # plt.show()

            self._fill_ro_data(ref_img, obj_img, i)
            i += 1

    def _fill_ro_data(self, ref_img, obj_img, i):
        if not self.settings['DIFF']:
            self.ref_data[i, :, :] = ref_img
            self.obj_data[i] = self._get_obj_data(obj_img)
        else:
            self.ref_data = self._get_diff_data(ref_img, obj_img)

    def _get_obj_data(self, img):
        if self.settings['BACKET']:
            return np.sum(img)
        return img[self.settings['Y_CORR'], self.settings['X_CORR']]

    def _get_diff_data(self, ref_img, obj_img):
        if not crop_shape(self.settings['REF_CROP']) == crop_shape(self.settings['OBJ_CROP']):
            y1, x1 = crop_shape(self.settings['REF_CROP'])
            y2, x2 = crop_shape(self.settings['OBJ_CROP'])
            x = np.min(x1, x2)
            y = np.min(y1, y2)
            return ref_img[:y, :x] - obj_img[:y, :x]

        else:
            return ref_img - obj_img

    def correlate(self):
        """
        Расчет корреляции между последовательностью суммарных сигналов в объектном плече
        и поточечными последовательностями сигналов в референсном плече
        """
        GetGhostData(
            self.ghost_data, self.obj_data, self.ref_data)

    def diff(self):
        """
        Расчет разности между последовательностями объектных и референсных изображений
        """
        np.mean(self.ref_data, axis=0, out=self.ghost_data)

    def spatial_coherence(self, x=0, y=0):
        """
        Расчет функции когерентности или поперечной корреляции
        """
        if x == 0:
            x = self.Nx // 2
        if y == 0:
            y = self.Ny // 2
        point_data = self.ref_data[:, y, x]
        GetGhostData(self.sc, point_data, self.ref_data)

    def time_coherence(self):
        self.tc.append(np.array([1] + [np.mean([np.corrcoef(
            self.ref_data[:-i, k, m], self.ref_data[i:, k, m])[0, 1]
            for k in np.arange(self.Ny // 2 - 20, self.Ny // 2 + 20)
            for m in np.arange(self.Nx // 2 - 20, self.Nx // 2 + 20)])
            for i in range(1, self.settings['TCPOINTS'])]))

    def contrast(self):

        self.cd = (self.ghost_data - np.mean(self.ghost_data)) / \
            self.ghost_data
        self.cd[np.abs(self.cd) > 1] = 0
        return np.mean(self.cd)

    def sc_width(self):
        d = self.sc[self.Ny // 2, :]
        spline = UnivariateSpline(np.arange(self.Nx), d - np.max(d) / 2, s=0)
        r = spline.roots()
        print(r)
        if not len(r):
            return 0
        return np.abs(r[1] - r[0])

    def tc_width(self):
        X, Y = self.tc
        spline = UnivariateSpline(
            X, Y - np.max(Y) / 2, s=0)
        r = spline.roots()
        print(r)
        return np.abs(r[0])

    def information(self):
        self.global_settings = load_settings(self.settings['GSFILE'])

        self.text_global = [self.settings["INFO"] +
                            '\n'] if "INFO" in self.settings else []
        self.text_global += [
            'Условия проведения эксперимента\n',
            'Источник теплового света основан на He-Ne лазере, \
излучение которого проходит через матовый диск, \
вращающийся с угловой скоростью {DISKV} град/с.'.format(**self.settings),
            'Диаметр лазерного пучка на матовом диске составляет {BEAMD:.2f} см.'.format(
                **self.global_settings),
            'Диск находится на расстоянии {AL1} см от линзы L1 \
с фокусным расстоянием {F1} см, и в сопряженной \
оптической плоскости на расстоянии {BL1} см от линзы \
строится изображение поверхности диска.'.format(**self.global_settings),
            'Это изображение передается в объектную плоскость системы линзой L2 \
с фокусным расстоянием {F2} см, стоящей в {AL2} см от диска и в {BL2} см от объекта.'.format(**self.global_settings),
            'В объектном плече стоит линза L3 с фокусным расстоянием {F3} см в {AL3} см \
от объекта и в {BL3} см от CCD-камеры'.format(**self.global_settings),
            'В опорном плече плоскость, идентичная объектной передается на CCD-камеру \
линзой L4 с фокусным расстоянием {F4} см, расположенной в {AL4} см от передаваемой плоскости \
и в {BL4} см от CCD-камеры'.format(**self.global_settings)]
        self.text_ccd = [
            'CCD-камера работает в режиме программного запуска с выдержкой {EXCERT} мс и \
усилением {AMPL}'.format(**self.settings),
            'Частота съемки составляет {FREQ} Гц.'.format(**self.settings),
            'Область регистрации объектного пучка составляет {0:d} на {0:d} точек'.format(
                *crop_shape(self.settings['OBJ_CROP'])),
            'Область регистрации опорного пучка составляет {0:d} на {0:d} точек'.format(
                self.Nx, self.Ny),
            'Всего обрабатывается {0:d} изображений.'.format(self.N)]
        self.text_cf = [
            'Ширина пространственной автокорреляционной функции в опорном пучке составляет {0:f} точек'.format(
                self.sc_width()),
            'Ширина временной корреляционной функции в опорном пучке составляет {0:f} c.'.format(
                self.tc_width())]
        self.text_gi = [
            'Средняя контрастность фантомного изображения составляет {0:.2%}'.format(
                self.contrast()),
            'Видность фантомного изображения составляет {0:.2%}'.format(np.mean(self.ghost_data) / np.max(self.ghost_data))]

        return self.text_global + self.text_ccd + self.text_cf + self.text_gi

    def save_information(self):
        with open(join(self.settings['DIR'], 'info.txt'), 'w') as f:
            f.write('\n'.join(self.information()))

    def save_pictures(self):
        # np.mean(self.ref_data)*np.mean(self.obj_data)
        #noise = np.mean(self.ghost_data)
        print("HEEREE")
        mean_Ghost = np.mean(self.ghost_data)
        print("Ghost_data", mean_Ghost)
        noise = np.mean(self.ghost_data)
       # self.ghost_data[np.where(self.ghost_data <= noise)] = 0

        EXT = 'png'

        plt.imsave(
            join(self.settings['DIR'], 'gi%d.%s' % (self.N, EXT)), self.ghost_data, format=EXT, dpi=150)
        plt.imsave(
            join(self.settings['DIR'], 'sc%d.%s' % (self.N, EXT)), self.sc, format=EXT, dpi=150)
        plt.imsave(join(self.settings['DIR'], 'rd.%s' % EXT), self.ref_data[
                   0], format=EXT, dpi=150)
        plt.imsave(join(self.settings['DIR'], 'cd%d.%s' % (
            self.N, EXT)), self.cd, format=EXT, dpi=150)

        plt.figure()
        plt.plot(*self.tc)
        plt.savefig(
            join(self.settings['DIR'], 'tc.%s' % EXT), format=EXT, dpi=150)

        plt.figure()
        # plt.xticks(fontsize=21)
        # plt.yticks(fontsize=21)
        #ax = plt.subplot(111)
        #ax.set_xlabel('Номер пикселя', fontsize=21)
        #ax.set_ylabel(r'Нормированная корреляционная функция', fontsize=21)

        plt.plot(np.arange(1, self.Ny + 1), self.sc[:, self.Nx // 2])
        plt.savefig(
            join(self.settings['DIR'], 'sc1d%d.%s' % (self.N, EXT)), format=EXT, dpi=150)


class ImgViewer:

    def __init__(self, init):
        """
        Инициализация изображения по пути к изображению 
            или принятому массиву пикселей numpy.ndarray
        """
        if type(init) == str:
            self.path = init
            self.data = [cv2.imread(self.path, 0)]
        elif type(init) == np.ndarray:
            self.path = 'generated_img'
            self.data = [np.abs(init)]

    def accumulate(self, data):
        """
        Добавление изображений в виде массивов numpy.ndarray или list в список отображения
        """
        if type(data) == list:
            data = np.array(data)
        elif type(data) != np.ndarray:
            raise ValueError(
                'Additional data must be "list" or "ndarray", not "%s"' % type(data))
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
