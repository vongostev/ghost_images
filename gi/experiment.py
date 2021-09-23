# -*- coding: utf-8 -*-
'''
Created on Mon Jun  7 17:40:40 2021

@author: von.gostev
'''
from os import listdir
from os.path import isfile, join, dirname, realpath

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import UnivariateSpline

import json
from skimage.transform import downscale_local_mean
from skimage import io

from joblib import Parallel, delayed


def low_res(img, n):
    return downscale_local_mean(img, (n, n))


def crop(img, c):
    return img[c[2]:c[3], c[0]:c[1]]


def crop_shape(c):
    return (c[3] - c[2], c[1] - c[0])


class GISettings:

    def __init__(self, path):
        with open(path, 'r') as f:
            settings = json.load(f)
        for attr in settings:
            setattr(self, attr, settings[attr])

        if hasattr(self, 'DIR'):
            self.settings_path = dirname(path)
            self.DIR = realpath(join(self.settings_path, self.DIR))
            self.GSFILE = join(self.settings_path, self.GSFILE)


def ImgFinder(settings):
    dir_name = settings.DIR
    return [join(dir_name, f) for f in listdir(dir_name)[:settings.N]
            if isfile(join(dir_name, f)) and f.endswith(settings.EXT)]


def get_diff_img(ref_img, obj_img, settings):
    if ref_img.shape != obj_img.shape:
        ny_ref, nx_ref = crop_shape(settings.REF_CROP)
        ny_obj, nx_obj = crop_shape(settings.OBJ_CROP)

        nx = np.min(nx_ref, nx_obj)
        ny = np.min(ny_ref, ny_obj)
        return ref_img[:ny, :nx] - obj_img[:ny, :nx]
    else:
        return ref_img - obj_img


def get_obj_and_ref_imgs(path, settings):
    img = io.imread(path, 0)

    ref_img = crop(img, settings.REF_CROP)
    obj_img = crop(img, settings.OBJ_CROP)

    if settings.BACKET:
        obj_data = np.sum(obj_img)
    else:
        obj_data = obj_img[settings.Y_CORR, settings.X_CORR]

    if settings.DIFF:
        ref_data = get_diff_img(ref_img, obj_img, settings)
    else:
        ref_data = ref_img

    return ref_data.astype(np.uint8), obj_data.astype(np.uint8)


def data_correlation(obj_data, ref_data):
    def gi(pixel_data):
        return np.nan_to_num(np.corrcoef(obj_data, pixel_data))[0, 1]
    return np.apply_along_axis(gi, 0, ref_data)


def FWHM(n, data1d):
    npeak = np.argmax(data1d)
    spline = UnivariateSpline(np.arange(n), data1d - np.max(data1d) / 2, s=0)
    r = spline.roots()
    if len(r) > 1:
        return abs(r[1] - r[0])
    elif len(r) == 1:
        return 2 * abs(r[0] - npeak)
    return 0


def xycorr_width(sc):
    ny, nx = sc.shape
    ypeak, xpeak = np.unravel_index(np.argmax(sc), sc.shape)
    xslice = sc[ypeak, :]
    yslice = sc[:, xpeak]
    return FWHM(nx, xslice), FWHM(ny, yslice)


def xycorr(self, p, w):
    x, y = p
    lx = max(x - w // 2, 0)
    ly = max(y - w // 2, 0)
    tx = min(x + w // 2, self.Nx)
    ty = min(y + w // 2, self.Ny)
    point_data = self.ref_data[:, y, x]
    sc = data_correlation(point_data, self.ref_data[:, ly:ty, lx:tx])
    return xycorr_width(sc)


class ImgAnalyser:

    def __init__(self, settings_file, n_images=0):
        '''
        ARGUMENTS
        ---------

        settings_file -- путь к файлу с настройками эксперимента в формате json
        '''
        self.settings = GISettings(settings_file)
        if n_images:
            self.settings.N = n_images

        print('Experiment settings:', json.dumps(
            self.settings.__dict__, indent=4))

        self.N = self.settings.N
        self.Ny, self.Nx = crop_shape(self.settings.REF_CROP)
        print(f'Reference images size is {self.Nx}x{self.Ny}')

        self.obj_data = np.zeros(self.N)
        self.ref_data = np.zeros((self.N, self.Ny, self.Nx), dtype=np.int16)
        self.gi = np.zeros((self.Ny, self.Nx), dtype=np.float32)

        self.sc = np.zeros((self.Ny, self.Nx), dtype=np.float32)
        self.times = np.linspace(
            0, self.settings.TCPOINTS / self.settings.FREQ,
            self.settings.TCPOINTS)
        self.tc = np.ones(self.settings.TCPOINTS)
        self.cd = np.zeros((self.Ny, self.Nx), dtype=np.float32)
        self.g2 = 0

        self._create_data()

    def _create_data(self):
        '''
        Здесь создается список объектных и референсных изображений
        self.obj_data -- изображения объекта
        self.ref_data -- изображения референсного пучка
        '''
        imgs_path = ImgFinder(self.settings)
        if len(imgs_path) == 0:
            raise IOError(
                'Не найдено изображений в выбранной папке: %s' % self.settings.DIR)

        for i, path in enumerate(imgs_path):
            self.ref_data[i, :, :], self.obj_data[i] = \
                get_obj_and_ref_imgs(path, self.settings)

    def calculate_ghostimage(self):
        '''
        Расчет корреляции между последовательностью суммарных сигналов в объектном плече
        и поточечными последовательностями сигналов в референсном плече
        '''
        self.gi = data_correlation(self.obj_data, self.ref_data)

    @property
    def diff(self):
        '''
        Расчет разности между последовательностями объектных и референсных изображений
        '''
        return np.mean(self.ref_data, axis=0)

    def calculate_xycorr(self, x=0, y=0):
        '''
        Расчет функции когерентности или поперечной корреляции
        '''
        if x == 0:
            x = self.Nx // 2
        if y == 0:
            y = self.Ny // 2
        point_data = self.ref_data[:, y, x]
        self.sc = data_correlation(point_data, self.ref_data)

    def calculate_xycorr_widths(self, window_points: int = 50, nx: int = 10, ny: int = 10,
                                n_jobs: int = -2):
        """
        Расчет ширин функции когерентности или поперечной корреляции для разных пикселей


        Parameters
        ----------
        window_points : int, optional
            Width of the calculation window (in points) for the each point of calculation. The default is 50.
        nx : int, optional
            Number of points in x dimention. The default is 10.
            The centrum is in the centrum of self.ref_data images:
                self.Nx // 2 - nx // 2 <= x < self.Nx // 2 + nx // 2
        ny : int, optional
            Number of points in y dimention. The default is 10.
            The centrum is in the centrum of self.ref_data images:
                self.Ny // 2 - ny // 2 <= y < self.Ny // 2 + ny // 2
        n_jobs: int, optional
            Number of jobs in parallel calculations.
            The default is -2.

        Return
        ---------

        Two arrays with xy correlation function widths: by x  and by y.

        """

        X = np.arange(- nx // 2, nx // 2) + self.Nx // 2
        Y = np.arange(-ny // 2, ny // 2) + self.Ny // 2
        points = np.array(np.meshgrid(X, Y)).T.reshape(-1, 2)
        w = window_points

        _rawd = Parallel(n_jobs=n_jobs)(
            delayed(xycorr)(self, p, w) for p in points)
        _rawdx = np.array([w[0] for w in _rawd]).reshape((ny, nx))
        _rawdy = np.array([w[1] for w in _rawd]).reshape((ny, nx))
        self.sc_widths = (_rawdx, _rawdy)

    def calculate_timecorr(self, npoints=100):
        def cf1d(data, i):
            return np.nan_to_num(np.corrcoef(data[:-i], data[i:])[0, 1])

        rdim = self.Nx * self.Ny // 2
        ravel_data = self.ref_data.reshape((self.N, rdim * 2))
        ravel_data = ravel_data[:, rdim - npoints // 2: rdim + npoints // 2]
        self.tc[1:] = np.mean([np.apply_along_axis(cf1d, 1, ravel_data, i)
                               for i in range(1, self.settings.TCPOINTS)], axis=-1)

    def calculate_contrast(self):
        self.cd = (self.gi - np.mean(self.gi)) / self.gi
        self.cd[np.abs(self.cd) > 1] = 0

    def calculate_all(self):
        self.calculate_ghostimage()
        self.calculate_contrast()
        self.calculate_xycorr()
        self.calculate_timecorr()

    def g2_intensity(self, noise):
        self.g2 = np.mean((self.ref_data - noise)**2, axis=0) / \
            np.mean(self.ref_data - noise, axis=0)**2

    @property
    def ghost_data(self):
        return self.gi

    @property
    def timecorr_data(self):
        return self.tc

    @property
    def xycorr_data(self):
        return self.sc

    @property
    def contrast_data(self):
        return self.cd

    @property
    def contrast(self):
        return np.mean(self.cd)

    @property
    def xycorr_width(self):
        return xycorr_width(self.sc)

    @property
    def timecorr_width(self):
        return FWHM(len(self.times), self.tc) * self.times[1]

    @property
    def information(self):
        self.global_settings = GISettings(self.settings.GSFILE)

        sets = self.settings
        gsets = self.global_settings

        self.text_global = [getattr(sets, 'INFO', '') + '\n']
        self.text_global += [
            'Условия проведения эксперимента\n',
            'Источник теплового света основан на He-Ne лазере, ' +
            'излучение которого проходит через матовый диск, ' +
            f'вращающийся с угловой скоростью {gsets.DISKV} град/с.',
            f'Диаметр лазерного пучка на матовом диске составляет {gsets.BEAMD:.2f} см.',
            f'Диск находится на расстоянии {gsets.AL1} см от линзы L1 ' +
            f'с фокусным расстоянием {gsets.F1} см, и в сопряженной ' +
            f'оптической плоскости на расстоянии {gsets.BL1} см от линзы ' +
            'строится изображение поверхности диска.',
            'Это изображение передается в объектную плоскость системы линзой L2 ' +
            f'с фокусным расстоянием {gsets.F2} см, ' +
            f'стоящей в {gsets.AL2} см от диска и в {gsets.BL2} см от объекта.',
            f'В объектном плече стоит линза L3 с фокусным расстоянием {gsets.F3} см в {gsets.AL3} см ' +
            f'от объекта и в {gsets.BL3} см от CCD-камеры',
            'В опорном плече плоскость, идентичная объектной передается на CCD-камеру ' +
            f'линзой L4 с фокусным расстоянием {gsets.F4} см, ' +
            f'расположенной в {gsets.AL4} см от передаваемой плоскости и в {gsets.BL4} см от CCD-камеры']
        self.text_ccd = [
            f'CCD-камера работает в режиме программного запуска с выдержкой {sets.EXCERT} мс ' +
            f'и усилением {sets.AMPL}',
            f'Частота съемки составляет {sets.FREQ} Гц.',
            'Область регистрации объектного пучка составляет {0:d} на {1:d} точек'.format(
                *crop_shape(sets.OBJ_CROP)),
            f'Область регистрации опорного пучка составляет {self.Nx:d} на {self.Ny:d} точек',
            f'Всего обрабатывается {self.N:d} изображений.']
        self.text_cf = [
            f'Ширина пространственной автокорреляционной функции в опорном пучке составляет {self.xycorr_width:f} точек',
            f'Ширина временной корреляционной функции в опорном пучке составляет {self.timecorr_width:f} c.']
        self.text_gi = [
            f'Средняя контрастность фантомного изображения составляет {self.contrast:.2%}',
            f'Видность фантомного изображения составляет {np.mean(self.gi) / np.max(self.gi):.2%}']

        return self.text_global + self.text_ccd + self.text_cf + self.text_gi

    def save_information(self):
        with open(join(self.settings.DIR, 'info.txt'), 'w') as f:
            f.write('\n'.join(self.information))

    def save_pictures(self, img_ext='png', thresh_mean=False):
        # np.mean(self.ref_data)*np.mean(self.obj_data)
        if thresh_mean:
            noise = np.mean(self.gi)
            self.gi[np.where(self.gi <= noise)] = 0

        img_dir = self.settings.DIR
        plt.imsave(
            join(img_dir, f'gi{self.N}.{img_ext}'),
            self.gi, format=img_ext, dpi=150)
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
            self.data = [io.imread(self.path, 0)]
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
