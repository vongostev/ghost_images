# -*- coding: utf-8 -*-
'''
Created on Mon Jun  7 17:40:40 2021

@author: von.gostev
'''
from os import listdir
from os.path import isfile, join, dirname, realpath
import sys
import time
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import UnivariateSpline

import json
from skimage.transform import downscale_local_mean
from skimage import io

from joblib import Parallel, delayed, wrap_non_picklable_objects
from logging import Logger, StreamHandler, Formatter

log = Logger('EXP')

handler = StreamHandler(sys.stdout)
handler.setLevel(10)
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

IMG_CROP_DATA = 0
IMG_IMG_DATA = 1
IMG_NUM_DATA = 2


def low_res(img, n):
    return downscale_local_mean(img, (n, n))


def crop(img, c):
    return img[c[2]:c[3], c[0]:c[1]]


def crop_shape(c):
    return np.array([c[3] - c[2], c[1] - c[0]])


def imread(path, binning_order, crop_shape):
    img = io.imread(path, 0)
    img = crop(img, crop_shape)
    return low_res(img, binning_order).astype(np.uint8)


class GISettings:
    """
    Class to parse settings files:
        settings of the experiment and global settings
    """

    def __init__(self, path):
        log.info(f'Reading settings file {path}')
        with open(path, 'r') as f:
            settings = json.load(f)
        for attr in settings:
            setattr(self, attr, settings[attr])

        if hasattr(self, 'GSFILE'):
            self.settings_path = dirname(path)
            self.GSFILE = join(self.settings_path, self.GSFILE)
            log.info(f'Reading global settings')

        if hasattr(self, 'DIR'):
            self.DIR = realpath(join(self.settings_path, self.DIR))
            self.FORMAT = IMG_CROP_DATA
            log.info('Reference and objective data cropped from the one image\n' +
                     f'Ref and obj: {self.DIR}\n')
        if hasattr(self, 'REF_DIR'):
            self.REF_DIR = realpath(join(self.settings_path, self.REF_DIR))
        if hasattr(self, 'OBJ_DIR'):
            self.OBJ_DIR = realpath(join(self.settings_path, self.OBJ_DIR))
            self.FORMAT = IMG_IMG_DATA
            log.info('Reference and objective data loaded from different directories\n' +
                     f'Ref: {self.REF_DIR}\n' +
                     f'Obj: {self.OBJ_DIR}')
        if hasattr(self, 'OBJ_FILE'):
            self.OBJ_FILE = realpath(join(self.settings_path, self.OBJ_FILE))
            self.FORMAT = IMG_NUM_DATA
            log.info('Reference data loaded from images and objective data loaded a file\n' +
                     f'Ref: {self.REF_DIR}\n' +
                     f'Obj: {self.OBJ_FILE}')
        log.info(f'Settings loaded from {path}')


def find_images(dir_name, img_num, img_format):
    """
    Find paths to `img_num` images of `img_format` format
    in `dir_name` directory

    Parameters
    ----------
    dir_name : str
        Path to a directory containing images.
    img_num : int
        Number of images.
    img_format : str
        Format of images, for example bmp, png, etc.

    Returns
    -------
    list
        List of paths to images in dir_name.

    """
    return [join(dir_name, f) for f in listdir(dir_name)[:img_num]
            if isfile(join(dir_name, f)) and f.endswith(img_format)]


def get_images(dir_name, settings):
    """
    Find paths to `settings.N` images of `settings.EXT` format
    in `dir_name` directory with error processing

    Parameters
    ----------
    dir_name : str
        Path to a directory containing images.
    settings : GISettings
        Parsed settings of the experiment.

    Raises
    ------
    IOError
        No images found.

    Returns
    -------
    img_paths : list
        List of paths to images in dir_name.

    """
    img_num = settings.N
    img_format = settings.EXT
    img_paths = find_images(dir_name, img_num, img_format)
    if len(img_paths) == 0:
        raise IOError(f'Не найдено изображений в выбранной папке: {dir_name}')
    return img_paths


def get_diff_img(ref_img, obj_img, settings):
    if ref_img.shape != obj_img.shape:
        ny_ref, nx_ref = ref_img.shape
        ny_obj, nx_obj = obj_img.shape
        nx = np.min(nx_ref, nx_obj)
        ny = np.min(ny_ref, ny_obj)
        return ref_img[:ny, :nx] - obj_img[:ny, :nx]
    else:
        return ref_img - obj_img


def get_objref_imgcrop(path, settings):
    """
    Construct single reference and objective data pair
    if reference and objective data must be cropped from the one image

    Parameters
    ----------
    path : str
        Path to a directory containing reference and objective channels images.
    settings : GISettings
        Parsed settings of the experiment.

    Returns
    -------
    ref_data: np.ndarray
        An array with reference channel data.
    obj_data : np.ndarray
        An array with objective channel data.

    """
    return get_objref_twoimgs(path, path, settings)


def get_objref_twoimgs(ref_path, obj_path, settings):
    """
    Construct single reference and objective data pair
    if reference and objective data are images in different directories

    Parameters
    ----------
    ref_path : str
        Path to a directory containing reference channel images.
    obj_path : str
        Path to a directory containing objective channel images.
    settings : GISettings
        Parsed settings of the experiment.

    Returns
    -------
    ref_data: np.ndarray
        An array with reference channel data.
    obj_data : np.ndarray
        An array with objective channel data.

    """
    ref_img = imread(ref_path, settings.BINNING, settings.REF_CROP)
    obj_img = imread(obj_path, settings.BINNING, settings.OBJ_CROP)

    if settings.BACKET:
        obj_data = np.sum(obj_img)
    else:
        obj_data = obj_img[settings.Y_CORR, settings.X_CORR]

    if settings.DIFF:
        ref_data = get_diff_img(ref_img, obj_img, settings)
    else:
        ref_data = ref_img

    return ref_data.astype(np.uint8), obj_data


def get_ref_imgnum(ref_path, settings):
    """
    Construct single reference data image
    if reference data are images and objective data are numbers in a file

    Parameters
    ----------
    ref_path : str
        Path to a directory containing reference channel images.
    settings : GISettings
        Parsed settings of the experiment.

    Returns
    -------
    ref_data: np.ndarray
        An array with reference channel data.

    """
    return imread(ref_path, settings.BINNING, settings.REF_CROP)



def data_correlation(obj_data, ref_data, parallel_njobs=-1):
    def gi(pixel_data):
        return np.nan_to_num(np.corrcoef(obj_data, pixel_data))[0, 1]

    img_shape = ref_data.shape[1:]
    ref_data = ref_data.reshape(ref_data.shape[0], -1).T
    corr_data = Parallel(n_jobs=parallel_njobs)(
        delayed(gi)(s) for s in ref_data)
    return np.asarray(corr_data).reshape(img_shape)
    # return np.apply_along_axis(gi, 0, ref_data)


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


@wrap_non_picklable_objects
def xycorr(self, p, w):
    x, y = p
    lx = max(x - w // 2, 0)
    ly = max(y - w // 2, 0)
    tx = min(x + w // 2, self.Nx)
    ty = min(y + w // 2, self.Ny)
    point_data = self.ref_data[:, y, x]
    sc = data_correlation(point_data, self.ref_data[:, ly:ty, lx:tx],
                          self.parallel_njobs)
    return xycorr_width(sc)


class ObjRefGenerator:

    def __init__(self, settings, ref_data, obj_data, binning_order=1):
        '''
        Здесь создается список объектных и референсных изображений
        self.obj_data -- изображения объекта
        self.ref_data -- изображения референсного пучка
        '''
        self.settings = settings
        self.ref_data = ref_data
        self.obj_data = obj_data
        self.bo = binning_order

        if self.settings.FORMAT == IMG_CROP_DATA:
            self._create_data_crop()
        elif self.settings.FORMAT == IMG_IMG_DATA:
            self._create_data_twoimgs()
        elif self.settings.FORMAT == IMG_NUM_DATA:
            self._create_data_imgnum()

    def _create_data_crop(self):
        '''
        Здесь создается список объектных и референсных изображений,
        если данные представлены в виде картинок с двумя каналами одновременно
        '''
        img_paths = get_images(self.settings.DIR, self.settings)
        for i, path in enumerate(img_paths):
            self.ref_data[i, :, :], self.obj_data[i] = \
                get_objref_imgcrop(path, self.settings)

    def _create_data_twoimgs(self):
        '''
        Здесь создается список объектных и референсных изображений,
        если данные представлены в виде отдельных картинок на каждый канал
        '''
        ref_img_paths = get_images(self.settings.REF_DIR, self.settings)
        obj_img_paths = get_images(self.settings.OBJ_DIR, self.settings)
        img_paths = zip(ref_img_paths, obj_img_paths)

        for i, paths in enumerate(img_paths):
            self.ref_data[i, :, :], self.obj_data[i] = \
                get_objref_twoimgs(*paths, self.settings)

    def _create_data_imgnum(self):
        '''
        Здесь создается список объектных и референсных изображений,
        если данные представлены в виде картинок для референсного канала
        и текстового файла со значениями для объектного канала
        '''
        ref_img_paths = get_images(self.settings.REF_DIR, self.settings)
        self.obj_data = np.loadtxt(self.settings.OBJ_FILE).flatten()

        for i, path in enumerate(ref_img_paths):
            self.ref_data[i, :, :] = get_ref_imgnum(path, self.settings)


class ImgAnalyser:

    def __init__(self, settings_file, binning_order=1, n_images=0, parallel_njobs=-1):
        '''
        ARGUMENTS
        ---------

        settings_file -- путь к файлу с настройками эксперимента в формате json
        '''
        self.parallel_njobs = parallel_njobs
        self.settings = GISettings(settings_file)
        self.settings.BINNING = binning_order
        if n_images:
            self.settings.N = n_images

        log.info('Experiment settings:\n' + json.dumps(
            self.settings.__dict__, indent=4)[2:-2])

        self.N = self.settings.N
        self.Ny, self.Nx = (crop_shape(self.settings.REF_CROP) // binning_order).astype(int)
        log.info(f'Reference images size is {self.Nx}x{self.Ny}')

        self.obj_data = np.zeros(self.N)
        self.ref_data = np.zeros((self.N, self.Ny, self.Nx), dtype=np.uint8)
        self.gi = np.zeros((self.Ny, self.Nx), dtype=np.float32)

        self.sc = np.zeros((self.Ny, self.Nx), dtype=np.float32)
        self.times = np.linspace(
            0, self.settings.TCPOINTS / self.settings.FREQ,
            self.settings.TCPOINTS)
        self.tc = np.ones(self.settings.TCPOINTS)
        self.cd = np.zeros((self.Ny, self.Nx), dtype=np.float32)
        self.g2 = 0
        # !!!IMPORTANT: Side effects to ref_data and obj_data
        log.info('Loading obj and ref data')
        t = time.time()
        ObjRefGenerator(self.settings, self.ref_data, self.obj_data, self.bo)
        log.info(
            f'Obj and ref data loaded. Elapsed time {(time.time() - t):.3f} s')

    def calculate_ghostimage(self):
        '''
        Расчет корреляции между последовательностью
        суммарных сигналов в объектном плече
        и поточечными последовательностями сигналов в референсном плече
        '''
        log.info('Calculating ghost image')
        t = time.time()

        self.gi = data_correlation(self.obj_data, self.ref_data,
                                   self.parallel_njobs)
        log.info(
            f'Ghost image calculated. Elapsed time {(time.time() - t):.3f} s')

    @property
    def diff(self):
        '''
        Расчет разности между последовательностями
        объектных и референсных изображений
        '''
        return np.mean(self.ref_data, axis=0)

    def calculate_xycorr(self, x=0, y=0):
        '''
        Расчет функции когерентности или поперечной корреляции
        '''
        log.info('Calculating spatial correlation function')
        t = time.time()

        if x == 0:
            x = self.Nx // 2
        if y == 0:
            y = self.Ny // 2
        point_data = self.ref_data[:, y, x]
        self.sc = data_correlation(point_data, self.ref_data,
                                   self.parallel_njobs)
        log.info(
            f'Spatial correlation function calculated. Elapsed time {(time.time() - t):.3f} s')

    def calculate_xycorr_widths(self, window_points: int = 50,
                                nx: int = 10, ny: int = 10,
                                n_jobs: int = -2):
        """
        Расчет ширин функции когерентности или
        поперечной корреляции для разных пикселей

        Parameters
        ----------
        window_points : int, optional
            Width of the calculation window (in points)
            for the each point of calculation.
            The default is 50.
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
        log.info('Calculating spatial correlation function width in different pixels')
        t = time.time()

        X = np.arange(- nx // 2, nx // 2) + self.Nx // 2
        Y = np.arange(-ny // 2, ny // 2) + self.Ny // 2
        points = np.array(np.meshgrid(X, Y)).T.reshape(-1, 2)
        w = window_points

        _rawd = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(xycorr)(self, p, w) for p in points)
        _rawdx = np.array([w[0] for w in _rawd]).reshape((ny, nx))
        _rawdy = np.array([w[1] for w in _rawd]).reshape((ny, nx))
        self.sc_widths = (_rawdx, _rawdy)
        log.info(
            f'Spatial correlation function widths calculated. Elapsed time {(time.time() - t):.3f} s')

    def calculate_timecorr(self, npoints=100):
        log.info('Calculating time correlation function')
        t = time.time()

        def cf1d(data, i):
            return np.nan_to_num(np.corrcoef(data[:-i], data[i:])[0, 1])

        rdim = self.Nx * self.Ny // 2
        ravel_data = self.ref_data.reshape((self.N, rdim * 2))
        ravel_data = ravel_data[:, rdim - npoints // 2: rdim + npoints // 2]
        self.tc[1:] = np.mean([np.apply_along_axis(cf1d, 1, ravel_data, i)
                               for i in range(1, self.settings.TCPOINTS)],
                              axis=-1)
        log.info(
            f'Spatial time function calculated. Elapsed time {(time.time() - t):.3f} s')

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
    def xycorr_widths_data(self):
        return self.sc_widths

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
            'Область регистрации объектного пучка составляет {0:d} на {1:d} пикселей'.format(
                *crop_shape(sets.OBJ_CROP)),
            f'Область регистрации опорного пучка составляет {self.Nx:d} на {self.Ny:d} пикселей',
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
        Добавление изображений в виде
        массивов numpy.ndarray или list в список отображения
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
