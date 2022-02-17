# -*- coding: utf-8 -*-
'''
Created on Mon Jun  7 17:40:40 2021

@author: von.gostev
'''
from os import listdir
from os.path import isfile, join, dirname, realpath, basename
from types import ModuleType
import sys
import time
import numpy as np
from functools import cached_property
from tqdm import tqdm
import psutil

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import json
from cv2 import resize, imread as cv2imread
# !!! TODO compare with cv2.resize
# from skimage.transform import downscale_local_mean

from joblib import Parallel, delayed, wrap_non_picklable_objects
from logging import Logger, StreamHandler, Formatter


log = Logger('EXP')

handler = StreamHandler(sys.stdout)
handler.setLevel(20)
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

try:
    import cupy as cp
    _using_cupy = True
except ImportError as E:
    _using_cupy = False
    log.warn(
        f"ImportError : {E}, 'use_cupy' key is meaningless.")

IMG_CROP_DATA = 0
IMG_IMG_DATA = 1
IMG_NUM_DATA = 2


def norm2_memeff(x, axis=None, chunks=8, backend=None):
    if backend is None:
        backend = getbackend(x)

    if backend.__name__ == 'numpy':
        avmem = psutil.virtual_memory().available
    elif backend.__name__ == 'cupy':
        avmem = backend.get_default_memory_pool().free_bytes()
    if x.size > avmem / 3:
        log.debug('Memory efficient norm. Array is chunked')
        xlen = x.shape[0]
        indxs = np.linspace(0, xlen, chunks, dtype=np.int32)
        ssum = 0
        for i in range(chunks - 1):
            ssum += backend.sum(x[indxs[i]:indxs[i+1]] ** 2, axis=axis)
        return backend.sqrt(ssum)

    return backend.linalg.norm(x, axis=axis)


def getbackend(obj: object) -> ModuleType:
    module_name = type(obj).__module__.split('.')[0]
    if module_name in ['numpy', 'cupy']:
        return __import__(module_name)
    else:
        raise ValueError(
            f'Unknown backend `{module_name}`. Check object `{type(obj)}` type.')


def low_res(img, n):
    x, y = img.shape
    return resize(img, (y // n, x // n))
    # return downscale_local_mean(img, (n, n))


def crop(img, c):
    return img[c[2]:c[3], c[0]:c[1]]


def crop_shape(c):
    return np.asarray([c[3] - c[2], c[1] - c[0]])


def imread(path, binning_order, crop_shape):
    img = cv2imread(path, 0)
    img = crop(img, crop_shape)
    if binning_order > 1:
        img = low_res(img, binning_order)
    return img


class GISettings:
    """
    Class to parse settings files:
        settings of the experiment
    """
    PREFIX = ""

    def __init__(self, path):
        log.info(f'Reading settings file {path}')
        self.settings_path = dirname(path)

        with open(path, 'r') as f:
            settings = json.load(f)
        for attr in settings:
            setattr(self, attr, settings[attr])

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


def sort_by_num(fname, prefix='pattern'):
    name = basename(fname).split('.')[0]
    num = int(name.replace(prefix, ''))
    return int(num)


def find_images(dir_name: str, img_num: str, img_format: str, img_prefix: str):
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
    img_prefix : str
        Constant prefix of images names.
    Returns
    -------
    list
        List of paths to images in dir_name.

    """
    img_list = [join(dir_name, f) for f in listdir(dir_name)
                if isfile(join(dir_name, f)) and f.endswith(img_format)]

    def fsort(s): return sort_by_num(s, img_prefix)
    img_list = sorted(img_list, key=fsort) if img_prefix else img_list
    return img_list[:img_num]


def get_images(dir_name: str, settings: GISettings):
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
    img_paths = find_images(dir_name, img_num, img_format, settings.PREFIX)
    if len(img_paths) == 0:
        raise IOError(f'Не найдено изображений в выбранной папке: {dir_name}')
    return img_paths


def get_diff_img(ref_img, obj_img, settings: GISettings):
    if ref_img.shape != obj_img.shape:
        ny_ref, nx_ref = ref_img.shape
        ny_obj, nx_obj = obj_img.shape
        nx = np.min(nx_ref, nx_obj)
        ny = np.min(ny_ref, ny_obj)
        return ref_img[:ny, :nx] - obj_img[:ny, :nx]
    else:
        return ref_img - obj_img


@wrap_non_picklable_objects
def get_objref_imgcrop(path, settings: GISettings):
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
    return get_objref_twoimgs([path, path], settings)


@wrap_non_picklable_objects
def get_objref_twoimgs(ref_obj_paths: list, settings: GISettings):
    """
    Construct single reference and objective data pair
    if reference and objective data are images in different directories

    Parameters
    ----------
    ref_obj_paths : list(str, str)
        ref_obj_paths[0] <-> Path to a directory containing reference channel images.
        ref_obj_paths[1] <-> Path to a directory containing objective channel images.
    settings : GISettings
        Parsed settings of the experiment.

    Returns
    -------
    ref_data: np.ndarray
        An array with reference channel data.
    obj_data : np.ndarray
        An array with objective channel data.

    """
    ref_path, obj_path = ref_obj_paths
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

    return ref_data, obj_data


@wrap_non_picklable_objects
def get_ref_imgnum(ref_path: str, settings: GISettings):
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


def autocorr1d(data, i, backend=None):
    if backend is None:
        backend = getbackend(data)
    obj_data = data[:-i]
    ref_data = data[i:]
    od = obj_data - obj_data.mean(dtype=np.float32)
    rd = ref_data - ref_data.mean(dtype=np.float32)
    s1 = (od * rd).sum()
    s2 = norm2_memeff(od) * norm2_memeff(rd)
    return backend.nan_to_num(s1 / s2)


def corr1d3d(obj_data, ref_data, backend=None):
    if backend is None:
        backend = getbackend(obj_data)
    log.debug(
        'Compute correlation function using `np.einsum`')
    od = obj_data - obj_data.mean(dtype=np.float32)
    rm = ref_data.mean(axis=0, dtype=np.float32)
    s1 = backend.einsum('i,ijk->jk', od, ref_data, dtype=np.float32)
    s2 = norm2_memeff(od) * (norm2_memeff(ref_data, axis=0) - rm)
    return backend.nan_to_num(s1 / s2)


def corr3d3d(obj_data_3d, ref_data, backend=None):
    if backend is None:
        backend = getbackend(obj_data_3d)
    log.debug(
        'Compute correlation function using `np.einsum`')
    od = obj_data_3d - obj_data_3d.mean(axis=0, dtype=np.float32)
    rd = ref_data - ref_data.mean(axis=0, dtype=np.float32)
    s1 = backend.einsum('inm,ijk->nmjk', od, rd)
    s2 = backend.linalg.norm(od, axis=0) * backend.linalg.norm(rd, axis=0)
    return backend.nan_to_num(s1 / s2)


def FWHM(data1d, Xc=None, backend=None):
    if backend is None:
        backend = getbackend(data1d)
    return backend.sum(data1d > 0.5)


def xycorr_width(sc, p=None, backend=None):
    if backend is None:
        backend = getbackend(sc)
    sc[sc < 0.5] = 0
    if backend.all(sc == 0):
        return backend.asarray([np.nan, np.nan])
    if p is None:
        ny, nx = sc.shape
        # Calculate centroid
        Y, X = backend.mgrid[-ny // 2:ny // 2, -nx // 2:nx // 2]
        Xc = int(backend.average(X, weights=sc)) + nx // 2
        Yc = int(backend.average(Y, weights=sc)) + ny // 2
    else:
        Yc, Xc = p
    xslice = sc[:, Xc]
    yslice = sc[Yc]
    return backend.asarray(
        [FWHM(xslice, Xc, backend), FWHM(yslice, Yc, backend)])


@wrap_non_picklable_objects
def xycorr(self, p, w):
    x, y = p
    lx = max(x - w // 2, 0)
    ly = max(y - w // 2, 0)
    tx = min(x + w // 2, self.Nx)
    ty = min(y + w // 2, self.Ny)
    sc = corr1d3d(self.ref_data[:, y, x], self.ref_data[:, ly:ty, lx:tx])
    return xycorr_width(sc, backend=self.xp)


class ObjRefGenerator:

    xp = np

    def __init__(self, settings: GISettings, binning_order: int = 1,
                 parallel_njobs: int = 1, use_cupy=False):
        """
        Здесь создается список объектных и референсных изображений
        self.obj_data -- изображения объекта
        self.ref_data -- изображения референсного пучка

        Parameters
        ----------
        settings : GISettings
            Набор настроек.
        binning_order : int, optional
            Порядок биннинга. The default is 1.
        parallel_njobs : int, optional
            Число потоков вычислений. The default is 1.

        Returns
        -------
        None.

        """
        if use_cupy and _using_cupy:
            self.xp = cp

        self.settings = settings
        self.bo = binning_order
        self.njobs = parallel_njobs

        if self.settings.FORMAT == IMG_CROP_DATA:
            self._create_data_crop()
        elif self.settings.FORMAT == IMG_IMG_DATA:
            self._create_data_twoimgs()
        elif self.settings.FORMAT == IMG_NUM_DATA:
            self._create_data_imgnum()

    def __parallel_read(self, read_func, paths):
        res = Parallel(n_jobs=self.njobs, backend='threading')(
            delayed(read_func)(path, self.settings)
            for path in tqdm(paths, position=0, leave=True))
        print()
        return res

    def _create_data_crop(self):
        '''
        Здесь создается список объектных и референсных изображений,
        если данные представлены в виде картинок с двумя каналами одновременно
        '''
        img_paths = get_images(self.settings.DIR, self.settings)
        data_list = self.__parallel_read(
            get_objref_imgcrop, img_paths)
        ref_data_list, obj_data_list = zip(*data_list)
        self.ref_data = self.xp.asarray(ref_data_list, dtype=np.uint8)
        self.obj_data = self.xp.asarray(obj_data_list, dtype=np.float32)

    def _create_data_twoimgs(self):
        '''
        Здесь создается список объектных и референсных изображений,
        если данные представлены в виде отдельных картинок на каждый канал
        '''
        ref_img_paths = get_images(self.settings.REF_DIR, self.settings)
        obj_img_paths = get_images(self.settings.OBJ_DIR, self.settings)
        img_paths = zip(ref_img_paths, obj_img_paths)
        data_list = self.__parallel_read(
            get_objref_twoimgs, img_paths)
        ref_data_list, obj_data_list = zip(*data_list)
        self.ref_data = self.xp.asarray(ref_data_list, dtype=np.uint8)
        self.obj_data = self.xp.asarray(obj_data_list, dtype=np.float32)

    def _create_data_imgnum(self):
        '''
        Здесь создается список объектных и референсных изображений,
        если данные представлены в виде картинок для референсного канала
        и текстового файла со значениями для объектного канала
        '''
        ref_img_paths = get_images(self.settings.REF_DIR, self.settings)
        ref_data_list = self.__parallel_read(
            get_ref_imgnum, ref_img_paths)
        self.ref_data = self.xp.asarray(ref_data_list, dtype=np.uint8)
        obj_file_name_split = self.settings.OBJ_FILE.split('.')
        ext = obj_file_name_split[-1]
        if ext == 'npy':
            obj_data = self.xp.load(
                self.settings.OBJ_FILE)
        elif ext in ['txt', 'csv', 'dat']:
            obj_data = self.xp.loadtxt(
                self.settings.OBJ_FILE)
        else:
            raise NotImplementedError(
                f'Objective channel data must be in `npy`, `txt`, `csv`, or `dat` format, not `{ext}`')
        self.obj_data = obj_data.flatten()[:self.settings.N].astype(np.float32)

    def unpack(self):
        return self.ref_data, self.obj_data


class GIExpDataProcessor:

    _g2 = None
    sc_widths = None
    xp = np

    def __init__(self, settings_file: str, binning_order: int = 1,
                 n_images: int = 0, parallel_njobs: int = -1,
                 parallel_reading: bool = True, use_cupy=False):

        self.use_cupy = use_cupy
        if use_cupy and _using_cupy:
            self.xp = cp

        self.parallel_njobs = parallel_njobs
        self.settings = GISettings(settings_file)
        self.settings.BINNING = binning_order
        if n_images:
            self.settings.N = n_images

        log.info('Experiment settings:\n' + json.dumps(
            self.settings.__dict__, indent=4)[2:-2])

        self.imgs_number = self.settings.N
        self.Ny, self.Nx = (crop_shape(
            self.settings.REF_CROP) // binning_order).astype(int)
        log.info(f'Reference images size is {self.Nx}x{self.Ny}')

        self.obj_data = self.xp.zeros(self.imgs_number, dtype=np.float32)
        self.ref_data = self.xp.zeros((self.imgs_number, self.Ny, self.Nx),
                                      dtype=np.uint8)
        self.gi = self.xp.zeros((self.Ny, self.Nx), dtype=np.float32)

        self.sc = self.xp.zeros((self.Ny, self.Nx), dtype=np.float32)
        self.times = np.linspace(
            0, self.settings.TCPOINTS / self.settings.FREQ,
            self.settings.TCPOINTS)
        self.tc = self.xp.ones(self.settings.TCPOINTS)
        self.cd = self.xp.zeros((self.Ny, self.Nx), dtype=np.float32)

        log.info('Loading obj and ref data')
        t = time.time()
        data_generator = ObjRefGenerator(
            self.settings, binning_order,
            parallel_njobs if parallel_reading else 1,
            use_cupy=self.use_cupy and _using_cupy)
        self.ref_data, self.obj_data = data_generator.unpack()
        # Update Nx, Ny because of rounding in low_res
        self.Ny, self.Nx = self.ref_data.shape[1:]
        log.info(
            f'Obj and ref data loaded. Elapsed time {(time.time() - t):.3f} s')

    def _np(self, data):
        """Convert cupy or numpy arrays to numpy array.

        Parameters
        ----------
        data : Tuple[numpy.ndarray, cupy.ndarray]
            Input data.

        Returns
        -------
        data : numpy.ndarray
            Converted data.

        """
        # Return numpy array from numpy or cupy array
        if self.xp.__name__ == 'cupy':
            return data.get()
        return data

    def calculate_ghostimage(self, data_start=None, data_end=None):
        '''
        Расчет корреляции между последовательностью
        суммарных сигналов в объектном плече
        и поточечными последовательностями сигналов в референсном плече
        '''
        if data_start is None:
            data_start = 0
        if data_end is None:
            data_end = self.imgs_number
        log.info('Calculating ghost image')
        t = time.time()

        self.gi = corr1d3d(self.obj_data[data_start:data_end],
                           self.ref_data[data_start:data_end])
        log.info(
            f'Ghost image calculated. Elapsed time {(time.time() - t):.3f} s')

    def calculate_xycorr(self, x=0, y=0, window_points: int = None):
        '''
        Расчет функции когерентности или поперечной корреляции
        '''
        log.info('Calculating spatial correlation function')
        t = time.time()

        w = window_points

        if x == 0:
            x = self.Nx // 2
        if y == 0:
            y = self.Ny // 2
        self.sc_point = (y, x)
        point_data = self.ref_data[:, y, x]
        self.sc = corr1d3d(
            point_data, self.ref_data if w is None else self.ref_data[:, y-w:y+w, x-w:x+w])
        log.info(
            f'Spatial correlation function calculated. Elapsed time {(time.time() - t):.3f} s')

    def calculate_xycorr_widths(self, window_points: int = 10,
                                nx: int = 10, ny: int = 10):
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

        Return
        ---------

        Two arrays with xy correlation function widths: by x  and by y.

        """
        log.info(
            'Calculating spatial correlation function width in different pixels')
        t = time.time()
        self.sc_widths = self.xp.empty((nx * ny, 2))
        points = self.xp.mgrid[
            (self.Nx - nx) // 2:(self.Nx + nx) // 2,
            (self.Ny - ny) // 2:(self.Ny + ny) // 2].T.reshape((-1, 2))
        for i, p in tqdm(enumerate(points), position=0, leave=True):
            self.sc_widths[i] = xycorr(self, p, window_points)
        print()
        self.sc_widths = self.sc_widths.swapaxes(1, 0).reshape((2, ny, nx))
        log.info(
            f'Spatial correlation function widths calculated. Elapsed time {(time.time() - t):.3f} s')

    def calculate_timecorr(self, tcpoints=None):
        log.info('Calculating time correlation function')
        t = time.time()

        if tcpoints is None:
            if hasattr(self, "settings"):
                tcpoints = self.settings.TCPOINTS
            else:
                raise ValueError(
                    'Number of time correlation points is undefined.' +
                    f'Please set {type(self).__name__}.settings.TCPOINTS or `tcpoints` argument')

        ravel_data = self.ref_data.mean(axis=(1, 2))
        self.tc[1:] = self.xp.asarray(
            [autocorr1d(ravel_data, i, self.xp) for i in self.xp.arange(1, tcpoints)])
        log.info(
            f'Time correlation function calculated. Elapsed time {(time.time() - t):.3f} s')

    def calculate_contrast(self):
        _gi = self.xp.abs(self.gi)
        _mean_gi = self.xp.min(_gi)
        self.cd = (_gi - _mean_gi) / (_gi + _mean_gi)
        log.info('Ghost image contrast calculated')

    def calculate_all(self):
        self.calculate_ghostimage()
        self.calculate_contrast()
        self.calculate_xycorr()
        self.calculate_timecorr()
        self.calculate_g2()

    def calculate_g2(self, noise=0):
        self._g2 = self.xp.mean((self.ref_data - noise)**2, axis=0) / \
            self.xp.mean(self.ref_data - noise, axis=0)**2

    @property
    def g2_data(self):
        return self._np(self._g2)

    @cached_property
    def g2(self):
        return self._g2.mean()

    @property
    def ghost_data(self):
        return self._np(self.gi)

    @property
    def timecorr_data(self):
        return self._np(self.tc)

    @property
    def xycorr_data(self):
        return self._np(self.sc)

    @property
    def xycorr_widths_data(self):
        return self._np(self.sc_widths)

    @property
    def contrast_data(self):
        return self._np(self.cd)

    @cached_property
    def contrast(self):
        return self.contrast_data.mean()

    @cached_property
    def xycorr_width(self):
        if self.sc_widths is None:
            return xycorr_width(self.xycorr_data, self.sc_point, self.xp)
        else:
            return self.sc_widths[
                self.sc_widths > 0].reshape((2, -1)).mean(axis=1)

    @cached_property
    def timecorr_width(self):
        return FWHM(self.tc) * self.times[1]


class ImgViewer:

    def __init__(self, init):
        '''
        Инициализация изображения по пути к изображению
            или принятому массиву пикселей numpy.ndarray
        '''
        if type(init) == str:
            self.path = init
            self.data = [cv2imread(self.path, 0)]
        elif type(init) == np.ndarray:
            self.path = 'generated_img'
            self.data = [np.abs(init)]

    def accumulate(self, data):
        '''
        Добавление изображений в виде
        массивов numpy.ndarray или list в список отображения
        '''
        if type(data) == list:
            data = np.asarray(data)
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
