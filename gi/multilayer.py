# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 22:29:46 2022

@author: vonGostev
"""
from collections.abc import Iterable
from typing import Union, Callable, Any, Iterable as IterType
from functools import reduce
import cv2

from .experiment import getbackend


class FilterComposer:

    slots = ('__name__', '_fs', '_args', '_n', 'f')

    def __init__(self, funcs: Union[list, Callable], fargs: Iterable, strategy: str = '*'):
        super().__init__()

        if funcs is None or (isinstance(funcs, Iterable) and len(funcs) == 0):
            raise ValueError("Function list is empty")

        if strategy not in ['+', '*']:
            raise ValueError('Unknown strategy of filters composition')

        self._fs = funcs
        self._args = fargs
        self._n = len(self._fs) if isinstance(self._fs, Iterable) else 1.

        if strategy == '+':
            self._r = lambda x, y: x.astype(float) + y.astype(float)
        elif strategy == '*':
            self._r = lambda x, y: x.astype(float) * y.astype(float)

        if callable(funcs):
            self.f = lambda X, Y: funcs(X, Y, *fargs)
            self.__name__ = f"FilterComposer({strategy})[{funcs.__name__}]"
            return

        _sfuncs = ', '.join([
            f.__name__ + '(' +
            str(0 if fargs == () or not isinstance(fargs[0], Iterable) else i) +
            ')' for i, f in enumerate(funcs)])
        self.__name__ = f"FilterComposer({strategy})[{_sfuncs}]"

        if len(fargs) == 0:
            self.f = self.__construct_from_flatten_fargs(funcs, fargs)
            return

        if not isinstance(fargs[0], Iterable):
            self.f = self.__construct_from_flatten_fargs(funcs, fargs)
            return
        else:
            if len(funcs) != len(fargs):
                raise ValueError(
                    f"Length {len(funcs)} of function list is not equal {len(fargs)}")
            self.f = self.__construct_from_different_fargs(funcs, fargs)
            return

    def __construct_from_flatten_fargs(self, funcs, fargs):
        def _f(X, Y):
            return reduce(self._r, [f(X, Y, *fargs) for f in funcs])
        return _f

    def __construct_from_different_fargs(self, funcs, fargs):
        def _f(X, Y):
            return reduce(
                self._r, [f(X, Y, *args) for f, args in zip(funcs, fargs)])
        return _f

    def __call__(self, X, Y, *args, **kwargs):
        return self.f(X, Y) / self._n

    def __repr__(self):
        return self.__name__


def filter_scale(f: Callable, scale_factor: float = 1) -> Callable:

    def fi(X: Union['numpy.ndarray', 'cupy.ndarray'],
           Y: Union['numpy.ndarray', 'cupy.ndarray'], *args):
        return scale_factor * f(X, Y, *args)
    fi.__name__ = f'{f.__name__}_scl'
    return fi


def filter_inverse(f: Callable) -> Callable:

    def fi(X: Union['numpy.ndarray', 'cupy.ndarray'],
           Y: Union['numpy.ndarray', 'cupy.ndarray'], *args):
        return 1 - f(X, Y, *args)
    fi.__name__ = f'{f.__name__}_inv'
    return fi


def filter_from_img(path: str, npoints: int):
    img = cv2.imread(path, 0)
    img_rescaled = cv2.resize(img, dsize=(
        npoints, npoints), interpolation=cv2.INTER_LINEAR) / 255

    def fi(X: Union['numpy.ndarray', 'cupy.ndarray'],
           Y: Union['numpy.ndarray', 'cupy.ndarray'], *args):
        xp = getbackend(X)
        return xp.asarray(img_rescaled)
    fi.__name__ = f'img:{path}'
    return fi


def multilayer_object(filters: IterType[Callable],
                      args: IterType[Union[IterType, Any]],
                      strategy: str = '*') -> IterType[FilterComposer]:
    # O(n^2) algorithm
    partial_layers = []
    for i in range(1, len(filters) + 1):
        lfuncs = list(map(filter_inverse, filters[:i-1])) + [filters[i-1]]
        if args == () or not isinstance(args[0], Iterable):
            largs = args
        else:
            largs = args[:i]
        partial_layers.append((lfuncs, largs))
    return [FilterComposer(*layer, strategy=strategy) for layer in partial_layers]
