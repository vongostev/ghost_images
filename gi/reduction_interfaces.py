# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:30:42 2021

@author: vonGostev
"""
import __init__
import logging
from collections import defaultdict
from os import remove
from time import perf_counter

import matplotlib.pyplot as plt

from reduction import GIDenseReduction, GISparseReduction, GIDenseReductionIter
from measurement_model import GIMeasurementModel, pad_or_trim_to_shape, TraditionalGI
from compressive_sensing import (GICompressiveSensingL1DCT,
                                 GICompressiveSensingL1Haar, GICompressiveTC2,
                                 GICompressiveAnisotropicTotalVariation,
                                 GICompressiveAnisotropicTotalVariation2)

logger = logging.getLogger("Fiber-GI-reduction")
logger.propagate = False  # Do not propagate to the root logger
logger.setLevel(logging.INFO)
logger.handlers = []
# fh = logging.FileHandler("processing.log")
# fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
ch.setFormatter(formatter)
# logger.addHandler(fh)
logger.addHandler(ch)


def show_single_method(obj_data, ref_data) -> None:
    measurement = obj_data
    model = GIMeasurementModel(ref_data)

    # if isinstance(img_id, int):
    #     src_img = pad_or_trim_to_shape(load_demo_image(img_id), model.img_shape)

    # "dct", no noise: 1e-5
    # "dct", 1e-2 noise: at least 1

    # basis = "eig"
    # # thr_coeff_values = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 1.]
    # # thr_coeff_values = [10, 100, 1e3, 1e4, 1e5]
    # thr_coeff_values = [5e-4, 5e-3]
    # for thr_coeff in thr_coeff_values:
    #     result = sparse_reduction(measurement, mt_op, img_shape,
    #                           thresholding_coeff=thr_coeff, basis=basis)
    #     diff_sq = np.linalg.norm(result - src_img)**2
    #     save_image_for_show(result.clip(0, None), "red_sparse_{}_{:.0e}_{}_{:.0e}".format(
    #         img_id, noise_var, basis, thr_coeff
    #     ), rescale=True)
    #     with open("red_sparse_diff.txt", "a", encoding="utf-8") as f:
    #         f.write("{}\t{:.1g}\t{}\t{:.1g}\t{:.3g}\n".format(img_id, noise_var, basis, thr_coeff, diff_sq))

    # result = GICompressiveAnisotropicTotalVariation2(model)(measurement, alpha=1e-5)
    # # print(np.linalg.norm(result - src_img)**2)

    # result = GISparseReduction(model)(measurement, thresholding_coeff=0.1,
    #                                    basis="eig")
    # print(np.linalg.norm(result - src_img)**2)

    estimator = GIDenseReductionIter(model)
    result = estimator(measurement, n_iter=100000)

    estimator_bis = GISparseReduction(model)
    result_bis = estimator_bis(measurement, 1e6, skip_tv=False)
    # result_bis = estimator_bis(measurement, 5e6, skip_tv=False)
    # result_bis = estimator_bis(measurement, 1e8, skip_tv=True)

    result2 = TraditionalGI(model)(measurement)

    plt.subplot(131)
    plt.imshow(result, cmap=plt.cm.gray)  # pylint: disable=E1101
    plt.title("Линейная редукция измерения")
    plt.subplot(132)
    plt.imshow(result_bis, cmap=plt.cm.gray)  # pylint: disable=E1101
    plt.title("Редукция измерения, предлагаемый метод")
    plt.subplot(133)
    plt.imshow(result2, cmap=plt.cm.gray)  # pylint: disable=E1101
    plt.title("Обычное ФИ")

    plt.show()


def show_methods(obj_data, ref_data, show: bool = True) -> None:
    t_start = perf_counter()
    measurement = obj_data
    model = GIMeasurementModel(ref_data)

    logger.info("Setting up took %.3g s", perf_counter() - t_start)

    # TODO Think of a better identifier
    img_id = None
    src_img = None
    noise_var = 0.
    size = model.pixel_size * model.img_shape[0]/2

    estimates = {}

    t_estim_part = perf_counter()
    estimates[TraditionalGI.name] = TraditionalGI(model)(measurement)
    logger.info("Traditional GI formation took %.3g s.",
                perf_counter() - t_estim_part)
    plt.imshow(estimates[TraditionalGI.name], cmap=plt.cm.gray,  # pylint: disable=E1101
               extent=[-size, size, -size, size])
    plt.xlabel("x, мкм")
    plt.ylabel("y, мкм")
    plt.show()
    # whitened_measurement = noise_var**0.5 * measurement

    alpha_values = {("tc2", 3, 1e-1): 6e-3, ("tva2", 3, 1e-1): 0.158,
                    ("tc2", 2, 1e-1): 1e-3, ("tva2", 2, 1e-1): 0.158,
                    ("tc2", 6, 1e-1): 6e-3,
                    ("tc2", 7, 1e-1): 1e-3}

    alpha_values = defaultdict(lambda: 1e-5, alpha_values)
    # None, corresponding to alpha = 0+ would be a more accurate default value,
    # especially for no noise,
    # but this provides the same results faster.

    cs_processing_methods = [GICompressiveSensingL1DCT,
                             GICompressiveSensingL1Haar,
                             GICompressiveTC2,
                             GICompressiveAnisotropicTotalVariation,
                             # GICompressiveAnisotropicTotalVariation2
                             GIDenseReduction,
                             GISparseReduction
                             ]

    for processing_method in cs_processing_methods:
        t_estim_start = perf_counter()
        estimates[processing_method.name] = processing_method(model)(
            measurement,
            alpha=alpha_values[(processing_method.name,
                                img_id, float(noise_var))]
        )
        logger.info("Estimation using %s took %.3g s.", processing_method.name,
                    perf_counter() - t_estim_start)
        plt.imshow(estimates[processing_method.name], cmap=plt.cm.gray,  # pylint: disable=E1101
                   extent=[-size, size, -size, size])
        plt.xlabel("x, мкм")
        plt.ylabel("y, мкм")
        plt.show()

    tau_values = {(2, 0.0): 1.0, (2, 0.1): 1,
                  (3, 0.): 1e-05, (3, 0.1): 0.1,
                  (6, 0.): 1.0, (6, 0.1): 1,
                  (7, 0.): 10.0, (7, 0.1): 1.,
                  (8, 0.): 1e-05, (8, 0.1): 1e-5}
    tau_values = defaultdict(lambda: 1., tau_values)

    # processing_methods = [TraditionalGI] + \
    #     cs_processing_methods + [GIDenseReduction, GISparseReduction]

    # estimates[GIDenseReduction.name] = GIDenseReduction(model)(measurement)
    # estimates[GISparseReduction.name] = GISparseReduction(model)(
    #     measurement, tau_values[(img_id, noise_var)], basis="eig"
    # )
    t_end = perf_counter()
    logger.info("show_methods for %d patterns and %s shape took %.3g s",
                measurement.size, ref_data[0].shape, t_end - t_start)
