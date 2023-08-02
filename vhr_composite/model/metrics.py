from vhr_composite.model import kernels

import logging
import time
import os
from typing import Tuple

import numpy as np
import numba as nb
import xarray as xr


CLASS_0: int = 0
CLASS_1: int = 1
CLASS_2: int = 2
CLASS_0_ALIAS: int = 1
CLASS_1_ALIAS: int = 2
CLASS_2_ALIAS: int = 4
HOST_DTYPE: np.dtype = np.uint8
HOST_FILL: np.uint32 = np.uint32((2**32)-1)
HOST_FILL_LEGACY: np.uint32 = np.uint32(10)
NODATA_LOWER_BOUND = 0
NODATA_UPPER_BOUND = HOST_FILL
LAYER_AXIS: int = 2
LAYER_COUNT: int = 3
CLASS_COUNT: int = 3
Y_AXIS: int = 0
X_AXIS: int = 1
Z_AXIS: int = 2
SUM_NO_MODE: int = 0
NO_DATA: int = 10


# CPU Algs
def mode(arr, gpu=True):
    """
    CPU kernel to calculate the simple mode along the Z dimension of a
    3D numpy array using a histogram-based method.

    Parameters:
    -----------
    arr: 3D numpy array
        Input array of shape (Y, X, Z).
    mode_values: 2D numpy array
        Output array to store the mode values along the Z dimension,
        of shape (Y, X).

    Returns:
    --------
    None
    """

    input_array_shape = arr.shape

    if len(arr.shape) < 3:
        error_msg = 'Must be more than one layer in time dimension'
        error_msg = f'{error_msg}. Input shape: {input_array_shape}'
        raise RuntimeError(error_msg)

    output_mode_shape = arr.shape[:2]
    output_dtype = arr.dtype

    output_mode_array = np.zeros(output_mode_shape, dtype=output_dtype)

    if gpu:
        raise NotImplementedError()
    else:
        output_mode_array = kernels.calculate_mode_cpu_kernel(
            arr, output_mode_array)
    return output_mode_array


def number_observations(arr, nodata=HOST_FILL, gpu=True):
    """
    CPU kernel to calculate the number of valid occurances
    along the Z dimension of a 3D numpy array.

    Parameters:
    -----------
    arr: 3D numpy array
        Input array of shape (Y, X, Z).
    mode_values: 2D numpy array
        Output array to store the mode values along the Z dimension,
        of shape (Y, X).

    Returns:
    --------
    None
    """

    input_array_shape = arr.shape

    if len(arr.shape) < 3:
        error_msg = 'Must be more than one layer in time dimension'
        error_msg = f'{error_msg}. Input shape: {input_array_shape}'
        raise RuntimeError(error_msg)

    output_nobs_shape = arr.shape[:2]
    output_dtype = arr.dtype

    output_nobs_array = np.zeros(output_nobs_shape, dtype=output_dtype)

    if gpu:
        raise NotImplementedError()
    else:
        output_nobs_array = kernels.calculate_sum_across_axis(
            arr, output_nobs_array, nodata=nodata)

    output_nobs_array = np.expand_dims(output_nobs_array, axis=0)
    return output_nobs_array


# Define the GPU kernel using Numba
def multi_mode(arr, nodata=HOST_FILL, gpu=True):
    """_summary_

    Args:
        arr (np.ndarray):
            Input array from which to calculate multi-mode product.
        nodata (np.uint32, optional): integer
            No-data value to ignore in calculations.
            Defaults to HOST_FILL.
        gpu (bool, optional):
            If True, will attempt to run the calculation using a GPU.
            Defaults to True.

    Raises:
        RuntimeError:
            If the input array has less than 3 dimensions.
        RuntimeError:
            If the no-data value is not within range.

    Returns:
        np.ndarray: Multi-mode result
    """

    input_array_shape = arr.shape

    if len(arr.shape) < 3:
        error_msg = 'Must be more than one layer in time dimension'
        error_msg = f'{error_msg}. Input shape: {input_array_shape}'
        raise RuntimeError(error_msg)

    mm_result_shape = arr.shape[:2]
    result_dtype = np.uint64
    if not nodata:
        nodata = HOST_FILL
    if (nodata <= NODATA_LOWER_BOUND) or (nodata > NODATA_UPPER_BOUND):
        error_msg = 'no-data value must be within range' + \
            f' ({NODATA_LOWER_BOUND}, {NODATA_UPPER_BOUND}'
        raise RuntimeError(error_msg)

    mm_result_array = np.zeros(mm_result_shape, dtype=result_dtype)

    if gpu:
        mm_result_array_gpu = nb.cuda.to_device(mm_result_array)
        arr_gpu = nb.cuda.to_device(arr)
        threadsperblock = (16, 16)
        blockspergrid_x = (
            mm_result_shape[1] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (
            mm_result_shape[0] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        kernels.calculate_multi_mode_gpu_kernel[blockspergrid,
                                                threadsperblock](
            arr_gpu, mm_result_array_gpu, nodata)
        mm_result_array = mm_result_array_gpu.copy_to_host()
    else:
        mm_result_array = kernels.calculate_multi_mode_cpu_kernel(
            arr, mm_result_array)

    mm_result_array = np.expand_dims(mm_result_array, axis=0)
    return mm_result_array


def max_(arr, gpu=True):
    """
    GPU kernel to calculate the mode along the Z
    dimension of a 3D array on the GPU using Numba
    with a histogram-based method.

    Parameters:
    -----------
    arr: 3D numpy array
        Input array of shape (N, N, Z).
    mode_values: 2D numpy array
        Output array to store the mode values
        along the Z dimension, of shape (N, N).

    Returns:
    --------
    None
    """

    input_array_shape = arr.shape

    if len(arr.shape) < 3:
        error_msg = 'Must be more than one layer in time dimension'
        error_msg = f'{error_msg}. Input shape: {input_array_shape}'
        raise RuntimeError(error_msg)

    max_result_shape = arr.shape[:2]
    result_dtype = arr.dtype  # np.uint64

    max_result_array = np.zeros(max_result_shape, dtype=result_dtype)

    if gpu:
        max_result_array_gpu = nb.cuda.to_device(max_result_array)
        arr_gpu = nb.cuda.to_device(arr)
        threadsperblock = (16, 16)
        blockspergrid_x = (
            max_result_shape[1] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (
            max_result_shape[0] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        kernels.calculate_max_gpu_kernel[blockspergrid, threadsperblock](
            arr_gpu, max_result_array_gpu)
        max_result_array = max_result_array_gpu.copy_to_host()
    else:
        max_result_array = kernels.calculate_max_cpu_kernel(
            arr, max_result_array)
    return max_result_array


def min_(arr, gpu=True):
    """
    GPU kernel to calculate the mode along the Z
    dimension of a 3D array on the GPU using Numba
    with a histogram-based method.

    Parameters:
    -----------
    arr: 3D numpy array
        Input array of shape (N, N, Z).
    mode_values: 2D numpy array
        Output array to store the mode values along
        the Z dimension, of shape (N, N).

    Returns:
    --------
    None
    """

    input_array_shape = arr.shape

    if len(arr.shape) < 3:
        error_msg = 'Must be more than one layer in time dimension'
        error_msg = f'{error_msg}. Input shape: {input_array_shape}'
        raise RuntimeError(error_msg)

    min_result_shape = arr.shape[:2]
    result_dtype = arr.dtype

    min_result_array = np.zeros(min_result_shape, dtype=result_dtype)

    if gpu:
        min_result_array_gpu = nb.cuda.to_device(min_result_array)
        arr_gpu = nb.cuda.to_device(arr)
        threadsperblock = (16, 16)
        blockspergrid_x = (
            min_result_shape[1] + threadsperblock[0] - 1) \
            // threadsperblock[0]
        blockspergrid_y = (
            min_result_shape[0] + threadsperblock[1] - 1) \
            // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        kernels.calculate_min_gpu_kernel[blockspergrid, threadsperblock](
            arr_gpu, min_result_array_gpu)
        min_result_array = min_result_array_gpu.copy_to_host()
    else:
        min_result_array = kernels.calculate_min_cpu_kernel(
            arr, min_result_array)
    return min_result_array


def mean(arr, gpu=True):
    raise NotImplementedError()


def std(arr, gpu=True):
    raise NotImplementedError()


@nb.njit
def _mode_sum_product(array: np.ndarray) -> Tuple[int, int]:
    """
    Multi-modal function
    Given a single dimension host array where each index is a class
    return all occurences of the max in the array such that if
    multiple classes have the same max, return the sum.
    :param array: np.ndarray, Flat array to calculate multi-mode
    :return max_element_to_return: int, element
    """
    max_element = np.max(array)
    sum_element_to_return = int(np.sum(array))
    if max_element == SUM_NO_MODE:
        return HOST_FILL_LEGACY, sum_element_to_return
    max_indices = np.argwhere(array == max_element).flatten()
    max_pl = np.where(max_indices == CLASS_0, CLASS_0_ALIAS, max_indices)
    max_pl = np.where(max_indices == CLASS_1, CLASS_1_ALIAS, max_pl)
    max_pl = np.where(max_indices == CLASS_2, CLASS_2_ALIAS, max_pl)
    max_element_to_return = int(np.sum(max_pl))
    return max_element_to_return, sum_element_to_return


@nb.njit
def _get_nobservations(array: np.ndarray) -> int:
    """
    Multi-modal function
    Given a single dimension host array where each index is a class
    return all occurences of the max in the array such that if
    multiple classes have the same max, return the sum.
    :param array: np.ndarray, Flat array to calculate multi-mode
    :return max_element_to_return: int, element
    """
    sum_element_to_return = np.sum(array)
    return sum_element_to_return


@nb.njit(parallel=True)
def _fast_iterate_mode(mode: np.ndarray, array: np.ndarray) -> np.ndarray:
    """
    Iterate through first two dims of 3d host array
    to get the mode for the z axis.
    """
    for y in nb.prange(array.shape[Y_AXIS]):
        for x in nb.prange(array.shape[X_AXIS]):
            mode[y, x] = _mode_sum_product(array[y, x, :])
    return mode


@nb.njit(parallel=True)
def _fast_iterate_mode_nobs(mode: np.ndarray,
                            nobs: np.ndarray,
                            array: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Iterate through first two dims of 3d host array
    to get the mode for the z axis.
    """
    for y in nb.prange(array.shape[Y_AXIS]):
        for x in nb.prange(array.shape[X_AXIS]):
            mode[y, x], nobs[y, x] = _mode_sum_product(array[y, x, :])
    return mode, nobs


def _get_sum(binary_class: str,
             logger: logging.Logger) -> np.ndarray:
    """
    Given a binary class occurance ndarray, sum along the z
    axis. Converts zarr dask array to device-backed dask array,
    performs computation, converts back to host array.
    """
    st = time.time()
    sum_class = binary_class.sum(axis=LAYER_AXIS, dtype=HOST_DTYPE)
    et = time.time()
    logger.info('Mode sum - Sum time: {}'.format(et-st))
    return sum_class


def calculate_mode(grid_cell_data_array: xr.DataArray,
                   classes: dict,
                   from_disk: bool = False,
                   calculate_nobservations: bool = True,
                   grid_cell_zarr_path: str = None,
                   logger: logging.Logger = None) -> np.ndarray:
    """
    Get the mode from a series of binary class occurance arrays.
    """
    num_classes = len(classes.keys())

    if from_disk:
        if not os.path.exists(grid_cell_zarr_path):
            msg = f'{grid_cell_zarr_path} does ' + \
                'cannot be found or does not exist.'
            raise FileNotFoundError(msg)
        grid_cell_data_array = xr.from_zarr(grid_cell_zarr_path)

    grid_cell_shape = grid_cell_data_array.shape
    class_sums_shape = (grid_cell_shape[Y_AXIS], grid_cell_shape[X_AXIS],
                        num_classes)

    class_sums = np.zeros(class_sums_shape, dtype=HOST_DTYPE)

    for class_id, class_value in classes.items():
        class_binary_array = xr.where(
            grid_cell_data_array == class_value, 1, 0).astype(HOST_DTYPE)
        class_sums[:, :, class_id] = _get_sum(class_binary_array, logger).data

    mode = np.zeros(
        (class_sums.shape[Y_AXIS], class_sums.shape[X_AXIS]), dtype=HOST_DTYPE)

    if calculate_nobservations:
        nobs = np.zeros(
            (class_sums.shape[Y_AXIS],
             class_sums.shape[X_AXIS]), dtype=HOST_DTYPE)

    logger.info('Computing mode')
    st = time.time()

    if calculate_nobservations:
        mode, nobs = _fast_iterate_mode_nobs(mode, nobs, class_sums)

        et = time.time()
        logger.info('Mode compute time {}'.format(round(et-st, 3)))
        return mode, nobs

    else:
        mode = _fast_iterate_mode(mode, class_sums)

        et = time.time()
        logger.info('Mode compute time {}'.format(round(et-st, 3)))
        return mode


if __name__ == '__main__':

    from scipy import stats

    # Setting up testing arrays
    test_classes = {0: 0, 1: 1, 2: 2}
    test_shape = (200, 200, 20)
    test_array = np.random.randint(0, 16, size=test_shape)
    test_array = test_array.astype(np.uint8)

    print(np.uint64(2**8))

    print(np.unique(test_array))
    print("{0:32b}".format(HOST_FILL))

    test_mode_control = stats.mode(test_array, axis=2)[0][:, :, 0]
    test_mode_results = mode(test_array, gpu=False)
    np.testing.assert_equal(test_mode_control, test_mode_results)

    del test_mode_control, test_mode_results

    test_mm_cpu_results = multi_mode(test_array, gpu=False)
    test_mm_gpu_results = multi_mode(test_array, gpu=True)
    np.testing.assert_equal(test_mm_cpu_results, test_mm_gpu_results)

    del test_mm_cpu_results, test_mm_gpu_results

    test_nobs_cpu_results = number_observations(test_array, gpu=False)

    del test_nobs_cpu_results
    del test_array

    test_array = np.random.rand(*test_shape).astype(np.float32)

    test_max_cpu_results = max_(test_array, gpu=False)
    test_max_gpu_results = max_(test_array, gpu=True)
    np.testing.assert_equal(test_max_cpu_results, test_max_gpu_results)

    del test_max_cpu_results, test_max_gpu_results, test_array

    test_array = np.random.rand(*test_shape).astype(np.float32)

    test_min_cpu_results = min_(test_array, gpu=False)
    test_min_gpu_results = min_(test_array, gpu=True)
    np.testing.assert_equal(test_min_cpu_results, test_min_gpu_results)

    del test_min_cpu_results, test_min_gpu_results
