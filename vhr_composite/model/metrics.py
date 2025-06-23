import logging
import time
import os

from typing import Tuple

import numpy as np
import numba as nb
import xarray as xr


# TODO: replace all of this with the stuff from the config file
CLASS_0: int = 0
CLASS_1: int = 1
CLASS_2: int = 2
CLASS_3: int = 3
CLASS_4: int = 4
CLASS_5: int = 5
CLASS_0_ALIAS: int = 0
CLASS_1_ALIAS: int = 1
CLASS_2_ALIAS: int = 2 # TODO: senegal only
CLASS_3_ALIAS: int = 3
CLASS_4_ALIAS: int = 4
CLASS_5_ALIAS: int = 5
HOST_DTYPE: np.dtype = np.uint8
LAYER_AXIS: int = 2
LAYER_COUNT: int = 3
CLASS_COUNT: int = 3
Y_AXIS: int = 0
X_AXIS: int = 1
SUM_NO_MODE: int = 0
NO_DATA: int = 10

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

from vhr_composite.model import kernels

# --------------------------------------------------------------------------
# SKELETON FUNCTION PT. 5
# Change function name to fit alg
# Change "alg" out with whatever you're doin
# --------------------------------------------------------------------------
@nb.njit
def _alg_product(input_array: np.ndarray) -> int:
    """
    reduction algorithm, takes 3d input, returns scalar
    """
    output_scalar = 0
    return output_scalar


# --------------------------------------------------------------------------
# SKELETON FUNCTION PT. 4
# Change function name to fit alg
# Change "alg" out with whatever you're doing
# --------------------------------------------------------------------------
@nb.njit(parallel=True)
def _fast_iterate_alg(
        output_array: np.ndarray, input_array: np.ndarray) -> np.ndarray:
    """
    Iterate through first two dims of 3d host array
    to get the mode for the z axis.
    """
    for y in nb.prange(input_array.shape[Y_AXIS]):
        for x in nb.prange(input_array.shape[X_AXIS]):
            output_array[y, x] = _alg_product(input_array[y, x, :])
    return output_array


# --------------------------------------------------------------------------
# SKELETON FUNCTION PT. 3
# Change function name to fit alg
# Change "alg" out with whatever you're doing
# --------------------------------------------------------------------------
def calculate_alg(grid_cell_data_array: xr.DataArray,
                  from_disk: bool = False,
                  grid_cell_zarr_path: str = None,
                  logger: logging.Logger = None) -> np.ndarray:
    """
    Reduction algorithm, skeleton function
    """
    if from_disk:
        if not os.path.exists(grid_cell_zarr_path):
            msg = f'{grid_cell_zarr_path} does ' + \
                'cannot be found or does not exist.'
            raise FileNotFoundError(msg)
        grid_cell_data_array = xr.from_zarr(grid_cell_zarr_path)
    grid_cell_shape = grid_cell_data_array.shape
    output_array = np.zeros(
        (grid_cell_shape[Y_AXIS], grid_cell_shape[X_AXIS]), dtype=HOST_DTYPE)
    logger.info('Computing alg')
    st = time.time()
    output_array = _fast_iterate_alg(output_array, grid_cell_data_array)
    et = time.time()
    logger.info('Alg compute time {}'.format(round(et-st, 3)))
    return output_array


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
        return NO_DATA, sum_element_to_return
    max_indices = np.argwhere(array == max_element).flatten()
    max_pl = np.where(max_indices == CLASS_0, CLASS_0_ALIAS, max_indices)
    max_pl = np.where(max_indices == CLASS_1, CLASS_1_ALIAS, max_pl)
    max_pl = np.where(max_indices == CLASS_2, CLASS_2_ALIAS, max_pl)
    max_pl = np.where(max_indices == CLASS_3, CLASS_3_ALIAS, max_pl)
    max_pl = np.where(max_indices == CLASS_4, CLASS_4_ALIAS, max_pl)
    max_pl = np.where(max_indices == CLASS_5, CLASS_5_ALIAS, max_pl)
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
def _fast_iterate_mode_nobs(mode: np.ndarray, nobs: np.ndarray, array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterate through first two dims of 3d host array
    to get the mode for the z axis.
    """
    for y in nb.prange(array.shape[Y_AXIS]):
        for x in nb.prange(array.shape[X_AXIS]):
            mode[y, x], nobs[y, x] = _mode_sum_product(array[y, x, :])
    return mode, nobs


# @nb.njit(parallel=True)
# def _fast_iterate_mode_nobs(mode: np.ndarray, nobs: np.ndarray, array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#    """
#    Iterate through first two dims of 3d host array
#    to get the mode for the z axis.
#    """
#    for y in nb.prange(array.shape[Y_AXIS]):
#        for x in nb.prange(array.shape[X_AXIS]):
#            mode[y, x], nobs[y, x] = _mode_sum_product(array[y, x, :])
#            nobs[y, x] = _get_nobservations(array[y, x, :])
#    return mode, nobs


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

# --------------------------------------------------------------------------
# Confidence Functions
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# calculate_trend_confidence
# --------------------------------------------------------------------------
def calculate_trend_confidence(time_series, window_size=4):
    """
    Calculate confidence based on temporal consistency of classifications.
    Parameters:
    - time_series: Series or array of class codes in temporal order
    - window_size: Size of the rolling window for trend analysis
    Returns:
    - Trend confidence score between 0 and 1
    """
    if len(time_series) < 2:
        return 1.0  # Not enough data points to detect a trend
    # Convert to numpy array if it's not already
    series = np.array(time_series)
    # Basic trend stability (changes across entire series)
    changes = np.sum(np.diff(series) != 0)
    max_changes = len(series) - 1
    if max_changes == 0:
        return 1.0  # Only one observation
    # Calculate trend stability (1 - normalized number of changes)
    overall_stability = 1 - (changes / max_changes)
    # Calculate run lengths
    run_lengths = []
    current_run = 1
    for i in range(1, len(series)):
        if series[i] == series[i-1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    # Add the last run
    run_lengths.append(current_run)
    # Calculate mean run length and normalize by series length
    mean_run_length = np.mean(run_lengths) / len(series) if run_lengths else 1.0
    # Now implement the window_size parameter for local stability analysis
    local_stabilities = []
    if len(series) >= window_size:
        # Calculate stability within rolling windows
        for i in range(len(series) - window_size + 1):
            window = series[i:i+window_size]
            window_changes = np.sum(np.diff(window) != 0)
            max_window_changes = window_size - 1
            window_stability = 1 - (window_changes / max_window_changes)
            local_stabilities.append(window_stability)
        # Average local stability
        local_stability = np.mean(local_stabilities)
    else:
        # If series is shorter than window_size, use overall stability
        local_stability = overall_stability
    # Combine metrics with weights
    # - 50% overall stability (global pattern)
    # - 30% local stability (recent patterns)
    # - 20% mean run length (consistency of runs)
    trend_confidence = (0.5 * overall_stability + 
                        0.3 * local_stability + 
                        0.2 * mean_run_length)
    return trend_confidence

# --------------------------------------------------------------------------
# calculate_trend_confidence_numba
# --------------------------------------------------------------------------
@nb.njit
def calculate_trend_confidence_numba(series, window_size=4):
    n = len(series)
    if n < 2:
        return 1.0

    changes = 0
    for i in range(1, n):
        if series[i] != series[i - 1]:
            changes += 1
    overall_stability = 1.0 - changes / (n - 1)

    run_lengths = []
    current_run = 1
    for i in range(1, n):
        if series[i] == series[i - 1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    mean_run_length = np.mean(np.array(run_lengths)) / n

    if n >= window_size:
        local_stabilities = []
        for i in range(n - window_size + 1):
            window = series[i:i + window_size]
            window_changes = 0
            for j in range(1, window_size):
                if window[j] != window[j - 1]:
                    window_changes += 1
            local_stabilities.append(1.0 - window_changes / (window_size - 1))
        local_stability = np.mean(np.array(local_stabilities))
    else:
        local_stability = overall_stability

    return round(
        0.5 * overall_stability + 0.3 * local_stability + 0.2 * mean_run_length, 2)

# --------------------------------------------------------------------------
# calculate_distribution_confidence
# --------------------------------------------------------------------------
def calculate_distribution_confidence(time_series):
    """
    Calculate a simplified distribution confidence score that considers:
    1. Number of unique classes relative to time series length
    2. Distribution of classes (dominance of top classes)
    3. Length of the time series
    Parameters:
    - time_series: Array-like sequence of land cover classes
    Returns:
    - Confidence score between 0 and 1
    """
    # Handle empty or single-element time series
    if len(time_series) <= 1:
        return 1.0 if len(time_series) == 1 else 0.0
    # Get unique classes and counts
    unique_classes, counts = np.unique(time_series, return_counts=True)
    n_classes = len(unique_classes)
    # If there's only 1 unique class, return perfect confidence
    if n_classes == 1:
        return 1.0
    # Sort counts in descending order
    sorted_counts = np.sort(counts)[::-1]
    # Calculate class diversity score
    # - Fewer classes relative to time series length = higher score
    # - Maximum number of classes is capped at length of time series
    max_possible_classes = min(len(time_series), 10)  # Cap at 10 to avoid excessive penalties for very long series
    class_diversity_score = 1 - ((n_classes - 1) / (max_possible_classes - 1)) if max_possible_classes > 1 else 1.0
    # Calculate dominance score
    # - Higher percentage of observations in top 1-2 classes = higher score
    if n_classes == 2:
        # If only 2 classes, score is automatically high
        dominance_score = 1.0
    else:
        # Calculate percentage of observations in top 2 classes
        top_two_percentage = (sorted_counts[0] + sorted_counts[1]) / len(time_series)
        dominance_score = top_two_percentage
    # Calculate length adjustment
    # - Longer time series get more benefit of the doubt with multiple classes
    # - Short time series should have fewer classes to get high confidence
    length_factor = min(1.0, len(time_series) / 10)  # Saturates at length 10
    # Combine scores
    # Base score from class diversity and dominance
    base_score = 0.7 * class_diversity_score + 0.3 * dominance_score
    # Apply length adjustment: longer series with multiple classes get a boost
    if n_classes > 2:
        adjusted_score = base_score * (0.7 + 0.3 * length_factor)
    else:
        # 2 classes always get high scores, slightly higher for longer series
        adjusted_score = base_score * (0.95 + 0.05 * length_factor)
    return adjusted_score

# --------------------------------------------------------------------------
# calculate_distribution_confidence_numba
# --------------------------------------------------------------------------
@nb.njit
def calculate_distribution_confidence_numba(series):
    n = len(series)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0

    # Step 1: Count class frequencies manually
    # Assume class codes are int or float, round to int for class bins
    class_codes = []
    counts = []

    for i in range(n):
        code = int(series[i])
        found = False
        for j in range(len(class_codes)):
            if class_codes[j] == code:
                counts[j] += 1
                found = True
                break
        if not found:
            class_codes.append(code)
            counts.append(1)

    n_classes = len(class_codes)
    if n_classes == 1:
        return 1.0

    # Step 2: Sort counts descending
    for i in range(len(counts)):
        for j in range(i + 1, len(counts)):
            if counts[j] > counts[i]:
                tmp = counts[i]
                counts[i] = counts[j]
                counts[j] = tmp

    # Step 3: Class diversity score
    max_possible_classes = n if n < 10 else 10
    if max_possible_classes > 1:
        class_diversity_score = 1.0 - ((n_classes - 1) / (max_possible_classes - 1))
    else:
        class_diversity_score = 1.0

    # Step 4: Dominance score
    if n_classes == 2:
        dominance_score = 1.0
    else:
        top1 = counts[0]
        top2 = counts[1] if len(counts) > 1 else 0
        dominance_score = (top1 + top2) / n

    # Step 5: Length adjustment
    length_factor = 1.0 if n >= 10 else n / 10.0

    # Step 6: Combine scores
    base_score = 0.7 * class_diversity_score + 0.3 * dominance_score
    if n_classes > 2:
        adjusted_score = base_score * (0.7 + 0.3 * length_factor)
    else:
        adjusted_score = base_score * (0.95 + 0.05 * length_factor)

    return adjusted_score
