import dask
import numba
import numpy as np
import xarray as xr
from pathlib import Path
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

dask.config.set(scheduler='processes')  # or use distributed for more control


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

@numba.njit
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


def run_accelerated_function(data_array, function, input_core_dims: str = 'time'):

    # Apply over (time) dimension for each (y, x)
    result = xr.apply_ufunc(
        function,
        data_array,
        input_core_dims=[[input_core_dims]],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.float32]
    )

    # Result is a 2D DataArray with shape (y, x)
    # Trigger computation and monitor progress
    with ProgressBar():
        result.compute()

    # do it in parallel for speed for more files
    return result.expand_dims(dim='band', axis=0)


@numba.njit
def calculate_distribution_confidence_numba_old(time_series):
    n = len(time_series)
    if n == 0:
        return 0.0
    elif n == 1:
        return 1.0

    # Manual histogram count assuming 0–255 class values
    unique = np.zeros(256, dtype=np.int32)
    for val in time_series:
        idx = int(val)
        if 0 <= idx < 256:
            unique[idx] += 1

    # Extract counts and number of classes
    counts = []
    n_classes = 0
    for i in range(256):
        if unique[i] > 0:
            counts.append(unique[i])
            n_classes += 1

    if n_classes == 1:
        return 1.0

    # Sort counts descending
    sorted_counts = np.array(counts)
    sorted_counts[::-1].sort()

    max_possible_classes = min(n, 10)
    if max_possible_classes > 1:
        class_diversity_score = 1 - ((n_classes - 1) / (max_possible_classes - 1))
    else:
        class_diversity_score = 1.0

    if n_classes == 2:
        dominance_score = 1.0
    else:
        top_two_sum = sorted_counts[0] + sorted_counts[1] if len(sorted_counts) > 1 else sorted_counts[0]
        dominance_score = top_two_sum / n

    length_factor = n / 10.0 if n < 10 else 1.0

    base_score = 0.7 * class_diversity_score + 0.3 * dominance_score

    if n_classes > 2:
        adjusted_score = base_score * (0.7 + 0.3 * length_factor)
    else:
        adjusted_score = base_score * (0.95 + 0.05 * length_factor)

    return adjusted_score

from numba import njit
import numpy as np

@numba.njit
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


def main():

    # TODO: choose files with glob

    # iterate over each filename
    # open zarr file, this is just an example for now
    filename = '/explore/nobackup/projects/3sl/development/cnn_landcover_composite/ethiopia-v11/Amhara.M1BS.h26v42.zarr'
    lc_ds = xr.open_zarr(filename)

    # Remove band since its fixed (size=1)
    lc_ds = lc_ds.squeeze(dim='band', drop=True)
    lc_ds = lc_ds.chunk({'time': -1, 'y': 500, 'x': 500})
    lc_array = lc_ds[Path(filename).stem]

    confidence_trend = run_accelerated_function(
        lc_array, calculate_trend_confidence_numba)
    confidence_distribution = run_accelerated_function(
        lc_array, calculate_distribution_confidence_numba)

    confidence_trend.rio.to_raster(
        f'{Path(filename).stem}_confidence_trend.tif')
    confidence_distribution.rio.to_raster(
        f'{Path(filename).stem}_confidence_distribution_numba_cons.tif')
    return


if __name__ == "__main__":
    main()