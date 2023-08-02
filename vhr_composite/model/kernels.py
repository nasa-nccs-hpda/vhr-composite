import numpy as np
import numba as nb

from numba import cuda
from numba.types import uint32


HOST_DTYPE: np.dtype = np.uint8
HOST_FILL: np.uint32 = np.uint32((2**32)-1)

Y_AXIS: int = 0
X_AXIS: int = 1
Z_AXIS: int = 2


# CPU Algs
@nb.njit(parallel=True)
def calculate_mode_cpu_kernel(arr, mode_values, nodata=HOST_FILL):
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

    for y in nb.prange(arr.shape[Y_AXIS]):
        for x in nb.prange(arr.shape[X_AXIS]):

            # Setup histogram
            classes = np.zeros(33, dtype=np.uint32)
            for k in range(classes.shape[0]):
                classes[k] = np.uint64(k)

            histogram = np.zeros(33, dtype=np.uint32)
            for z in nb.prange(arr.shape[Z_AXIS]):
                value = arr[y, x, z]
                if value != nodata:
                    histogram[value] += 1

            mode = np.uint32(0)
            nodata_mode = np.uint32(0)
            max_count = 0

            for k in range(33):
                if histogram[k] > max_count:
                    max_count = histogram[k]
                    mode = classes[k]

            # Store the mode value in the mode_values array
            mode_values[y, x] = mode
            if (mode == nodata_mode) and (max_count == 0):
                mode_values[y, x] = nodata

    return mode_values


# Define the GPU kernel using Numba
@cuda.jit
def calculate_multi_mode_gpu_kernel(arr, mode_values, nodata=HOST_FILL):
    """
    GPU kernel to calculate the mode along the Z dimension of a
    3D array on the GPU using Numba with a histogram-based method.

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

    # Get the dimensions of the input array
    N, Z = arr.shape[0], arr.shape[2]

    # Get the indices of the current thread
    i, j = nb.cuda.grid(2)

    nodata = uint32(nodata)

    # Check if the thread is within the valid range of indices
    if i < N and j < N:
        classes = nb.cuda.local.array(31, dtype=nb.uint32)
        for k in range(classes.shape[0]):
            classes[k] = uint32(2**k)
        # Initialize an array to store the histogram
        histogram = nb.cuda.local.array(31, dtype=nb.uint32)

        # Initialize the histogram with zeros
        for k in range(31):
            histogram[k] = 0

        # Count the occurrences of each value in the slice using a histogram
        for z in range(Z):
            value = arr[i, j, z]
            if value != nodata:
                histogram[value] += 1

        # Find the mode by finding the value with the maximum count
        mode = uint32(0)
        nodata_mode = uint32(0)
        max_count = 0
        for k in range(31):
            if histogram[k] > max_count:
                max_count = histogram[k]
                mode = classes[k]
            elif (histogram[k] == max_count) and (max_count > 0):
                mode = mode | classes[k]
        if mode == nodata_mode:
            mode = nodata

        mode_values[i, j] = mode


@nb.njit(parallel=True)
def calculate_multi_mode_cpu_kernel(arr, mode_values, nodata=HOST_FILL):
    """
    GPU kernel to calculate the mode along the Z dimension
    of a 3D array on the GPU using Numba with a histogram-based method.

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

    for y in nb.prange(arr.shape[Y_AXIS]):
        for x in nb.prange(arr.shape[X_AXIS]):

            # Setup histogram
            classes = np.zeros(31, dtype=np.uint32)
            for k in range(classes.shape[0]):
                classes[k] = np.uint32(2**k)

            histogram = np.zeros(31, dtype=np.uint32)
            for k in range(31):
                histogram[k] = 0
            for z in range(arr.shape[Z_AXIS]):
                value = arr[y, x, z]
                if value != nodata:
                    histogram[value] += 1

            mode = np.uint32(0)
            nodata_mode = np.uint32(0)
            max_count = 0

            for k in range(31):
                if histogram[k] > max_count:
                    max_count = histogram[k]
                    mode = classes[k]
                elif histogram[k] == max_count:
                    mode = mode | classes[k]

            # Store the mode value in the mode_values array
            mode_values[y, x] = mode
            if (mode == nodata_mode) and (max_count == 0):
                mode_values[y, x] = nodata

    return mode_values


@nb.njit()
def calculate_sum_across_axis(arr, nobs_values, nodata=HOST_FILL):
    """
    CPU kernel to calculate the sum across an axis
    using Numpy.

    Parameters:
    -----------
    arr: 3D numpy array
        Input array of shape (N, N, Z).
    nobs_values: 2D numpy array
        Output array to store the sum across the axis
        values along the Z dimension, of shape (N, N).

    Returns:
    --------
    None
    """
    nobs_values = np.count_nonzero((arr < 32) & (arr != nodata), axis=Z_AXIS)
    return nobs_values


def calculate_max_cpu_kernel(arr, max_values):
    """
    CPU kernel to calculate the max value along the Z dimension of
    a 3D array on the CPU using Numpy.

    Parameters:
    -----------
    arr: 3D numpy array
        Input array of shape (N, N, Z).
    max_values: 2D numpy array
        Output array to store the max values along the
        Z dimension, of shape (N, N).

    Returns:
    --------
    None
    """
    max_values = np.max(arr, axis=Z_AXIS)
    return max_values


# Define the GPU kernel using Numba
@cuda.jit
def calculate_max_gpu_kernel(arr, max_values):
    """
    GPU kernel to calculate the max along the Z dimension
    of a 3D array on the GPU using Numba with a histogram-based
    method.

    Parameters:
    -----------
    arr: 3D numpy array
        Input array of shape (N, N, Z).
    max_values: 2D numpy array
        Output array to store the max values
        along the Z dimension, of shape (N, N).

    Returns:
    --------
    None
    """

    # Get the dimensions of the input array
    N, Z = arr.shape[0], arr.shape[2]

    # Get the indices of the current thread
    i, j = nb.cuda.grid(2)

    # Check if the thread is within the valid range of indices
    if i < N and j < N:

        max_value = 0
        for z in range(Z):
            value_to_compare = arr[i, j, z]
            if value_to_compare > max_value:
                max_value = value_to_compare

        max_values[i, j] = max_value


def calculate_min_cpu_kernel(arr, min_values):
    """
    CPU kernel to calculate the min value along the Z dimension of
    a 3D array on the CPU using Numpy.

    Parameters:
    -----------
    arr: 3D numpy array
        Input array of shape (N, N, Z).
    min_values: 2D numpy array
        Output array to store the mode values
        along the Z dimension, of shape (N, N).

    Returns:
    --------
    None
    """
    min_values = np.min(arr, axis=Z_AXIS)
    return min_values


# Define the GPU kernel using Numba
@cuda.jit
def calculate_min_gpu_kernel(arr, min_values):
    """
    GPU kernel to calculate the min value along the Z dimension of
    a 3D array on the GPU using Numba with a histogram-based method.

    Parameters:
    -----------
    arr: 3D numpy array
        Input array of shape (N, N, Z).
    min_values: 2D numpy array
        Output array to store the mode values along the
        Z dimension, of shape (N, N).

    Returns:
    --------
    None
    """

    # Get the dimensions of the input array
    N, Z = arr.shape[0], arr.shape[2]

    # Get the indices of the current thread
    i, j = nb.cuda.grid(2)

    # Check if the thread is within the valid range of indices
    if i < N and j < N:

        min_value = np.inf
        for z in range(Z):
            value_to_compare = arr[i, j, z]
            if value_to_compare < min_value:
                min_value = value_to_compare

        min_values[i, j] = min_value


def calculate_mean_cpu_kernel(arr, mean_values):
    """
    CPU kernel to calculate the mean value along the Z dimension of
    a 3D array on the CPU using Numpy.

    Parameters:
    -----------
    arr: 3D numpy array
        Input array of shape (N, N, Z).
    mean_values: 2D numpy array
        Output array to store the mean values along
        the Z dimension, of shape (N, N).

    Returns:
    --------
    None
    """
    mean_values = np.mean(arr, axis=Z_AXIS)
    return mean_values


# Define the GPU kernel using Numba
@cuda.jit
def calculate_mean_gpu_kernel(arr, mean_values):
    """
    GPU kernel to calculate the mean along the Z dimension
    of a 3D array on the GPU using Numba.

    Parameters:
    -----------
    arr: 3D numpy array
        Input array of shape (N, N, Z).
    mean_values: 2D numpy array
        Output array to store the mean values
        along the Z dimension, of shape (N, N).

    Returns:
    --------
    None
    """

    # Get the dimensions of the input array
    N, Z = arr.shape[0], arr.shape[2]

    # Get the indices of the current thread
    i, j = nb.cuda.grid(2)

    # Check if the thread is within the valid range of indices
    if i < N and j < N:

        sum_value = 0
        for z in range(Z):
            sum_value = sum_value + arr[i, j, z]

        mean_values[i, j] = sum_value / Z


def calculate_std_cpu_kernel(arr, std_values):
    """
    CPU kernel to calculate the std value along the Z dimension of
    a 3D array on the CPU using Numpy.

    Parameters:
    -----------
    arr: 3D numpy array
        Input array of shape (N, N, Z).
    min_values: 2D numpy array
        Output array to store the std values
        along the Z dimension, of shape (N, N).

    Returns:
    --------
    None
    """
    std_values = np.std(arr, axis=Z_AXIS)
    return std_values


# Define the GPU kernel using Numba
@cuda.jit
def calculate_std_gpu_kernel(arr, std_values):
    """
    GPU kernel to calculate the std along the Z dimension
    of a 3D array on the GPU.

    Parameters:
    -----------
    arr: 3D numpy array
        Input array of shape (N, N, Z).
    mode_values: 2D numpy array
        Output array to store the std values along the
        Z dimension, of shape (N, N).

    Returns:
    --------
    None
    """

    # Get the dimensions of the input array
    N, Z = arr.shape[0], arr.shape[2]

    # Get the indices of the current thread
    i, j = nb.cuda.grid(2)

    # Check if the thread is within the valid range of indices
    if i < N and j < N:

        sum_value = 0
        for z in range(Z):
            sum_value = sum_value + arr[i, j, z]

        mean_value = sum_value / Z
        summ_diff_mean = 0
        for z in range(Z):
            summ_diff_mean = summ_diff_mean + abs(arr[i, j, z] - mean_value)**2

        inter_std = summ_diff_mean / Z
        std_values[i, j] = inter_std ** 0.5


if __name__ == '__main__':

    from scipy import stats

    # Setting up testing arrays
    test_classes = {0: 0, 1: 1, 2: 2}
    test_shape = (200, 200, 20)
    test_array = np.random.randint(0, 33, size=test_shape)
    test_array = test_array.astype(np.uint8)
    test_mode_values = np.empty(test_shape[:2], dtype=np.uint32)

    test_mode_control = stats.mode(test_array, axis=2)[0][:, :, 0]
    test_mode_results = calculate_mode_cpu_kernel(
        test_array, test_mode_values, )
    np.testing.assert_equal(test_mode_control, test_mode_results)

    test_mode_values = np.empty(test_shape[:2], dtype=np.uint32)
    test_mode_results = calculate_multi_mode_cpu_kernel(test_array,
                                                        test_mode_values)
    # Testing fast iterate mode JIT

    # ---
    # GPU - Multi-mode
    # ---
    # Create an empty array to store the mode values
    mode_values = np.empty((200, 200), dtype=np.uint8)
    # Copy the mode_values array to the GPU
    mode_values_gpu = nb.cuda.to_device(mode_values)

    arr_gpu = nb.cuda.to_device(test_array)

    threadsperblock = (16, 16)
    blockspergrid_x = (test_shape[1] + threadsperblock[0] - 1) \
        // threadsperblock[0]
    blockspergrid_y = (test_shape[0] + threadsperblock[1] - 1) \
        // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    calculate_multi_mode_gpu_kernel[blockspergrid, threadsperblock](
        arr_gpu, mode_values_gpu, HOST_FILL)
    mode_values = mode_values_gpu.copy_to_host()

    del test_mode_values, test_mode_results, test_mode_control
    del arr_gpu, mode_values_gpu

    # ---
    # GPU - Max
    # ---

    test_float_array = np.random.rand(200, 200, 20)
    test_float_array_gpu = nb.cuda.to_device(test_float_array)

    # Create an empty array to store the mode values
    max_values = np.empty((200, 200), dtype=np.float64)
    # Copy the mode_values array to the GPU
    max_values_gpu = nb.cuda.to_device(max_values)

    threadsperblock = (16, 16)
    blockspergrid_x = (test_shape[1] + threadsperblock[0] - 1) \
        // threadsperblock[0]
    blockspergrid_y = (test_shape[0] + threadsperblock[1] - 1) \
        // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    calculate_max_gpu_kernel[blockspergrid, threadsperblock](
        test_float_array_gpu, max_values_gpu)
    max_values_out = max_values_gpu.copy_to_host()
    max_values_np = calculate_max_cpu_kernel(test_float_array, max_values)
    np.testing.assert_equal(max_values_np, max_values_out)

    del max_values, max_values_gpu, test_float_array, test_float_array_gpu
    del max_values_out, max_values_np

    # ---
    # GPU - Min
    # ---

    test_float_array = np.random.rand(200, 200, 20)
    test_float_array_gpu = nb.cuda.to_device(test_float_array)

    # Create an empty array to store the mode values
    min_values = np.empty((200, 200), dtype=np.float64)
    # Copy the mode_values array to the GPU
    min_values_gpu = nb.cuda.to_device(min_values)

    threadsperblock = (16, 16)
    blockspergrid_x = (test_shape[1] + threadsperblock[0] - 1) \
        // threadsperblock[0]
    blockspergrid_y = (test_shape[0] + threadsperblock[1] - 1) \
        // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    calculate_min_gpu_kernel[blockspergrid, threadsperblock](
        test_float_array_gpu, min_values_gpu)
    min_values_out = min_values_gpu.copy_to_host()
    min_values_np = calculate_min_cpu_kernel(test_float_array, min_values)
    np.testing.assert_equal(min_values_np, min_values_out)

    del min_values, min_values_gpu, test_float_array, test_float_array_gpu
    del min_values_out, min_values_np

    # ---
    # GPU - Min
    # ---
    test_float_array = np.random.rand(200, 200, 20)
    test_float_array_gpu = nb.cuda.to_device(test_float_array)

    # Create an empty array to store the mode values
    mean_values = np.empty((200, 200), dtype=np.float64)
    # Copy the mode_values array to the GPU
    mean_values_gpu = nb.cuda.to_device(mean_values)

    threadsperblock = (16, 16)
    blockspergrid_x = (test_shape[1] + threadsperblock[0] - 1) \
        // threadsperblock[0]
    blockspergrid_y = (test_shape[0] + threadsperblock[1] - 1) \
        // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    calculate_mean_gpu_kernel[blockspergrid, threadsperblock](
        test_float_array_gpu, mean_values_gpu)
    mean_values_out = mean_values_gpu.copy_to_host()
    mean_values_np = calculate_mean_cpu_kernel(test_float_array, mean_values)
    np.testing.assert_almost_equal(mean_values_np, mean_values_out)

    del mean_values, mean_values_gpu, test_float_array, test_float_array_gpu
    del mean_values_out, mean_values_np

    # ---
    # GPU - Min
    # ---
    test_float_array = np.random.rand(200, 200, 20)
    test_float_array_gpu = nb.cuda.to_device(test_float_array)

    # Create an empty array to store the mode values
    std_values = np.empty((200, 200), dtype=np.float64)
    # Copy the mode_values array to the GPU
    std_values_gpu = nb.cuda.to_device(std_values)

    threadsperblock = (16, 16)
    blockspergrid_x = (test_shape[1] + threadsperblock[0] - 1) \
        // threadsperblock[0]
    blockspergrid_y = (test_shape[0] + threadsperblock[1] - 1) \
        // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    calculate_std_gpu_kernel[blockspergrid, threadsperblock](
        test_float_array_gpu, std_values_gpu)
    std_values_out = std_values_gpu.copy_to_host()
    std_values_np = calculate_std_cpu_kernel(test_float_array, std_values)
    np.testing.assert_almost_equal(std_values_np, std_values_out, decimal=0)

    del std_values, std_values_gpu, test_float_array, test_float_array_gpu
    del std_values_out, std_values_np
