import xarray as xr
import numpy as np

from vhr_composite.model.utils import convert_to_bit_rep


def fill_holes(
        data_array: xr.DataArray,
        reduced_stack: xr.DataArray,
        datetimes_to_fill,
        nodata_value,
        logger):
    reduced_stack_ndarray = reduced_stack.data
    for datetime in datetimes_to_fill:
        if not (reduced_stack_ndarray == nodata_value).any():
            logger.info("No more no-data to fill")
            break
        array_per_datetime = data_array.sel(time=datetime)
        bit_rep_array_per_datetime = convert_to_bit_rep(
            array_per_datetime, nodata_value)
        reduced_stack_ndarray = np.where(reduced_stack_ndarray == nodata_value,
                                         bit_rep_array_per_datetime,
                                         reduced_stack_ndarray)
    reduced_stack.data = reduced_stack_ndarray
    return reduced_stack


if __name__ == '__main__':
    from vhr_composite.model.utils import TqdmLoggingHandler
    from vhr_composite.model.composite import Composite
    import logging
    import pathlib
    import time
    from numba.types import uint32

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    sh = TqdmLoggingHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    def random_assign_nodata(arr, nodata_value):
        # Calculate the number of elements to be assigned as no-data
        num_elements = arr.size
        num_nodata_elements = int(0.2 * num_elements)

        # Generate random indices for no-data elements
        nodata_indices = np.random.choice(
            num_elements, num_nodata_elements, replace=False)

        # Create a copy of the array to avoid modifying the original array
        result_arr = np.copy(arr)

        # Assign the no-data value to the selected indices
        result_arr.flat[nodata_indices] = nodata_value

        return result_arr

    test_shape = (1, 500, 500, 20)
    test_array = np.random.randint(0, 3, size=test_shape)
    logger.info(test_array.max())
    test_array = test_array.astype(np.uint32)

    nodata_value: np.uint32 = np.uint32(10)
    nodata_value = uint32(nodata_value)
    logger.info("{0:32b}".format(nodata_value))
    logger.info(nodata_value)

    test_array = random_assign_nodata(test_array, nodata_value)

    # Dummy time coordinate values
    time_values = np.arange(test_array.shape[3])

    # Create the xarray DataArray
    da = xr.DataArray(test_array,
                      dims=('band', 'y', 'x', 'time'),
                      coords={'band': [1],
                              'y': np.arange(test_array.shape[1]),
                              'x': np.arange(test_array.shape[2]),
                              'time': time_values,
                              'spatial_ref': 'your_spatial_reference_here'})

    output_path = pathlib.Path('dummy_multi_mode.tif')

    good_datetimes = [n for n in range(5)]
    logger.info(good_datetimes)
    bad_datetimes = [n for n in range(5, 20)]
    logger.info(bad_datetimes)

    da_good = da.sel(time=good_datetimes)
    da_good.data[:, :, :, :3] = nodata_value
    da_good.data = random_assign_nodata(da_good.data, nodata_value)
    da_good.data[:, 100, 100, :] = nodata_value
    logger.info(da_good.data[:, 100, 100, :])
    logger.info(np.count_nonzero(da_good.data == nodata_value))

    da_bad = da.sel(time=bad_datetimes)

    st = time.time()
    reduced_stack = Composite.reduce_stack('multi_mode',
                                           da_good,
                                           output_path,
                                           overwrite=False,
                                           nodata=nodata_value,
                                           gpu=True,
                                           logger=logger)

    logger.info(
        f"N nodata: {np.count_nonzero(reduced_stack.data == nodata_value)}")
    logger.info("{0:32b}".format(reduced_stack.data.min()))
    logger.info("{0:32b}".format(reduced_stack.data.max()))
    logger.info(reduced_stack.data.max())
    et = time.time()

    legacy_style = Composite.reduce_stack_legacy_mode('mutli_mode',
                                                      da_good,
                                                      output_path,
                                                      overwrite=False,
                                                      gpu=False,
                                                      logger=logger)
    logger.info(
        f"N nodata: {np.count_nonzero(legacy_style.data == nodata_value)}")
    logger.info("{0:32b}".format(legacy_style.data.min()))
    logger.info("{0:32b}".format(legacy_style.data.max()))
    logger.info(legacy_style.data.max())

    logger.info('Pre-QA comparison')
    np.testing.assert_equal(reduced_stack.data, legacy_style.data)
    logger.info('Comparison passed')

    legacy_style = Composite.reduce_stack_legacy_mode_w_qa('mutli_mode',
                                                           da_good,
                                                           da_bad,
                                                           bad_datetimes,
                                                           output_path,
                                                           overwrite=False,
                                                           gpu=False,
                                                           logger=logger)
    logger.info(
        f"N nodata: {np.count_nonzero(legacy_style.data == nodata_value)}")
    logger.info("{0:32b}".format(legacy_style.data.min()))
    logger.info("{0:32b}".format(legacy_style.data.max()))
    logger.info(legacy_style.data.max())

    reduced_stack_hole_filled = fill_holes(
        da_bad, reduced_stack, bad_datetimes, nodata_value, logger)
    count_check = np.count_nonzero(
        reduced_stack_hole_filled.data == nodata_value)
    logger.info(
        f"N nodata: {count_check}")
    logger.info("{0:32b}".format(reduced_stack_hole_filled.data.min()))
    logger.info("{0:32b}".format(reduced_stack_hole_filled.data.max()))
    logger.info(reduced_stack_hole_filled.data.max())

    np.testing.assert_equal(reduced_stack_hole_filled.data, legacy_style.data)
