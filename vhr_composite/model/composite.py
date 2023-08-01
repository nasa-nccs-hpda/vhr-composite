import os
import logging
import tqdm
import rasterio
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import pandas as pd
import numpy as np

from osgeo import gdal

from typing import Union

from vhr_composite.model.utils import convert_to_bit_rep

from vhr_composite.model import metrics
from vhr_composite.model.metrics import calculate_mode
from vhr_composite.model.metrics import CLASS_0, CLASS_1, CLASS_2
from vhr_composite.model.metrics import CLASS_0_ALIAS
from vhr_composite.model.metrics import CLASS_1_ALIAS, CLASS_2_ALIAS

TIME: str = "time"
X: str = "x"
Y: str = "y"
BAND: int = 1
CLOUD_CLEAR_VALUE: int = 0
NO_DATA: int = 255
OTHER_DATA: int = 10
FILL_DATA: int = 10
BURN_AREA_VALUE: int = 3
STRIP_FILE_PATH_KEY: str = 'landcover'
MASK_FILE_PATH_KEY: str = 'cloudmask'
GRID_CELL_NAME_PRE_STR: str = 'CAS.M1BS'
DIM: str = "time"


class Composite(object):

    def __init__(self,
                 name: str,
                 grid_geopackage_path: str,
                 model_output_geopackage_path: str,
                 output_dir: str,
                 logger: logging.Logger) -> None:
        """
        Initializes the Composite object
        """
        self._name = name
        self._grid_geopackage_path = grid_geopackage_path
        self._model_output_geopackage_path = model_output_geopackage_path
        self._logger = logger

        os.makedirs(output_dir, exist_ok=True)
        print(output_dir)
        self._output_dir = output_dir

        if not os.path.exists(self._grid_geopackage_path) or \
                not os.path.exists(self._model_output_geopackage_path):
            msg = '{} does not exist'.format(self._grid_geopackage_path)
            raise FileNotFoundError(msg)

        self._grid = gpd.read_file(self._grid_geopackage_path)
        self._model_output = \
            gpd.read_file(self._model_output_geopackage_path)

    def generate_grid_via_intersect(self, calculate_mode: bool = False,
                                    classes: dict = None) -> None:
        """
        Generate the gridded zarrs from the strips that intersect
        each grid cell
        """
        self._logger.info('Calculating intersect')
        intersection = self._get_intersection()
        for tile in tqdm.tqdm(intersection.keys()):
            tile_df = self._grid[self._grid['tile'] == tile]
            tiles_df = [tile_df for _ in range(len(intersection[tile]))]
            strips = intersection[tile]
            landcovers = strips['landcover']
            cloudmasks = strips['cloudmask']
            datetimes = strips['datetime']
            arrays = list(map(Composite.strip_to_grid,
                              landcovers, cloudmasks,
                              datetimes, tiles_df))
            arrays = [array for array in arrays if array is not None]
            name = 'CAS.M1BS.{}'.format(tile)
            concat_array = xr.concat(arrays, dim='time', fill_value=10)
            concat_dataset = concat_array.to_dataset(name=name)
            tile_path = os.path.join(self._output_dir, f'{name}.zarr')
            concat_dataset.to_zarr(tile_path)
            # Take this part out, too much going on
            if calculate_mode and classes:
                self.calculate_mode_per_tile(tile_path=tile_path,
                                             classes=classes,
                                             tile_dataset_input=concat_dataset,
                                             )
        return None

    def generate_grid(self, tile_list: list) -> None:
        """
        Generate the gridded zarrs from a pre-calculated intersection
        of grid cells and
        """
        for tile in tqdm.tqdm(tile_list):
            self._logger.info(f'{tile}- Processing {tile}')
            name = '{}.{}'.format(GRID_CELL_NAME_PRE_STR, tile)
            tile_path = os.path.join(self._output_dir, f'{name}.zarr')
            if os.path.exists(tile_path):
                self._logger.info(f'{tile}- {tile_path} already exists')
                continue
            tile_df = self._grid[self._grid['tile'] == tile]
            strips = self._model_output[
                self._model_output['tile'] == tile]
            len_strips = len(strips)
            self._logger.info(f'{tile}- Processing {len_strips} strips')
            tiles_df = [tile_df for _ in range(len_strips)]
            landcovers = strips['landcover'].values
            cloudmasks = strips['cloudmask'].values
            datetimes = strips['datetime'].values
            if len_strips < 1:
                continue
            arrays = list(map(
                Composite.strip_to_grid,
                landcovers, cloudmasks,
                datetimes, tiles_df))
            arrays = [array for array in arrays if array is not None]

            concat_array = xr.concat(arrays, dim='time', fill_value=10)
            concat_dataset = concat_array.to_dataset(name=name)

            self._logger.info(f'{tile}-Writing to {tile_path}')
            concat_dataset.to_zarr(tile_path)
            arrays = None
            concat_array = None
            concat_dataset = None
        return None

    def generate_single_grid(
            self,
            tile: str,
            variable_name: str,
            write_out: bool = False) -> Union[xr.Dataset, None]:
        """
        Generate the gridded zarrs from a pre-calculated intersection
        of grid cells and
        """
        self._logger.info(f'{tile}- Processing {tile}')
        name = variable_name
        tile_path = os.path.join(self._output_dir, f'{name}.zarr')
        if os.path.exists(tile_path):
            self._logger.info(f'{tile}- {tile_path} already exists')
            return xr.open_zarr(tile_path)
        tile_df = self._grid[self._grid['tile'] == tile]
        strips = self._model_output[
            self._model_output['tile'] == tile]
        len_strips = len(strips)
        self._logger.info(f'{tile}- Processing {len_strips} strips')
        tiles_df = [tile_df for _ in range(len_strips)]
        landcovers = strips['landcover'].values
        cloudmasks = strips['cloudmask'].values
        datetimes = strips['datetime'].values
        if len_strips < 1:
            return None
        arrays = list(map(
            Composite.strip_to_grid,
            landcovers, cloudmasks,
            datetimes, tiles_df))
        arrays = [array for array in arrays if array is not None]

        if len(arrays) < 1:
            return None

        concat_array = xr.concat(arrays, dim='time', fill_value=10)
        concat_dataset = concat_array.to_dataset(name=name)
        if write_out:
            self._logger.info(f'{tile}-Writing to {tile_path}')
            concat_dataset.to_zarr(tile_path)
            return concat_dataset
        arrays = None
        concat_array = None
        return concat_dataset

    def _get_intersection(self) -> dict:
        """
        For each grid cell, find the strips that intersect
        """
        intersection_dict = {}
        for i in tqdm.tqdm(range(len(self._grid))):
            intersects = self._model_output['geometry'].apply(
                lambda shp: shp.intersects(self._grid.iloc[i]['geometry']))
            does_intersect = intersects[intersects]
            if len(does_intersect) > 0:
                intersection_dict[self._grid.iloc[i]['tile']] = \
                    self._model_output.loc[list(does_intersect.index)]
        self._logger.info(f'Found {len(intersection_dict)} intersections')
        return intersection_dict

    @staticmethod
    def strip_to_grid(
            land_cover_path: str,
            cloud_mask_path: str,
            timestamp: pd._libs.tslibs.timestamps.Timestamp,
            grid_geodataframe: gpd.GeoDataFrame) -> Union[xr.DataArray, None]:
        """
        This function opens a strip and clips it to a given geometry
        and performs some basic quality assurance checking.

        :param strip_path: str, the file path for the strip to read and clip
        :param cloud_mask_path: str, the file path for the corresponding
        cloud mask
        :param timestamp: the time-stamp for the
        :param grid_geodataframe: geopandas.DataFrame, the dataframe whose
        geometry is used to clip the strip to
        :return: rioxarray.DataArray, the clipped DataArray strip
        """
        try:
            strip_data_array = rxr.open_rasterio(land_cover_path)
            cloud_mask_data_array = rxr.open_rasterio(cloud_mask_path)
        except rasterio.errors.RasterioIOError:
            error_file = os.path.basename(
                land_cover_path).replace('.tif', '.txt')
            with open(error_file, 'w') as fh:
                fh.writelines([land_cover_path+'\n'+cloud_mask_path+'\n'])
            return None

        geometry_to_clip = grid_geodataframe.geometry.values
        geometry_crs = grid_geodataframe.crs
        strip_data_array = strip_data_array.rio.clip(geometry_to_clip,
                                                     crs=geometry_crs)

        strip_data_array = strip_data_array.assign_coords(time=timestamp)
        strip_data_array = strip_data_array.expand_dims(dim=TIME)
        strip_data_array = strip_data_array.where((
            (cloud_mask_data_array == 0) &
            (strip_data_array != BURN_AREA_VALUE) &
            (strip_data_array != NO_DATA)
        ), other=OTHER_DATA).astype(np.uint8)

        return strip_data_array

    def calculate_mode_per_tile(self,
                                tile_path: str,
                                classes: dict,
                                tile_dataset_input: xr.Dataset = None) -> None:
        """
        Given a landcover zarr or dataset, calculate the mode
        and write to GTiff
        """
        tile_dataset = tile_dataset_input if tile_dataset_input \
            else xr.open_zarr(tile_path)
        variable_name = os.path.basename(tile_path).split('.zarr')[0]
        # Select the array without the band, transpose to time-last format
        tile_array = tile_dataset[variable_name].sel(
            band=BAND).transpose(Y, X, TIME)
        mode = self._calculate_mode(tile_array, classes, logger=self._logger)
        self._logger.info('Compute mode - Configuring mode to data array')
        # Add the band to the mode
        mode_with_band = np.zeros((BAND, mode.shape[0], mode.shape[1]))
        mode_with_band[0, :, :] = mode

        # Make the coordinates that will be used to make the mode ndarray
        # a xr.DataArray
        coords = dict(
            band=tile_dataset.band,
            y=tile_dataset.y,
            x=tile_dataset.x,
            spatial_ref=tile_dataset.spatial_ref,
        )
        name = '{}.mode.{}'.format(variable_name, self._experiment_name)
        self._logger.info(f'Compute mode - Appending {name} to {tile_path}')
        mode_data_array = self._make_data_array(mode_with_band,
                                                coords,
                                                name)

        # Write to GTiff
        tile_raster_path = tile_path.replace('.zarr', f'{name}.tif')
        mode_data_array.rio.to_raster(tile_raster_path)
        return None

    @staticmethod
    def _add_non_qa_data(tile_datarray: xr.DataArray,
                         mode: np.ndarray,
                         not_passed_qa_datetimes: list,
                         logger: logging.Logger) -> np.ndarray:
        for qa_datetime_to_select in not_passed_qa_datetimes:
            if not (mode == OTHER_DATA).any():
                logger.info('No more nodata, exiting')
                break
            model_output_per_datetime = tile_datarray.sel(
                time=qa_datetime_to_select)
            model_output_per_datetime_ndarray = \
                model_output_per_datetime.values
            model_output_per_datetime_ndarray_ed = np.where(
                model_output_per_datetime_ndarray == CLASS_0,
                CLASS_0_ALIAS,
                model_output_per_datetime_ndarray)
            model_output_per_datetime_ndarray_ed = np.where(
                model_output_per_datetime_ndarray == CLASS_1,
                CLASS_1_ALIAS,
                model_output_per_datetime_ndarray_ed)
            model_output_per_datetime_ndarray_ed = np.where(
                model_output_per_datetime_ndarray == CLASS_2,
                CLASS_2_ALIAS,
                model_output_per_datetime_ndarray_ed)
            mode = np.where(mode == OTHER_DATA,
                            model_output_per_datetime_ndarray_ed, mode)
        return mode

    def calculate_mode(self,
                       tile_path: str,
                       tile_raster_output_path: str,
                       classes: dict,
                       rows_to_use: list,
                       output_nobservations: bool = True,
                       tile_dataset_input: xr.Dataset = None) -> None:
        """
        Given a landcover zarr or dataset, calculate the mode
        and write to GTiff
        """
        tile_dataset = tile_dataset_input if tile_dataset_input \
            else xr.open_zarr(tile_path)

        variable_name = os.path.basename(tile_path).split('.zarr')[0]
        variable_name = variable_name.replace('ETZ', 'CAS')

        name = f'{variable_name}.mode'

        if os.path.exists(tile_raster_output_path):
            self._logger.info(f'{tile_raster_output_path} already exists.')
            return None

        try:
            # Select the array without the band, transpose to time-last format
            tile_array = \
                tile_dataset[variable_name].sel(time=rows_to_use)
        except KeyError:
            self._logger.error(
                f'Could not find all times in passed {rows_to_use}')
            return None

        self._logger.info(tile_array.time)
        tile_array = tile_array.sel(
            band=BAND).transpose(Y, X, TIME)

        # Make the coordinates that will be used to make the mode ndarray
        # a xr.DataArray
        coords = dict(
            band=tile_dataset.band,
            y=tile_dataset.y,
            x=tile_dataset.x,
            spatial_ref=tile_dataset.spatial_ref,
        )

        if output_nobservations:
            self._logger.info('Calculating n-observations in addition to mode')
            mode, nobs = self._calculate_mode(
                tile_array,
                classes,
                output_nobservations,
                logger=self._logger)
            nobs_with_band = np.zeros((BAND, mode.shape[0], mode.shape[1]))
            nobs_with_band[0, :, :] = nobs
            nobs_data_array = self._make_data_array(nobs_with_band,
                                                    coords,
                                                    name)
            variable_name = variable_name.replace('CAS', 'ETZ')
            nobs_name = f'{variable_name}.nobservations.noqa'
            nobs_raster_output_path = \
                tile_raster_output_path.replace('mode', 'nobservations')
            self._logger.info(
                f'Compute mode - Writing {nobs_name} ' +
                f'to {nobs_raster_output_path}')
            nobs_data_array.rio.to_raster(nobs_raster_output_path,
                                          dtype=np.uint16,
                                          compress='lzw')
        else:
            mode = self._calculate_mode(
                tile_array, classes, logger=self._logger)

        # Add the band to the mode
        mode_with_band = np.zeros((BAND, mode.shape[0], mode.shape[1]))
        mode_with_band[0, :, :] = mode

        mode_data_array = self._make_data_array(mode_with_band,
                                                coords,
                                                name)

        # Write to GTiff
        self._logger.info(
            f'Compute mode - Writing {name} to {tile_raster_output_path}')
        mode_data_array.rio.to_raster(tile_raster_output_path,
                                      dtype=np.uint8,
                                      compress='lzw')
        warpOptions = ['COMPRESS=LZW']
        warped_tile = tile_raster_output_path.replace('.tif', 'warp.tif')
        _ = gdal.Warp(warped_tile, tile_raster_output_path,
                      warpOptions=warpOptions)
        return None

    def calculate_mode_qa(self,
                          tile_path: str,
                          tile_raster_output_path: str,
                          classes: dict,
                          passed_qa_datetimes: list,
                          not_passed_qa_datetimes: list,
                          output_nobservations: bool = True,
                          tile_dataset_input: xr.Dataset = None) -> None:
        """
        Given a landcover zarr or dataset, calculate the mode
        and write to GTiff
        """
        tile_dataset = tile_dataset_input if tile_dataset_input \
            else xr.open_zarr(tile_path)

        variable_name = os.path.basename(tile_path).split('.zarr')[0]
        variable_name = variable_name.replace('ETZ', 'CAS')

        self._logger.info(variable_name)

        name = f'{variable_name}.mode'

        if os.path.exists(tile_raster_output_path):
            self._logger.info(f'{tile_raster_output_path} already exists.')
            return None

        try:
            # Select the array without the band, transpose to time-last format
            tile_array = \
                tile_dataset[variable_name].sel(time=passed_qa_datetimes)
        except KeyError as e:
            print(e)
            self._logger.error(
                'Good - Could not find all times' +
                f' in passed {passed_qa_datetimes}')
            return None

        self._logger.info(tile_array.time)
        try:
            bad_tile_array = tile_dataset[variable_name].sel(
                time=not_passed_qa_datetimes)
        except KeyError:
            self._logger.error(
                'Bad - Could not find all times' +
                f' in not-passed {not_passed_qa_datetimes}'
            )
            return None
        tile_array = tile_array.sel(
            band=BAND).transpose(Y, X, TIME)

        bad_tile_array = bad_tile_array.sel(band=BAND).transpose(Y, X, TIME)

        # Make the coordinates that will be used to make the mode ndarray
        # a xr.DataArray
        coords = dict(
            band=tile_dataset.band,
            y=tile_dataset.y,
            x=tile_dataset.x,
            spatial_ref=tile_dataset.spatial_ref,
        )

        if output_nobservations:
            self._logger.info('Calculating n-observations in addition to mode')
            mode, nobs = self._calculate_mode(
                tile_array,
                classes,
                output_nobservations,
                logger=self._logger)
            nobs_with_band = np.zeros((BAND, mode.shape[0], mode.shape[1]))
            nobs_with_band[0, :, :] = nobs
            nobs_data_array = self._make_data_array(nobs_with_band,
                                                    coords,
                                                    name)
            variable_name = variable_name.replace('CAS', 'ETZ')
            nobs_name = f'{variable_name}.nobservations'
            nobs_raster_output_path = \
                tile_raster_output_path.replace('mode.QAD', 'nobservations')
            self._logger.info(
                f'Compute mode - Writing {nobs_name} ' +
                f'to {nobs_raster_output_path}')
            nobs_data_array.rio.to_raster(nobs_raster_output_path,
                                          dtype=np.uint16,
                                          compress='lzw')
        else:
            mode = self._calculate_mode(
                tile_array, classes, logger=self._logger)

        self._logger.info('Compute mode - Filling holes with bad data')
        mode = self._add_non_qa_data(bad_tile_array, mode,
                                     not_passed_qa_datetimes)
        self._logger.info('Compute mode - Configuring mode to data array')

        # Add the band to the mode
        mode_with_band = np.zeros((BAND, mode.shape[0], mode.shape[1]))
        mode_with_band[0, :, :] = mode

        mode_data_array = self._make_data_array(mode_with_band,
                                                coords,
                                                name)

        # Write to GTiff
        self._logger.info(
            f'Compute mode - Writing {name} to {tile_raster_output_path}')
        mode_data_array.rio.to_raster(tile_raster_output_path,
                                      dtype=np.uint8,
                                      compress='lzw')
        warpOptions = ['COMPRESS=LZW']
        warped_tile = tile_raster_output_path.replace('.tif', 'warp.tif')
        _ = gdal.Warp(warped_tile, tile_raster_output_path,
                      warpOptions=warpOptions)
        return None

    def fill_holes(
            self,
            data_array: xr.DataArray,
            reduced_stack: np.ndarray,
            datetimes_to_fill,
            nodata_value):
        for datetime in datetimes_to_fill:
            if not (reduced_stack == nodata_value).any():
                self._logger.info("No more no-data to fill")
                break
            array_per_datetime = data_array.sel(time=datetime)
            bit_rep_array_per_datetime = convert_to_bit_rep(array_per_datetime)
            reduced_stack = np.where(reduced_stack == nodata_value,
                                     bit_rep_array_per_datetime,
                                     reduced_stack)
        return reduced_stack

    @staticmethod
    def reduce_stack(algorithm: str,
                     tile_dataarray: xr.DataArray,
                     output_path: str,
                     overwrite: bool = False,
                     nodata: np.uint32 = metrics.HOST_FILL,
                     gpu=True,
                     logger=None) -> xr.DataArray:

        if output_path.exists() and not overwrite:
            log_msg = f'{output_path} already exists.'
            logger.info(log_msg)

        tile_data_array_no_band = tile_dataarray.sel(band=BAND)
        tile_data_array_prepped = tile_data_array_no_band.transpose(Y, X, TIME)
        tile_ndarray = tile_data_array_prepped.data
        tile_ndarray = np.ascontiguousarray(
            tile_ndarray, dtype=tile_ndarray.dtype)

        algorithm_to_use = eval(f"metrics.{algorithm}")
        reduced_stack = algorithm_to_use(tile_ndarray, nodata=nodata, gpu=gpu)
        coords = Composite.get_coords(tile_dataarray)
        variable_name = Composite.make_variable_name(output_path)
        reduced_stack_data_array = Composite.make_data_array(
            reduced_stack, coords, variable_name)
        return reduced_stack_data_array

    @staticmethod
    def get_coords(dataset: xr.Dataset) -> dict:
        assert hasattr(dataset, 'band')
        assert hasattr(dataset, 'y')
        assert hasattr(dataset, 'x')
        assert hasattr(dataset, 'spatial_ref')

        coords = dict(
            band=dataset.band,
            y=dataset.y,
            x=dataset.x,
            spatial_ref=dataset.spatial_ref,
        )

        return coords

    @staticmethod
    def make_variable_name(output_path):
        return output_path.stem

    @staticmethod
    def reduce_stack_legacy_mode(algorithm: str,
                                 tile_dataarray: xr.DataArray,
                                 output_path: str,
                                 overwrite: bool = False,
                                 gpu=True,
                                 logger=None) -> xr.DataArray:

        if output_path.exists() and not overwrite:
            log_msg = f'{output_path} already exists.'
            logger.info(log_msg)

        tile_data_array_no_band = tile_dataarray.sel(band=BAND)
        tile_data_array_prepped = tile_data_array_no_band.transpose(Y, X, TIME)

        classes = {0: 0, 1: 1, 2: 2}

        mode, _ = Composite._calculate_mode(tile_data_array_prepped,
                                            classes,
                                            calculate_nobservations=True,
                                            logger=logger)

        reduced_stack = np.expand_dims(mode, axis=0)

        # algorithm_to_use = eval(f"metrics.{algorithm}")
        coords = Composite.get_coords(tile_dataarray)
        variable_name = Composite.make_variable_name(output_path)
        reduced_stack_data_array = Composite.make_data_array(
            reduced_stack, coords, variable_name)
        return reduced_stack_data_array

    @staticmethod
    def reduce_stack_legacy_mode_w_qa(algorithm: str,
                                      tile_dataarray: xr.DataArray,
                                      bad_tile_array: xr.DataArray,
                                      not_passed_qa_datetimes: list,
                                      output_path: str,
                                      overwrite: bool = False,
                                      gpu=True,
                                      logger=None) -> xr.DataArray:

        if output_path.exists() and not overwrite:
            log_msg = f'{output_path} already exists.'
            logger.info(log_msg)

        tile_data_array_no_band = tile_dataarray.sel(band=BAND)
        tile_data_array_prepped = tile_data_array_no_band.transpose(Y, X, TIME)

        bad_tile_data_array_no_band = bad_tile_array.sel(band=BAND)
        bad_tile_data_array_prepped = bad_tile_data_array_no_band.transpose(
            Y, X, TIME)

        classes = {0: 0, 1: 1, 2: 2}

        mode, _ = Composite._calculate_mode(tile_data_array_prepped,
                                            classes,
                                            calculate_nobservations=True,
                                            logger=logger)
        print(mode.shape)
        mode = Composite._add_non_qa_data(bad_tile_data_array_prepped, mode,
                                          not_passed_qa_datetimes,
                                          logger=logger)
        print(mode.shape)
        reduced_stack = np.expand_dims(mode, axis=0)

        # algorithm_to_use = eval(f"metrics.{algorithm}")
        coords = Composite.get_coords(tile_dataarray)
        variable_name = Composite.make_variable_name(output_path)
        reduced_stack_data_array = Composite.make_data_array(
            reduced_stack, coords, variable_name)
        return reduced_stack_data_array

    @staticmethod
    def reduce_stack_legacy_nobs(algorithm: str,
                                 tile_dataarray: xr.DataArray,
                                 output_path: str,
                                 overwrite: bool = False,
                                 gpu=True,
                                 logger=None) -> xr.DataArray:

        if output_path.exists() and not overwrite:
            log_msg = f'{output_path} already exists.'
            logger.info(log_msg)

        tile_data_array_no_band = tile_dataarray.sel(band=BAND)
        tile_data_array_prepped = tile_data_array_no_band.transpose(Y, X, TIME)

        classes = {0: 0, 1: 1, 2: 2}

        _, nobs = Composite._calculate_mode(tile_data_array_prepped,
                                            classes,
                                            calculate_nobservations=True,
                                            logger=logger)

        reduced_stack = np.expand_dims(nobs, axis=0)

        # algorithm_to_use = eval(f"metrics.{algorithm}")
        coords = Composite.get_coords(tile_dataarray)
        variable_name = Composite.make_variable_name(output_path)
        reduced_stack_data_array = Composite.make_data_array(
            reduced_stack, coords, variable_name)
        return reduced_stack_data_array

    # --------------------------------------------------------------------------
    # SKELETON FUNCTION PT. 1
    # Change function name to fit alg
    # Change "alg" out with whatever you're doing
    # --------------------------------------------------------------------------

    def calculate_algorithm_per_tile(
            self,
            tile_path: str,
            tile_dataset_input: xr.Dataset = None) -> None:
        """
        Given a landcover zarr or dataset, calculate the mode
        and write to GTiff
        """
        tile_dataset = tile_dataset_input if tile_dataset_input \
            else xr.open_zarr(tile_path)
        variable_name = os.path.basename(tile_path).split('.zarr')[0]
        # Select the array without the band, transpose to time-last format
        tile_array = tile_dataset[variable_name].sel(
            band=BAND).transpose(Y, X, TIME)
        output_array = self._calculate_alg(tile_array, logger=self._logger)
        self._logger.info('Compute alg')
        # Add the band to the mode
        output_array_with_band = np.zeros((BAND, output_array.shape[0],
                                           output_array.shape[1]))
        output_array_with_band[0, :, :] = output_array

        # Make the coordinates that will be used to make the mode ndarray
        # a xr.DataArray
        coords = dict(
            band=tile_dataset.band,
            y=tile_dataset.y,
            x=tile_dataset.x,
            spatial_ref=tile_dataset.spatial_ref,
        )
        name = '{}.alg.{}'.format(variable_name, self._experiment_name)
        self._logger.info(f'Compute alg - Appending {name} to {tile_path}')
        mode_data_array = self._make_data_array(output_array_with_band,
                                                coords,
                                                name)

        # Write to GTiff
        tile_raster_path = tile_path.replace('.zarr', f'{name}.tif')
        mode_data_array.rio.to_raster(tile_raster_path)
        return None

    @staticmethod
    def make_data_array(
            ndarray: np.ndarray,
            coords: dict,
            name: str) -> xr.DataArray:
        """
        Given a ndarray, make it a Xarray DataArray
        """
        data_array = xr.DataArray(
            data=ndarray,
            dims=['band', 'y', 'x'],
            coords=coords,
            attrs=dict(
                description="Mode of model results"
            ),
        )
        data_array.name = name
        return data_array

    @staticmethod
    def _calculate_mode(tile_array: xr.DataArray,
                        classes: dict,
                        calculate_nobservations: bool = True,
                        logger: logging.Logger = None):
        """
        Object-oriented wrapper for mode calculation function.
        """
        return calculate_mode(tile_array,
                              classes,
                              calculate_nobservations=calculate_nobservations,
                              logger=logger)


if __name__ == '__main__':
    from vhr_composite.model.utils import TqdmLoggingHandler
    import pathlib
    import time

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    sh = TqdmLoggingHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    test_shape = (1, 500, 500, 20)
    test_array = np.random.randint(0, 3, size=test_shape)
    logger.info(test_array.max())
    test_array = test_array.astype(np.uint8)

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
    st = time.time()
    reduced_stack = Composite.reduce_stack('multi_mode',
                                           da,
                                           output_path,
                                           overwrite=False,
                                           gpu=True,
                                           logger=logger)
    et = time.time()
    logger.info(f'{round((et - st), 5)}: GPU new alg')
    logger.info(reduced_stack.data)

    st = time.time()
    legacy_style = Composite.reduce_stack_legacy_mode('mutli_mode',
                                                      da,
                                                      output_path,
                                                      overwrite=False,
                                                      gpu=False,
                                                      logger=logger)
    et = time.time()
    logger.info(f'{round((et - st), 5)}: CPU old alg')
    logger.info(legacy_style.data.max())
    # legacy_decoded = decode_legacy(legacy_style.data)
    # logger.info(legacy_decoded.max())
    logger.info(legacy_style.data)

    np.testing.assert_equal(reduced_stack.data, legacy_style.data)

    st = time.time()
    num_obs_stack = Composite.reduce_stack('number_observations',
                                           da,
                                           output_path,
                                           overwrite=False,
                                           gpu=False,
                                           logger=logger)
    et = time.time()
    logger.info(f'{round((et - st), 5)}: CPU NOBS NEW')

    st = time.time()
    num_obs_legacy = Composite.reduce_stack_legacy_nobs('number_observations',
                                                        da,
                                                        output_path,
                                                        overwrite=False,
                                                        gpu=False,
                                                        logger=logger)
    et = time.time()
    logger.info(f'{round((et - st), 5)}: CPU NOBS OLD')
    np.testing.assert_equal(num_obs_stack.data, num_obs_legacy.data)
