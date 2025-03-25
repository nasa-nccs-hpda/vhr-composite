import os
import pdb
import math
import tqdm
import shutil
import logging
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd

from osgeo import gdal

from typing import Union

from vhr_composite.model import metrics
from vhr_composite.model.utils import convert_to_bit_rep
from vhr_composite.model.metrics import calculate_mode
from vhr_composite.model.metrics import calculate_alg
from vhr_composite.model.metrics import CLASS_0, CLASS_1, CLASS_2, CLASS_3, CLASS_4, CLASS_5
from vhr_composite.model.metrics import CLASS_0_ALIAS, CLASS_1_ALIAS, CLASS_2_ALIAS, CLASS_3_ALIAS, CLASS_4_ALIAS, CLASS_5_ALIAS

TIME: str = "time"
X: str = "x"
Y: str = "y"
BAND: int = 1
CLOUD_CLEAR_VALUE: int = 0
NO_DATA: int = 255
OTHER_DATA: int = 255
FILL_DATA: int = 10
BURN_AREA_VALUE: int = 15
STRIP_FILE_PATH_KEY: str = 'landcover'
MASK_FILE_PATH_KEY: str = 'cloudmask'
GRID_CELL_NAME_PRE_STR: str = 'CAS.M1BS'
DIM: str = "time"

# default max classes = 6
CLASS_VALUES: list = [i for i in range(6)]


class Composite(object):

    def __init__(self,
                 name: str,
                 grid_geopackage_path: str,
                 model_output_geopackage_path: str,
                 output_dir: str
            ) -> None:
        """
        Initializes the Composite object
        """
        self._name = name
        self._grid_geopackage_path = grid_geopackage_path
        self._model_output_geopackage_path = model_output_geopackage_path

        logging.info('*' * 60)
        logging.info('* Initializing compositing *')
        logging.info('*' * 60)

        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        logging.info(f'Using output directory at {self._output_dir}')

        if not os.path.exists(self._grid_geopackage_path) or \
                not os.path.exists(self._model_output_geopackage_path):
            msg = f'{self._grid_geopackage_path} does not exist'
            raise FileNotFoundError(msg)

        self._grid = gpd.read_file(self._grid_geopackage_path)
        logging.info(f'Loaded grid: {self._grid_geopackage_path}')

        self._model_output = \
            gpd.read_file(self._model_output_geopackage_path)
        logging.info(f'Loaded metadata: {self._model_output_geopackage_path}')

    def generate_grid_via_intersect(self, calculate_mode: bool = False,
                                    classes: dict = None) -> None:
        """
        Generate the gridded zarrs from the strips that intersect
        each grid cell
        """
        logging.info('Calculating intersect')
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

    def generate_grid(
                self,
                tile_list: list,
                grid_cell_name_pre_str: str = 'COM.M1BS'
            ) -> None:
        """
        Generate the gridded zarrs from a pre-calculated intersection
        of grid cells
        """
        for tile in tqdm.tqdm(tile_list):
            logging.info(f'{tile} - Processing {tile}')
            name = '{}.{}'.format(grid_cell_name_pre_str, tile)
            tile_path = os.path.join(self._output_dir, f'{name}.zarr')
            if os.path.exists(tile_path):
                logging.info(f'{tile}- {tile_path} already exists')
                continue
            tile_df = self._grid[self._grid['tile'] == tile]
            strips = self._model_output[
                self._model_output['tile'] == tile]
            len_strips = len(strips)
            logging.info(f'{tile} - Processing {len_strips} strips')
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

            logging.info(f'{tile} - Writing to {tile_path}')
            concat_dataset.to_zarr(tile_path)
            arrays = None
            concat_array = None
            concat_dataset = None
        return None

    @staticmethod
    def reduce_stack(algorithm: str,
                     tile_dataarray: xr.DataArray,
                     output_path: str,
                     overwrite: bool = False,
                     nodata: np.uint32 = metrics.HOST_FILL,
                     gpu=True) -> xr.DataArray:

        if output_path.exists() and not overwrite:
            log_msg = f'{output_path} already exists.'
            logging.info(log_msg)

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

    def generate_single_grid(
                self,
                tile: str,
                write_out: bool = False,
                overwrite: bool = False,
                fill_value: int = 10,
                burn_area_value: int = BURN_AREA_VALUE,
                grid_cell_name_pre_str: str = 'COM.M1BS'
            ) -> Union[xr.Dataset, None]:
        """
        Generate the gridded zarrs from a pre-calculated intersection
        of grid cells
        """
        if overwrite:
            logging.info(f'OVERWRITE zarrs is on')

        logging.info(f'{tile} - Processing {tile}')
        name = '{}.{}'.format(grid_cell_name_pre_str, tile)
        tile_path = os.path.join(self._output_dir, f'{name}.zarr')

        # enable overwrite zarr and delete zarr path if it already exists
        # delete zarr to avoid overwrite err - this will delete entire
        # group not just dataset which is fine for here
        if os.path.exists(tile_path) and not overwrite:
            logging.info(
                f'{tile}- {tile_path} already exists or overwrite off')
            return xr.open_zarr(tile_path)
        elif os.path.exists(tile_path) and overwrite:
            shutil.rmtree(tile_path)

        tile_df = self._grid[self._grid['tile'] == tile]
        strips = self._model_output[
            self._model_output['tile'] == tile]
        len_strips = len(strips)
        logging.info(f'{tile} - Processing {len_strips} strips')
        tiles_df = [tile_df for _ in range(len_strips)]
        burn_values = [burn_area_value for _ in range(len_strips)]

        landcovers = strips['landcover'].values
        cloudmasks = strips['cloudmask'].values
        datetimes = strips['datetime'].values

        # Check if tile in grid .gpkg #*MW
        if tile_df.empty:
            logging.info(
                f'Tile {tile} is not in grid file')
            return None

        if len_strips < 1:
            return None

        arrays = list(map(
            Composite.strip_to_grid,
            landcovers,
            cloudmasks,
            datetimes,
            tiles_df,
            burn_values
        ))
        arrays = [array for array in arrays if array is not None]

        n_strips = len(arrays)
        if n_strips < 1:
            return None

        # concatenate on the time dimension
        concat_array = xr.concat(
            arrays, dim='time', fill_value=fill_value)
        concat_array.rio.write_nodata(fill_value)

        # ensure attributes eg FillValue are written to zarr
        concat_dataset = concat_array.to_dataset(
            name=name, promote_attrs=True)

        del arrays, concat_array

        if write_out:
            logging.info(
                f'{tile} - Writing ({n_strips} strips) ' +
                f'to {tile_path}'
            )
            concat_dataset.to_zarr(tile_path)

        return concat_dataset

    #*MW new function for calculate_binary_stats = True in .yaml
    @staticmethod
    def calculate_binary_class_stats(stat: str,
                                     tile_dataarray: xr.DataArray,
                                     output_path: str,
                                     overwrite: bool = False,
                                     nodata: np.uint32 = metrics.HOST_FILL,
                                     class_values: int = CLASS_VALUES, #*MW added
                                     gpu=True,
                                    ) -> xr.DataArray:

        """Similar to reduce_stack, but instead of reducing 4D time-series stack to a 3D arr
        [eg (band: 1, time: n_t, y: 5000, x: 5000) => (band: 1, y: 5000, x: 5000)], 
        this reduces the stack from time for each land cover class value and returns 
        a 4D array that has a 'class' dimension instead of a time dimension [eg 
        (band: 1, time: n_t, y: 5000, x: 5000) => (band: 1, time: n_c, y: 5000, x:5000)]
        Output could potentially be used in a different reduce_stack method that 
        gets composite land cover according to rules
        """
        # import pdb; pdb.set_trace()
        if output_path.exists() and not overwrite:
            log_msg = f'{output_path} already exists.'
            logging.info(log_msg)
            return None

        tile_data_array_no_band = tile_dataarray.sel(band=BAND)
        tile_data_array_prepped = tile_data_array_no_band.transpose(Y, X, TIME)
        tile_ndarray = tile_data_array_prepped.data
        tile_ndarray = np.ascontiguousarray(
            tile_ndarray, dtype=tile_ndarray.dtype)

        #*TD deal
        # #*MW tile might be all nodata (why? idk) and this will fail. may be moot (max classes should be user defined)
        try:
            max_class = tile_ndarray[tile_ndarray != nodata].max() #*TD get this from config - maybe some tiles have max_class=6, others 4
        except ValueError: #*TD maybe - either return None to skip binary calc stuff or return ND? maybe remove altogther
            #import pdb; pdb.set_trace()
            logging.info(f'Could not process {output_path} due to NoData input array values')
            return None
        # max_class = tile_ndarray[tile_ndarray != nodata].max() #*TD get this from config - maybe some tiles have max_class=6, others 4
            
        # max_class = 5 #6class ==> 0 to 5 #*TODO deal #*MW delete do not need max class anymore
        

        # faster approach        
        d1, d2, t = tile_ndarray.shape
        out_arr = np.zeros((d1, d2, len(class_values)), dtype=np.float32)
    
        if nodata is not None:
            valid_mask = tile_ndarray != nodata
        else:
            valid_mask = np.ones_like(tile_ndarray, dtype=bool)
        
        # iterate through each possible class value (0 to max_value)
        # for classn in range(max_class_value + 1):class_values
        for classn in class_values: #*MW
            
            # count class occurrences of the current value for valid pixels
            out_arr[..., classn] = np.sum((tile_ndarray == classn)
                                                           & valid_mask, 
                                                                axis=2)
        # convert to % of total valid observations
        valid_obs = np.sum(valid_mask, axis=2)  # total n valid obsvervation per pixel location
        valid_obs[valid_obs == 0] = nodata #*TD deal with nodata (output, others?) -  pixels with 0 observations should be nodata in output. Presume nobs could not be > specified output nodata value (val should be speficied this way anyways) so we can use as a placeholder for now to avoid divide by 0 error
        out_arr /= valid_obs[..., None] # out_arr = out_arr / valid_obs (with an added third dimension)
        out_arr *= 100 # out_arr = out_arr * 100
        
        # print(np.unique(out_arr.sum(axis=2))) # should essentially be 100 (guess sometimes it might not due to rounding errs)
        output_array = out_arr.round().astype('uint8') #*TD specify data type
        out_arr = None

        #*MW use valid_obs which is nodata where all inputs are = 0 to set output array to nodata
        output_array[(valid_obs == nodata)] = nodata

        # algorithm_to_use = eval(f"metrics.{algorithm}")
        # reduced_stack = algorithm_to_use(tile_ndarray, nodata=nodata, gpu=gpu)
        coords = Composite.get_coords(tile_dataarray)
        # modify coords from 'band' = [1] to 'class' = [0, 1, 2, .., max_class]
        # class_arr = [f'class={i}' for i in range(max_class_value+1)]
        class_arr = [f'class={i}' for i in class_values]
            
        # coords['band'] = xr.DataArray([i for i in range(max_class_value+1)],
        coords['band'] = xr.DataArray([i for i in class_values],
                        dims=['band'], name='band',
                        coords={'band': class_arr, 
                        'spatial_ref': coords['band'].coords['spatial_ref']})
        
        # for binary class pct output, shape is eg (5000, 5000, n_classes), 
        # tell function by passing the correct order via dims
        variable_name = Composite.make_variable_name(output_path)
        output_data_array = Composite.make_data_array(output_array, coords, 
                                    variable_name, dims=['y', 'x', 'band'],
                                    desc=f'Binary class statistics - {stat}')
        # return array as rio.to_raster() expects w band first
        output_data_array = output_data_array.transpose('band', 'y', 'x')
        
        return output_data_array

    @staticmethod
    def make_variable_name(output_path):
        return output_path.stem

    @staticmethod
    def make_data_array(
            ndarray: np.ndarray,
            coords: dict,
            name: str,
            dims: list = ['band', 'y', 'x'], #*MW make flexible for ndarray inputs with different dimensions
            desc: str = "Mode of model results",
            ) -> xr.DataArray:
        """
        Given a ndarray, make it a Xarray DataArray
        """
        data_array = xr.DataArray(
            data=ndarray,
            dims=dims,
            coords=coords,
            attrs=dict(
                description=desc
            ),
        )
        data_array.name = name
        return data_array

    def fill_holes(
            self,
            data_array: xr.DataArray,
            reduced_stack: np.ndarray,
            datetimes_to_fill,
            nodata_value):
        for datetime in datetimes_to_fill:
            if not (reduced_stack == nodata_value).any():
                logging.info("No more no-data to fill")
                break
            array_per_datetime = data_array.sel(time=datetime)
            bit_rep_array_per_datetime = convert_to_bit_rep(array_per_datetime)
            reduced_stack = np.where(reduced_stack == nodata_value,
                                     bit_rep_array_per_datetime,
                                     reduced_stack)
        return reduced_stack

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
        logging.info(f'Found {len(intersection_dict)} intersections')
        return intersection_dict

    @staticmethod
    def strip_to_grid_prior_wooten(
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

        # open land cover product
        try:
            strip_data_array = rxr.open_rasterio(land_cover_path)
        except rasterio.errors.RasterioIOError:
            error_file = os.path.basename(
                land_cover_path).replace('.tif', '.txt')
            with open(error_file, 'w') as fh:
                fh.writelines([f'{land_cover_path}\n'])
            return None

        try:
            cloud_mask_data_array = rxr.open_rasterio(cloud_mask_path)
        except (rasterio.errors.RasterioIOError, TypeError):
            error_file = os.path.basename(
                land_cover_path).replace('.tif', '.txt')
            with open(error_file, 'w') as fh:
                fh.writelines([f'{land_cover_path}\n'])

            # assign clear cloud mask array (ignore if there are clouds)
            cloud_mask_data_array = strip_data_array[0:1, :, :]
            cloud_mask_data_array = cloud_mask_data_array.where(
                cloud_mask_data_array != 0, 0)

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

    @staticmethod
    def strip_to_grid(
            land_cover_path: str,
            cloud_mask_path: str,
            timestamp: pd._libs.tslibs.timestamps.Timestamp,
            grid_geodataframe: gpd.GeoDataFrame,
            burn_area_value: int = BURN_AREA_VALUE,
                    ) -> Union[xr.DataArray, None]:
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

        November 2024: For whatever reason, concat in generate_single_grid()
        stopped working as expected/usual with the listed output of this function which is called 
        for each landcover input/tile nd was causing weird/messed up output tiles
        due to concat returning all arrays concatenated together spatially instead of in as a 
        stacked tile, causing the extent to go way outside of the tile bounds. 
        All of the crap added to this function was to get around this issue by forcing the arrays to be in the exact same grid (and 
        other issues that were new and/or caused by 'fixes' - it's a hacky soln 
        but is working for now in amhara and 3sl). in generate_single_grid(), concat is supposed 
        to take the tiled/masked landcover output stack list and to return an 
        array with the geographic union of the input arrays, but this suddenly 
        was not working. If issue turns out due to container/environment,
        should try to go back to the old func
        """

        #*MW need to check that the tile is in the provided grid, otherwise 
        # function will fail and no more tiles will be run. In theory this 
        # should not usually be a problem due to len_strips check in 
        # composite_single_grid, but if grid file was updated to eg exclude 
        # tiles while the gridded footprints file was not remade then there 
        # will be strips for the tile, but no geometry in the grid and 
        # we don't wanna process that tile - check if necessary?
        if grid_geodataframe.empty:
            return None
        
        # open land cover product
        try:
            strip_data_array = rxr.open_rasterio(land_cover_path)
        except rasterio.errors.RasterioIOError:
            error_file = os.path.basename(
                land_cover_path).replace('.tif', '.txt')
            with open(error_file, 'w') as fh:
                fh.writelines([f'{land_cover_path}\n'])
            return None

        if strip_data_array.rio.transform().e > 0: #*MW temp to check transform weirdness
            pdb.set_trace()

        try:
            cloud_mask_data_array = rxr.open_rasterio(cloud_mask_path)
        except (rasterio.errors.RasterioIOError, TypeError):
            error_file = os.path.basename(
                land_cover_path).replace('.tif', '.cloudmask.txt')
            with open(error_file, 'w') as fh:
                fh.writelines([f'{land_cover_path}\n'])

            # assign clear cloud mask array (ignore if there are clouds)
            cloud_mask_data_array = strip_data_array[0:1, :, :]
            cloud_mask_data_array = cloud_mask_data_array.where(
                cloud_mask_data_array != 0, 0)

        if cloud_mask_data_array.rio.transform().e > 0: #*MW temp to check transform weirdness
            import pdb; pdb.set_trace()
            
        geometry_to_clip = grid_geodataframe.geometry.values
        geometry_crs = grid_geodataframe.crs
        
        # *MW get info for fixing concat issue with padding - get this before 
        # clipping in case clipping changes pixel size - other solution could 
        # be to move this back down above strip_data_array0 = None and resample
        # if resolutions are different, but not sure if this would cause other errors. 
        #*MW set refr_arr to force arr coordinates and determine if padding needed
        xmin, ymin, xmax, ymax = geometry_to_clip.total_bounds
        x_res = strip_data_array.rio.resolution()[0]
        y_res = abs( strip_data_array.rio.resolution()[1])
        x_pixels = int(round(xmax-xmin) / x_res)
        y_pixels = int(round(ymax-ymin) / y_res)

        #*MW - sometimes (esp for trimmed outputs - tho may just be an issue while bypassing the footprinting - doesnt hurt) the strip may no longer be in the tile bounds 
        try:
            strip_data_array = strip_data_array.rio.clip(geometry_to_clip,
                                                             crs = geometry_crs)
        except rxr.exceptions.NoDataInBounds:
            return None

        #*MW temp - write strip array out to tif (clipped/cloud masked) - var prep
        write_temp_clipped = False
        if write_temp_clipped:
            bn = os.path.basename(land_cover_path)
            bnn = f"{grid_geodataframe['tile'].values[0]}__{bn.replace('.tif', '__clip0.tif')}"
            odir = '/explore/nobackup/projects/3sl/development/cnn_landcover_composite/ethopia-v10.1.test/'
            to0 = os.path.join(odir, bnn) # clip before "fix" - before cloud masking stuff right after clip
            to1 = os.path.join(odir, bnn.replace('clip0', 'clip1')) # clip before "fix" - after cloud masking stuff
            to2 = os.path.join(odir, bnn.replace('clip0', 'clip2')) # clip after "fix"
    
            #*MW temp write out clip0 (after clip befor emask)
            strip_data_array.rio.to_raster(to0, dtype = np.uint8, compress='lzw')
        
        # """
        #*MW 3. attempt to solving concat issue 0 try instead padding the clipped 
        # array (instead of entire strip), then clipping again, if initial clip is not
        # the size of tile 
        
        #*MW have to do cloud masking before fixing coords
        fill_value = strip_data_array._FillValue 
        try:
            NO_DATA=OTHER_DATA=fill_value #* come back to this - for now set everything to be the same (255)
            strip_data_array = strip_data_array.where((
                (cloud_mask_data_array == 0) &
                (strip_data_array != burn_area_value) & #*MW 
                (strip_data_array != NO_DATA) #*MW use strip_data_array._FillValue?
                ), other=OTHER_DATA).astype(np.uint8)
            
        except ValueError:
            #*MW sometimes landcover is like 1 pixel larger than the cloud mask. buffer cloud mask 2 pixelsif error with above:]
            # pad and try again
            # occassionally the nodata border in the cloud mask may actually be hundred some
            # pixels from covering landcover (at least for trimmed? why tho? seems backwards)
            # so just pad 500 JIX
            cloud_pad = 500
            cloud = cloud_mask_data_array.pad(x=cloud_pad, constant_values=255) # ch mode?
            cloud = cloud.pad(y=cloud_pad, constant_values=255)
            
            x_padded=cloud_mask_data_array.x.pad(x=cloud_pad, mode='linear_ramp', end_values=(cloud_mask_data_array.x.min()-cloud_pad,cloud_mask_data_array.x.max()+cloud_pad))

            y_padded=cloud_mask_data_array.y.pad(y=cloud_pad, mode='linear_ramp', 
                                                  end_values=(cloud_mask_data_array.y.max()+cloud_pad, cloud_mask_data_array.y.min()-cloud_pad))
            
            del cloud_mask_data_array
            cloud=cloud.assign_coords(x = x_padded)
            cloud=cloud.assign_coords(y = y_padded)

            try:
                strip_data_array = strip_data_array.where((
                    (cloud == 0) &
                    (strip_data_array != burn_area_value) & #*MW
                    (strip_data_array != NO_DATA) #*MW use strip_data_array._FillValue?
                    ), other=OTHER_DATA).astype(np.uint8)
            except ValueError:
                print('cloud padding hack did not work') # eg padding not large enough? or etc
                import pdb;pdb.set_trace() 

            del cloud, y_padded, x_padded


        # if x_pixels != 5000 and strip_data_array.x.size >= x_pixels:
        #     import pdb; pdb.set_trace()

        #*MW temp write out clip1 (after clip after emask)
        if write_temp_clipped:
            strip_data_array.rio.to_raster(to1, dtype = np.uint8, compress='lzw')
        # import pdb; pdb.set_trace()#*MW temp

        # check if arrays cover the tile in x/y directions. pad and clip if not
        strip_data_array0 = None
        if strip_data_array.x.size < x_pixels:

            # determine max amount of pixels needed to ensure full tile coverage
            x_pad = x_pixels-strip_data_array.x.size

            # pad array with constant value in both x directions
            strip_data_array0=strip_data_array.pad(x=x_pad, 
                                                     constant_values=fill_value)
            
            # pad x coords and assign new min/max extents
            xmin_pad = strip_data_array.x.values.min() - x_pad*x_res
            xmax_pad = strip_data_array.x.values.max() + x_pad*x_res
            if xmin_pad < 0 or xmax_pad < 0: # note that this may not work across equator/UTM zone?
                import pdb; pdb.set_trace()
            # new x coord array - note that x_padded goes from min to max as expected
            x_padded = strip_data_array.x.pad(x=x_pad, mode='linear_ramp', 
                                              end_values=(xmin_pad, xmax_pad))
            strip_data_array0 = strip_data_array0.assign_coords(x = x_padded)
            
            del x_padded, xmin_pad, xmax_pad
            
            strip_data_array = strip_data_array0 # in case fixing both
        
        if strip_data_array.y.size < y_pixels:

            # determine max amount of pixels needed to ensure full tile coverage
            y_pad = y_pixels-strip_data_array.y.size

            # pad array with constant value in both y directions
            strip_data_array0=strip_data_array.pad(y=y_pad, 
                                                   constant_values=fill_value)
            
            # pad y coords and assign new min/max extents
            ymin_pad = strip_data_array.y.values.min() - y_pad*y_res
            ymax_pad = strip_data_array.y.values.max() + y_pad*y_res
            if ymin_pad < 0 or ymax_pad < 0: # note that this may not work across equator/UTM zone?
                import pdb; pdb.set_trace()
            # new y coord array - note that y_padded goes from max to min
            y_padded = strip_data_array.y.pad(y=y_pad, mode='linear_ramp', 
                                              end_values=(ymax_pad, ymin_pad))
            strip_data_array0 = strip_data_array0.assign_coords(y = y_padded)

            del y_padded, ymin_pad, ymax_pad

        if strip_data_array0 is not None:
            # clip and force clipped da to take on correct x/y coords for tile
            strip_data_array = strip_data_array0.rio.clip(geometry_to_clip,
                                                                    crs=geometry_crs)
        del strip_data_array0
        
        #*MW force coords to match tile because this has been an issue for some strips/tiles
        # that did not need to be padded (x or y coords were slightly off)
        # 11/30 - in some cases (the subset of the tiles with strips that needed to be
        # reprojected - eg the farWest  Amhara tiles), the size of the tile (eg
        # x_pixels=4999) may be different from the result of the clip (eg strip_data_array.x.size)
        # The reqason for the above is the 4999 throws off the resulting pixel 
        # resolution to be incompatible with the (eg) 2m  resolution and the solution
        # is to "round" x_pixels or y_pixels for rare cases like the 4999
        # ACTUALLY - this should now be addressed above with rounding for x_pixels
        # and y_pixels calculation I THINK
        xx = np.linspace(math.floor(xmin)+x_res, math.floor(xmax), x_pixels)
        yy = np.linspace(math.ceil(ymax)-y_res, math.ceil(ymin), y_pixels)

        #try: # This should not be an issue anymore with the round in x/y_pixels =
            # strip_data_array = strip_data_array.assign_coords(x = xx)
        #except ValueError:
        #    import pdb; pdb.set_trace()
        strip_data_array = strip_data_array.assign_coords(x = xx)
        strip_data_array = strip_data_array.assign_coords(y = yy)
       
        #*MW temp write out clip1 (after clip after emask)
        if write_temp_clipped:
            strip_data_array.rio.to_raster(to2, dtype = np.uint8, 
                                                                 compress='lzw')
        
        strip_data_array = strip_data_array.assign_coords(time=timestamp)
        strip_data_array = strip_data_array.expand_dims(dim=TIME)

        del xx, yy

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
        mode = self._calculate_mode(tile_array, classes)
        logging.info('Compute mode - Configuring mode to data array')
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
        logging.info(f'Compute mode - Appending {name} to {tile_path}')
        mode_data_array = self._make_data_array(mode_with_band,
                                                coords,
                                                name)

        # Write to GTiff
        tile_raster_path = tile_path.replace('.zarr', f'{name}.tif')
        mode_data_array.rio.to_raster(tile_raster_path)
        return None

    def _add_non_qa_data(self,
                         tile_datarray: xr.DataArray,
                         mode: np.ndarray,
                         not_passed_qa_datetimes: list) -> np.ndarray:
        for qa_datetime_to_select in not_passed_qa_datetimes:
            if not (mode == OTHER_DATA).any():
                logging.info('No more nodata, exiting')
                break
            model_output_per_datetime = tile_datarray.sel(
                time=qa_datetime_to_select)
            model_output_per_datetime_ndarray = \
                model_output_per_datetime.values

            print("FIRST STEP: ", np.unique(model_output_per_datetime_ndarray))

            model_output_per_datetime_ndarray = np.where(
                model_output_per_datetime_ndarray == CLASS_0,
                CLASS_0_ALIAS,
                model_output_per_datetime_ndarray)

            print("SECOND STEP: ", np.unique(model_output_per_datetime_ndarray))

            model_output_per_datetime_ndarray = np.where(
                model_output_per_datetime_ndarray == CLASS_1,
                CLASS_1_ALIAS,
                model_output_per_datetime_ndarray)

            print("THIRD STEP: ", np.unique(model_output_per_datetime_ndarray))

            model_output_per_datetime_ndarray = np.where(
                model_output_per_datetime_ndarray == CLASS_2,
                CLASS_2_ALIAS,
                model_output_per_datetime_ndarray)
            model_output_per_datetime_ndarray = np.where(
                model_output_per_datetime_ndarray == CLASS_3,
                CLASS_3_ALIAS,
                model_output_per_datetime_ndarray)
            model_output_per_datetime_ndarray = np.where(
                model_output_per_datetime_ndarray == CLASS_4,
                CLASS_4_ALIAS,
                model_output_per_datetime_ndarray)
            model_output_per_datetime_ndarray = np.where(
                model_output_per_datetime_ndarray == CLASS_5,
                CLASS_5_ALIAS,
                model_output_per_datetime_ndarray)
            mode = np.where(mode == OTHER_DATA,
                            model_output_per_datetime_ndarray, mode)
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
        name = f'{variable_name}.mode'

        if os.path.exists(tile_raster_output_path):
            logging.info(f'{tile_raster_output_path} already exists.')
            return None

        try:
            # Select the array without the band, transpose to time-last format
            tile_array = \
                tile_dataset[variable_name].sel(time=rows_to_use)
        except KeyError:

            # There are some cases where folks might have not predicted
            # all the files that were available. In this case we select
            # the available ones.

            logging.error(
                f'Could not find all times in passed {rows_to_use}')
            logging.info('Selecting the available bands only')

            rows_to_use = list(
                set(rows_to_use).intersection(
                    tile_dataset[variable_name].time.values))

            # if even after the filter we cannot find any files, exit
            if len(rows_to_use) < 1:
                logging.error('No files were matched.')
                return None

            # Select the array without the band
            # transpose to time-last format
            tile_array = \
                tile_dataset[variable_name].sel(time=rows_to_use)

        logging.info(tile_array.time)
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
            logging.info('Calculating n-observations in addition to mode')
            mode, nobs = self._calculate_mode(
                tile_array,
                classes,
                output_nobservations)
            nobs_with_band = np.zeros((BAND, mode.shape[0], mode.shape[1]))
            nobs_with_band[0, :, :] = nobs
            nobs_data_array = self._make_data_array(nobs_with_band,
                                                    coords,
                                                    name)
            nobs_name = f'{variable_name}.nobservations.noqa'
            nobs_raster_output_path = \
                tile_raster_output_path.replace('mode', 'nobservations')
            logging.info(
                f'Compute mode - Writing {nobs_name} ' +
                f'to {nobs_raster_output_path}')
            nobs_data_array.rio.to_raster(nobs_raster_output_path,
                                          dtype=np.uint16,
                                          compress='lzw')
        else:
            mode = self._calculate_mode(
                tile_array, classes)

        # Add the band to the mode
        mode_with_band = np.zeros((BAND, mode.shape[0], mode.shape[1]))
        mode_with_band[0, :, :] = mode

        mode_data_array = self._make_data_array(mode_with_band,
                                                coords,
                                                name)

        # Write to GTiff
        logging.info(
            f'Compute mode - Writing {name} to {tile_raster_output_path}')
        mode_data_array.rio.to_raster(tile_raster_output_path,
                                      dtype=np.uint8,
                                      compress='lzw')
        warpOptions = ['COMPRESS=LZW']
        warped_tile = tile_raster_output_path.replace('.tif', 'warp.tif')
        _ = gdal.Warp(warped_tile, tile_raster_output_path, warpOptions=warpOptions)

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

        logging.info(variable_name)

        name = f'{variable_name}.mode'

        if os.path.exists(tile_raster_output_path):
            logging.info(f'{tile_raster_output_path} already exists.')
            return None

        try:
            # Select the array without the band, transpose to time-last format
            tile_array = \
                tile_dataset[variable_name].sel(time=passed_qa_datetimes)
        except KeyError as e:
            print(e)
            logging.error(
                f'Good - Could not find all times in passed {passed_qa_datetimes}')
            return None

        logging.info(tile_array.time)
        try:
            bad_tile_array = tile_dataset[variable_name].sel(
                time=not_passed_qa_datetimes)
        except KeyError:
            logging.error(
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
            logging.info('Calculating n-observations in addition to mode')
            mode, nobs = self._calculate_mode(
                tile_array,
                classes,
                output_nobservations)
            nobs_with_band = np.zeros((BAND, mode.shape[0], mode.shape[1]))
            nobs_with_band[0, :, :] = nobs
            nobs_data_array = self._make_data_array(nobs_with_band,
                                                    coords,
                                                    name)
            variable_name = variable_name.replace('CAS', 'ETZ')
            nobs_name = f'{variable_name}.nobservations'
            nobs_raster_output_path = \
                tile_raster_output_path.replace('mode.QAD', 'nobservations')
            logging.info(
                f'Compute mode - Writing {nobs_name} ' +
                f'to {nobs_raster_output_path}')
            nobs_data_array.rio.to_raster(nobs_raster_output_path,
                                          dtype=np.uint16,
                                          compress='lzw')
        else:
            mode = self._calculate_mode(
                tile_array, classes)

        logging.info('Compute mode - Filling holes with bad data')
        mode = self._add_non_qa_data(bad_tile_array, mode,
                                     not_passed_qa_datetimes)
        logging.info('Compute mode - Configuring mode to data array')

        # Add the band to the mode
        mode_with_band = np.zeros((BAND, mode.shape[0], mode.shape[1]))
        mode_with_band[0, :, :] = mode

        mode_data_array = self._make_data_array(mode_with_band,
                                                coords,
                                                name)

        # Write to GTiff
        logging.info(
            f'Compute mode - Writing {name} to {tile_raster_output_path}')
        mode_data_array.rio.to_raster(tile_raster_output_path,
                                      dtype=np.uint8,
                                      compress='lzw')
        warpOptions = ['COMPRESS=LZW']
        warped_tile = tile_raster_output_path.replace('.tif', 'warp.tif')
        _ = gdal.Warp(warped_tile, tile_raster_output_path, warpOptions=warpOptions)
        return None

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
        output_array = self._calculate_alg(tile_array)
        logging.info('Compute alg')
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
        logging.info(f'Compute alg - Appending {name} to {tile_path}')
        mode_data_array = self._make_data_array(output_array_with_band,
                                                coords,
                                                name)

        # Write to GTiff
        tile_raster_path = tile_path.replace('.zarr', f'{name}.tif')
        mode_data_array.rio.to_raster(tile_raster_path)
        return None

    def _make_data_array(
            self,
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

    def _calculate_mode(self, tile_array: xr.DataArray,
                        classes: dict,
                        calculate_nobservations: bool = True,
                        ):
        """
        Object-oriented wrapper for mode calculation function.
        """
        return calculate_mode(tile_array,
                              classes,
                              calculate_nobservations=calculate_nobservations,
                              )

    # --------------------------------------------------------------------------
    # SKELETON FUNCTION PT. 2
    # Change function name to fit alg
    # Change "alg" out with whatever you're doing
    # --------------------------------------------------------------------------
    def _calculate_alg(self, tile_array: xr.DataArray):
        """
        Object-oriented wrapper for skeleton alg function
        """
        return calculate_alg(tile_array)
