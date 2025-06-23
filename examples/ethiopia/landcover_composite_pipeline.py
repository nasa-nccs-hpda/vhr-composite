import os
import sys
import time
import tqdm
import logging
import argparse
import omegaconf
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from pathlib import Path
from vhr_composite.model.composite import Composite
from vhr_composite.model.footprints import Footprints
from vhr_composite.model.metadata import Metadata
from vhr_composite.model.utils import TqdmLoggingHandler
from vhr_composite.model import post_process


class LandCoverCompositePipeline(object):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config_filename: str):

        # Configuration file intialization
        self.conf = omegaconf.OmegaConf.load(config_filename)

        # create output directory
        os.makedirs(self.conf.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # _set_logger
    # -------------------------------------------------------------------------
    def _set_logger(self, description: str = ''):
        """
        Set logger configuration.
        """
        # setup logger configuration
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh_name = os.path.join(
            self.conf.output_dir,
            f'{description}.log')

        # set file handler logger configuration
        ch = logging.FileHandler(fh_name)
        sh = TqdmLoggingHandler()
        ch.setLevel(logging.INFO)
        sh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s; %(levelname)s; %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        sh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(sh)
        return logger

    # -------------------------------------------------------------------------
    # build_footprints
    # -------------------------------------------------------------------------
    def build_footprints(self) -> None:

        # set logger
        self._set_logger('build_footprints')

        # create object for building footprint
        footprints = Footprints(
            self.conf,
            input_data=self.conf.input_data_regex,
            output_filename=self.conf.footprints_filename,
            grid_filename=self.conf.grid_path
        )

        # start the generation of footprints
        footprints.build_footprints()

        return

    # -------------------------------------------------------------------------
    # extract_metadata
    # -------------------------------------------------------------------------
    def extract_metadata(self) -> None:

        # set logger
        self._set_logger('extract_metadata')

        # create object for building footprint
        metadata = Metadata(
            self.conf,
            input_filename=self.conf.footprints_filename,
            output_filename=self.conf.metadata_filename
        )

        # start the generation of metadata
        metadata.extract_metadata()

        return

    # -------------------------------------------------------------------------
    # composite
    # -------------------------------------------------------------------------
    def composite(self, tiles_filename: str) -> None:

        # set logger
        logger = self._set_logger(
            f'composite-{Path(tiles_filename).stem}')

        # make sure tiles_filename exists
        assert os.path.exists(tiles_filename), \
            f'{tiles_filename} does not exist.'

        # Get gdf with strips of interest
        metadata_gdf = gpd.read_file(self.conf.metadata_filename)
        logging.info(f'Reading in metadata {self.conf.metadata_filename}')

        # Some soil moisture values are NaN's, we
        # do not question why, we just fix. This is the way.
        if 'soilM_median' in metadata_gdf.columns:
            soil_m_median = metadata_gdf['soilM_median'].values
            soil_m_median = np.nan_to_num(soil_m_median, nan=9999.0)
            metadata_gdf['soilM_median'] = soil_m_median
            logging.info('Fixing soil moisture data for NaNs.')
        else:
            # adding all scenes as good since we do not have soil moisture data
            metadata_gdf['soilM_median'] = 2799

        # Set landcover outputs as columns in geodataframe
        logging.info('Adding landcover column to metadata.')
        if 'landcover' in self.conf.composite_fields:
            metadata_gdf['landcover'] = list(
                map(lambda f: os.path.join(
                    self.conf.landcover_dir,
                    f'{f}{self.conf.landcover_suffix}.tif'
                ), metadata_gdf['strip_id'])
            )
        else:
            metadata_gdf['landcover'] = np.nan

        # Set cloudmask outputs as columns in geodataframe
        logging.info('Adding cloudmask column to metadata.')
        if 'cloudmask' in self.conf.composite_fields:
            metadata_gdf['cloudmask'] = list(
                map(lambda f: os.path.join(
                    self.conf.cloudmask_dir,
                    f'{f}{self.conf.cloudmask_suffix}.tif'
                ), metadata_gdf['strip_id'])
            )
        else:
            metadata_gdf['cloudmask'] = np.nan

        # We're using acquisition time as our unique identifier,
        # I know, I know, what happens when two sats capture at the
        # same exact time down to the nanosecond? That is between this
        # code and God. (It'll fail)
        metadata_gdf['datetime'] = \
            pd.to_datetime(metadata_gdf['acq_time'])
        unique_gpk_id = Path(tiles_filename).stem
        logging.info(f'Unique gpkg id: {unique_gpk_id}')

        # Store model output gpkg into disk
        model_output_gdf_name = os.path.join(
            self.conf.output_dir,
            f'{self.conf.region}-{self.conf.model_name}-' +
            f'{self.conf.test_name}-{self.conf.grid_cell_name_pre_str}-' +
            f'{unique_gpk_id}.gpkg'
        )
        metadata_gdf.to_file(model_output_gdf_name, driver="GPKG")
        logging.info(f'Saved updated metadata file to {model_output_gdf_name}')

        # Initialize composite class
        composite = Composite(
            name=self.conf.test_name,
            grid_geopackage_path=self.conf.grid_path,
            model_output_geopackage_path=model_output_gdf_name,
            output_dir=self.conf.output_dir,
            logger=logger
        )

        # read clean metadata file
        metadata_gdf = gpd.read_file(model_output_gdf_name)
        logging.info(f'Metadata includes {metadata_gdf.shape[0]} strips.')
        logging.info(f'Years in metadata: {metadata_gdf.year.unique()}')

        # Filter the strips to use in mode and bad data filling to be within
        # the desired "epoch" or year range of the acquisition time
        datetime_mask = \
            (
                metadata_gdf[self.conf.datetime_column] >
                str(self.conf.start_year)) & \
            (metadata_gdf[self.conf.datetime_column] < str(self.conf.end_year))
        metadata_gdf_filtered = metadata_gdf[datetime_mask]
        logging.info(
            f'Filtered strips by date: {metadata_gdf_filtered.shape[0]}.')
        logging.info(f'Year range: {metadata_gdf_filtered.year.unique()}')

        # Read in batched tile list (100 tiles per file)
        # most regions usually occupy just 10 or so of these
        # 100-tile text file lists.
        with open(tiles_filename, 'r') as fh:
            tiles = fh.readlines()
            tiles = [tile.strip() for tile in tiles]
        assert len(tiles) > 0, f'No tiles found in {tiles_filename}'
        logging.info(
            f'Tiles provided in {Path(tiles_filename).stem}: {len(tiles)}')

        # Making this multi-threaded, will it mess with the Numba
        # parallel mode calculations???
        for tile in tqdm.tqdm(tiles):

            """


            # only take metadata for the specific tile
            metadata_per_tile_filtered = \
                metadata_gdf_filtered[metadata_gdf_filtered['tile'] == tile]

            # take the number of remaining filtered strips
            len_filtered_strips = len(metadata_per_tile_filtered)
            logger.info(f'Filtered strips in {tile}: {len_filtered_strips}')

            # if we end up with no strips, continue to next tile
            # TODO: do we really want this?
            if len_filtered_strips < 1:
                continue

            print(metadata_per_tile_filtered.toa_path.values)

            # perform composite computation
            composite.calculate_mode(
                tile_path=output_tile_filename,
                tile_raster_output_path=output_raster_filename,
                classes=self.conf.composite_classes,
                rows_to_use=list(
                    metadata_per_tile_filtered.datetime.values),
                tile_dataset_input=tile_grid_dataset
            )
            """
            # get the number of strips per tile
            len_strips = len(
                metadata_gdf[metadata_gdf['tile'] == tile])
            logger.info(f'Number of strips in {tile}: {len_strips}')

            if len_strips < 1:
                continue

            # name of the output file
            output_name = f'{self.conf.grid_cell_name_pre_str}.{tile}'

            # set the output filename
            output_tile_filename = os.path.join(
                self.conf.output_dir, f'{output_name}.zarr')
            logger.info(f'Processing {output_tile_filename}')

            # output of mode
            mode_name = \
                f'.{self.conf.start_year}.{self.conf.end_year}' + \
                f'.{self.conf.region}.mode'

            output_raster_filename = output_tile_filename.replace(
                '.zarr', f'{mode_name}.QAD.tif')
            logger.info(f'Storing output on {output_raster_filename}')

            # Don't do more work than we have to. Woo!
            if os.path.exists(output_raster_filename):
                logger.info(f'{output_raster_filename} already exists.')
                continue

            # TODO: what is this?
            tile_grid_dataset = composite.generate_single_grid(
                tile,  write_out=True, grid_cell_name_pre_str='Amhara.M1BS')

            if not tile_grid_dataset:
                continue

            tile_grid_data_array = tile_grid_dataset[output_name].astype(
                np.uint32)
            logger.info(tile_grid_data_array.data.max())
            logger.info(tile_grid_data_array.data.min())

            #print(tile_grid_data_array)

            metadata_per_tile_filtered = \
                metadata_gdf_filtered[metadata_gdf_filtered['tile'] == tile]

            len_filtered_strips = len(metadata_per_tile_filtered)
            logger.info(
                f'Number of filtered strips in {tile}: {len_filtered_strips}')
            if len_filtered_strips < 1:
                continue

            # Use the updated GDF to further filter by soil moisture QA
            good, bad = self.soilMoistureQA(metadata_per_tile_filtered)

            # When filling wholes we want to start with the "best" of the bad
            # i.e. the lowest soil moisture first
            bad = bad.sort_values(by='soilM_median')

            passed_qa_datetimes = list(good.datetime.values)
            not_passed_qa_datetimes = list(bad.datetime.values)
            tile_grid_data_array.sel(time=passed_qa_datetimes)
            tile_grid_data_array.sel(time=not_passed_qa_datetimes)

            logger.info(len(passed_qa_datetimes))
            logger.info(len(not_passed_qa_datetimes))

            tile_grid_ds_good = tile_grid_data_array.sel(
                time=passed_qa_datetimes)
            tile_grid_ds_bad = tile_grid_data_array.sel(
                time=not_passed_qa_datetimes)

            nodata_value = np.uint32(10)

            output_raster_filename = Path(output_raster_filename)

            logger.info('Reducing with multi-mode')

            reduced_stack = Composite.reduce_stack('multi_mode',
                                                tile_grid_ds_good,
                                                output_raster_filename,
                                                overwrite=False,
                                                nodata=nodata_value,
                                                gpu=True,
                                                logger=logger)

            logger.info('Filling holes with nodata')

            reduced_stack_hole_filled = post_process.fill_holes(
                tile_grid_ds_bad,
                reduced_stack,
                not_passed_qa_datetimes,
                nodata_value,
                logger)

            logger.info(f'Writing to zarr: {str(output_raster_filename)}')
            reduced_stack_hole_filled.rio.to_raster(str(output_raster_filename),
                                                    dtype=np.uint32,
                                                    compress='lzw')
            warpOptions = ['COMPRESS=LZW']
            warped_tile = str(output_raster_filename).replace('.tif', 'warp.tif')
            logger.info(f'Writing warped {warped_tile}')
            _ = gdal.Warp(warped_tile,
                        str(output_raster_filename),
                        warpOptions=warpOptions)
        return

    def soilMoistureQA(self, tileDF):
        goodSoilMoisture = tileDF['soilM_median'] < 2800
        badSoilMoisture = tileDF['soilM_median'] >= 2800
        goodDF = tileDF[goodSoilMoisture]
        badDF = tileDF[badSoilMoisture]
        badDF = badDF.sort_values(by='soilM_median')
        return goodDF, badDF


# -----------------------------------------------------------------------------
# main
#
# python landcover_composite_pipeline_cli.py -c config.yaml
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to generate Senegal composite.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='config_file',
                        help='Path to the configuration file (YAML)')
    parser.add_argument('-t',
                        '--tiles-filename',
                        dest='tiles_filename',
                        type=str,
                        required=False,
                        help='Filename with tiles to process')
    parser.add_argument(
                        '-s',
                        '--step',
                        type=str,
                        nargs='*',
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        default=[
                            'build_footprints',
                            'extract_metadata',
                            'composite'],
                        choices=[
                            'build_footprints',
                            'extract_metadata',
                            'composite'])
    args = parser.parse_args()

    # Setup timer to monitor script execution time
    timer = time.time()

    # setup pipeline object
    pipeline = LandCoverCompositePipeline(args.config_file)

    # Regression CHM pipeline steps
    if "build_footprints" in args.pipeline_step:
        pipeline.build_footprints()
    if "extract_metadata" in args.pipeline_step:
        pipeline.extract_metadata()
    if "composite" in args.pipeline_step:
        pipeline.composite(args.tiles_filename)

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())

# singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/vhr-composite-jordan-edits:/explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/tensorflow-caney-2023.05 python landcover_composite_pipeline.py -c composite_ethiopia_epoch1.yaml -t test_tile_0_test.txt -s composite