import sys
import logging
import os
import geopandas as gpd
import tqdm
import numpy as np
import pandas as pd

from vhr_composite.model.composite import Composite
from vhr_composite.model.utils import TqdmLoggingHandler


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh_name = f'grid-generation{os.path.basename(sys.argv[1])}'.replace(
        '.txt', '.log')
    ch = logging.FileHandler(fh_name)
    sh = TqdmLoggingHandler()
    ch.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(sh)

    # * Set some (hardcoded for now) variables
    region = 'CAS'  # Assume for now we are doing one region at a time
    test_name = 'qaTest2'
    model_name = 'otcb.v5'
    grid_cell_name_pre_str = 'CAS.M1BS'
    start_year = 2016
    end_year = 2023  # UPPER BOUND EXCLUSIVE (LEARNED THROUGH MISTAKES)
    datetime_column = 'datetime'

    output_dir = '/explore/nobackup/projects/3sl/data/Validation/composite/CAS/'

    grid_path = '/explore/nobackup/people/mwooten3/Senegal_LCLUC/' + \
        'Shapefiles/Grid/Senegal_Grid__all.shp'
    metadataFootprints = 'CAS_M1BS_metadataGrid.gpkg'

    # Add in our landcover products and cloud mask products to the metadata
    # footprints file. Because what's the point if we don't have some damn
    # LC products to work with.
    lcDir = '/explore/nobackup/projects/3sl/development/' + \
        'cnn_landcover/normalization/otcb.v5'
    cloudDir = '/explore/nobackup/projects/3sl/products/' + \
        'cloudmask/v1/{}'.format(region)  # CHanging to explore soon

    # Get gdf with strips of interest
    metadata_gdf = gpd.read_file(metadataFootprints)

    # Some soil moisture values are NaN's, we
    # do not question why, we just fix. This is the way.
    soil_m_median = metadata_gdf['soilM_medi'].values
    soil_m_median = np.nan_to_num(soil_m_median, nan=9999.0)
    metadata_gdf['soilM_medi'] = soil_m_median

    # Set as columns in geodataframe
    metadata_gdf['landcover'] = list(map(lambda f: os.path.join(
        lcDir,
        '{}-toa.otcb.tif'.format(f)),
        metadata_gdf['strip_id']))
    metadata_gdf['cloudmask'] = list(map(lambda f: os.path.join(
        cloudDir,
        '{}-toa.cloudmask.tif'.format(f)),
        metadata_gdf['strip_id']))

    # We're using acquisition time as our unique identifier,
    # I know, I know, what happens when two sats capture at the
    # same exact time down to the nanosecond? That is between this
    # code and god. (It'll fail)
    metadata_gdf['datetime'] = \
        pd.to_datetime(metadata_gdf['acq_time'])
    unique_gpk_id = os.path.basename(sys.argv[1]).replace('.txt', '')
    model_output_gdf_name = f'{region}.{model_name}.' + \
        f'{test_name}.{grid_cell_name_pre_str}.{unique_gpk_id}.gpkg'
    metadata_gdf.to_file(model_output_gdf_name, driver="GPKG")

    composite = Composite(name=test_name,
                          grid_geopackage_path=grid_path,
                          model_output_geopackage_path=model_output_gdf_name,
                          output_dir=output_dir,
                          logger=logger)

    # This makes sense, don't question it (:
    classes = {0: 0, 1: 1, 2: 2}

    metadata_gdf = gpd.read_file(model_output_gdf_name)

    # Filter the strips to use in mode and bad data filling to be within
    # the desired "epoch" or year range of the acquisition time
    datetime_mask = (metadata_gdf[datetime_column] > str(start_year)) & \
        (metadata_gdf[datetime_column] < str(end_year))
    metadata_gdf_filtered = metadata_gdf[datetime_mask]

    # Read in batched tile list (100 tiles per file)
    # most regions usually occupy just 10 or so of these
    # 100-tile text file lists.
    with open(sys.argv[1], 'r') as fh:
        tiles = fh.readlines()
        tiles = [tile.strip() for tile in tiles]

    # Making this multi-threaded, will it mess with the Numba
    # parallel mode calculations???
    for tile in tqdm.tqdm(tiles):

        len_strips = len(
            metadata_gdf[metadata_gdf['tile'] == tile])

        logger.info(f'Number of strips in {tile}: {len_strips}')

        if len_strips < 1:
            continue

        name = '{}.{}'.format(grid_cell_name_pre_str, tile)

        tile_path = os.path.join(output_dir, f'{name}.zarr')

        logger.info(tile_path)

        mode_name = f'.{start_year}.{end_year}.mode'

        tile_raster_path = tile_path.replace('.zarr', f'{mode_name}.tif')
        logger.info(tile_raster_path)

        # Don't do more work than we have to. Woo!
        if os.path.exists(tile_raster_path):
            logger.info(f'{tile_raster_path} already exists.')
            continue

        tile_grid_dataset = composite.generate_single_grid(tile,
                                                           write_out=True)

        metadata_per_tile_filtered = \
            metadata_gdf_filtered[metadata_gdf_filtered['tile'] == tile]

        len_filtered_strips = len(metadata_per_tile_filtered)
        logger.info(
            f'Number of filtered strips in {tile}: {len_filtered_strips}')
        if len_filtered_strips < 1:
            continue

        composite.calculate_mode(tile_path=tile_path,
                                 tile_raster_output_path=tile_raster_path,
                                 classes=classes,
                                 rows_to_use=list(
                                     metadata_per_tile_filtered.datetime.values),
                                 tile_dataset_input=tile_grid_dataset)


if __name__ == '__main__':
    sys.exit(main())
