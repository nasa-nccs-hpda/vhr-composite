import sys
import logging
import os
import pandas as pd
import geopandas as gpd
import numpy as np

from vhr_composite.model.composite import Composite
from vhr_composite.model.utils import TqdmLoggingHandler


def soilMoistureQA(tileDF):
    goodSoilMoisture = tileDF['soilM_medi'] < 1800
    badSoilMoisture = tileDF['soilM_medi'] >= 1800
    goodDF = tileDF[goodSoilMoisture]
    badDF = tileDF[badSoilMoisture]
    return goodDF, badDF


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler('grid-generation.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(TqdmLoggingHandler())
    # * Set some (hardcoded for now) variables
    region = 'CAS'  # Assume for now we are doing one region at a time
    testName = 'qaTest2'
    region = 'CAS'
    modelName = 'otcb.v5'
    grid_cell_name_pre_str = 'CAS.M1BS.'

    metadataFootprints = '/explore/nobackup/people/mwooten3/' + \
        'Senegal_LCLUC/Compositing/Grids/CAS_M1BS_metadataGrid.shp'

    # * Using 8-bit outputs I made for now as I no longer have access to older
    # output directory - eventually use the .cog files?
    lcDir = '/explore/nobackup/projects/3sl/development/' + \
        'cnn_landcover/normalization/otcb.v5'
    cloudDir = '/explore/nobackup/projects/3sl/products/' + \
        'cloudmask/v1/{}'.format(region)  # CHanging to explore soon

    grid_path = '/explore/nobackup/people/mwooten3/Senegal_LCLUC/' + \
        'Shapefiles/Grid/Senegal_Grid__all.shp'

    # Get gdf with strips of interest
    metadataGdf = gpd.read_file(metadataFootprints)
    soil_m_median = metadataGdf['soilM_medi'].values
    soil_m_median = np.nan_to_num(soil_m_median, nan=9999.0)
    metadataGdf['soilM_medi'] = soil_m_median

    # Set as columns in geodataframe
    metadataGdf['landcover'] = list(map(lambda f: os.path.join(
        lcDir,
        '{}-toa.otcb.tif'.format(f)),
        metadataGdf['strip_id']))
    metadataGdf['cloudmask'] = list(map(lambda f: os.path.join(
        cloudDir,
        '{}-toa.cloudmask.tif'.format(f)),
        metadataGdf['strip_id']))
    metadataGdf['datetime'] = pd.to_datetime(metadataGdf['acq_time'])

    tile_grps = metadataGdf.groupby(by='tile')
    df_by_tile = [df_tile_combo for df_tile_combo in tile_grps]
    tiles = {}
    for i in range(len(df_by_tile)):
        tiles[df_by_tile[i][0]] = df_by_tile[i][1]
    output_dir = '.'
    model_output_gdf_name = f'{region}.{modelName}.' + \
        f'{testName}.{grid_cell_name_pre_str}gpkg'
    metadataGdf.to_file(model_output_gdf_name, driver="GPKG")

    composite = Composite(name=testName,
                          grid_geopackage_path=grid_path,
                          model_output_geopackage_path=model_output_gdf_name,
                          output_dir=output_dir,
                          logger=logger)
    composite.generate_grid(
        tile_list=tiles, calculate_mode=False, classes={0: 0, 1: 1, 2: 2})


if __name__ == '__main__':
    sys.exit(main())
