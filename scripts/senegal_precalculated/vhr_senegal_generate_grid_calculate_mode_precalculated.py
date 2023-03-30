import sys
import logging
import os
import geopandas as gpd
import tqdm

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

    grid_path = '/explore/nobackup/people/mwooten3/Senegal_LCLUC/' + \
        'Shapefiles/Grid/Senegal_Grid__all.shp'

    output_dir = '/explore/nobackup/projects/3sl/data/Validation/composite/CAS'

    model_output_gdf_name = f'{region}.{model_name}.' + \
        f'{test_name}.{grid_cell_name_pre_str}.gpkg'

    composite = Composite(name=test_name,
                          grid_geopackage_path=grid_path,
                          model_output_geopackage_path=model_output_gdf_name,
                          output_dir=output_dir,
                          logger=logger)

    classes = {0: 0, 1: 1, 2: 2}

    metadataGdf = gpd.read_file(model_output_gdf_name)

    # Read in batched tile list (100 tiles per file)
    with open(sys.argv[1], 'r') as fh:
        tiles = fh.readlines()
        tiles = [tile.strip() for tile in tiles]

    for tile in tqdm.tqdm(tiles):

        len_strips = len(
            metadataGdf[metadataGdf['tile'] == tile])

        if len_strips < 1:
            continue

        name = '{}.{}'.format(grid_cell_name_pre_str, tile)

        tile_path = os.path.join(output_dir, f'{name}.zarr')

        logger.info(tile_path)

        tile_grid_dataset = composite.generate_single_grid(tile,
                                                           write_out=True)

        good, bad = soilMoistureQA(metadataGdf[metadataGdf['tile'] == tile])

        bad = bad.sort_values(by='soilM_medi')

        composite.calculate_mode_qa(tile_path=tile_path,
                                    classes=classes,
                                    passed_qa_datetimes=list(
                                        good.datetime.values),
                                    not_passed_qa_datetimes=list(
                                        bad.datetime.values),
                                    tile_dataset_input=tile_grid_dataset)


if __name__ == '__main__':
    sys.exit(main())
