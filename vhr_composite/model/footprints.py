import os
import logging
import rasterio
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
from glob import glob
from shapely.geometry import box
import xml.etree.ElementTree as ET


class Footprints(object):

    SST_RENAME_MAP = {
        'MEANSUNEL': 'strp_sun_elev',
        'MEANSUNAZ': 'strp_sun_az',
        'MEANSATEL': 'strp_sat_elev',
        'MEANSATAZ': 'strp_sat_az',
        'MEANOFFNADIRVIEWANGLE': 'strp_off_nadir'
    }
    # 'SUNZEN': 'strp_sun_zen',
    # 'SATZEN': 'strp_sat_zen'}

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(
                self,
                conf: dict,
                input_data: str,
                output_filename: str
            ):

        # Configuration file intialization
        self.conf = conf

        # create output directory
        os.makedirs(self.conf.output_dir, exist_ok=True)

        # set input object
        self.input_data = input_data

        # set output_filename
        self.output_filename = output_filename

        # set base fields
        # TODO: consider adding as global variable from class
        self.base_fields = [
            'xml_path', 'strip_id', 'sensor',
            'spec_type', 'catalog_id', 'date',
            'year', 'month', 'day', 'acq_time'
        ]

    # -------------------------------------------------------------------------
    # build_footprints
    # -------------------------------------------------------------------------
    def build_footprints(self):

        logging.info('Building footprints')

        # Build footprints base
        fp_gdf = self._build_footprints_base()

        # Add basefields to existing file
        fp_gdf = self._add_base_fields(fp_gdf)

        # Add geometry info from strip
        fp_gdf = self._add_sst_geometry_fields(fp_gdf)

        # Add region column if available in config file
        if 'region' in self.conf.keys():
            fp_gdf = fp_gdf.assign(region=self.conf.region)

        # Finally, write dataframe to disk
        fp_gdf.to_file(self.output_filename)
        logging.info(f'Updated {self.output_filename}.')

        return

    def _build_footprints_base(self):

        # Get base footprints .shp and return a geodataframe

        # _get_input_tifs will accept diff types of inputs
        input_tifs = self._get_input_tifs(self.input_data)
        logging.info(f'Found {len(input_tifs)} tifs to process.')

        # get column field for input data
        input_column_field = self.conf.input_column_field

        # gdf from raster list (replaces gdaltindex)
        gdf = self._gdf_from_raster_list(
            input_tifs, input_column_field, self.output_filename)

        return gdf

    def _get_input_tifs(self, input_passed: str):

        # Check if input is text file and read .tifs into list
        if input_passed.endswith('.txt'):
            with open(input_passed) as f:
                lines = [line.strip() for line in f.readlines()]
                input_tifs = [line for line in lines if line.endswith('.tif')]

        # Check if input is a directory path
        elif os.path.isdir(input_passed):
            input_tifs = glob(os.path.join(input_passed, '*tif'))

        # Check if input is a regex
        elif '*' in input_passed:
            input_tifs = glob(input_passed)

        else:
            raise RuntimeError(
                f'Input {input_passed} not recognized/does not exist.')

        if len(input_tifs) < 1:
            raise RuntimeError(
                f'No .tif files found for input {input_passed}')

        return input_tifs

    def _gdf_from_raster_list(
                self,
                input_tifs: list,
                gdf_column: str,
                output_filename: str
            ):

        # use rasterio to replace gdaltindex
        gdf_list = []
        for filename in input_tifs:

            # read raster
            raster = rxr.open_rasterio(filename)

            # get raster bounds
            bounds = raster.rio.bounds()

            # generate geodataframe from raster information
            raster_gdf = gpd.GeoDataFrame({
                gdf_column: [filename],
                'geometry': [
                        box(bounds[0], bounds[1], bounds[2], bounds[3])],
                }, crs=str(raster.rio.crs)
            )

            # if the CRS is different, we force them to be the same
            if str(raster_gdf.crs) != str(self.conf.epsg):
                raster_gdf = raster_gdf.to_crs(self.conf.epsg)

            # append to the list to concatenate
            gdf_list.append(raster_gdf)

        # generate dataframe
        gdf = gpd.GeoDataFrame(
            pd.concat(gdf_list, axis=0).reset_index(drop=True),
            crs=self.conf.epsg)

        # output to disk
        gdf.to_file(output_filename)
        logging.info(f'Saved {output_filename}.')
        return gdf

    def _add_base_fields(self, gdf):

        logging.info(f'Adding base fields to {self.output_filename}')

        # Assumes toa .tif named e.g. WV02_20190327_M1BS_109002003AE400-toa.tif

        # get column field for input data
        input_column_field = self.conf.input_column_field

        # stupid way to iterate through fields
        # TODO: change in the future to switch statement with Python3.10
        # or with dictionary that calls out the function
        for base_field in self.base_fields:

            logging.info(f'Adding {base_field}')

            if base_field == 'xml_path':
                gdf[base_field] = list(
                    map(
                        (lambda f: f.replace('.tif', '.xml')),
                        gdf[input_column_field]))

            elif base_field == 'strip_id':
                gdf[base_field] = list(
                    map((lambda f: os.path.basename(f).strip('-toa.tif')),
                        gdf[input_column_field]))

            elif base_field == 'sensor':
                gdf[base_field] = list(
                    map((lambda f: f[0:4]), gdf['strip_id']))

            elif base_field == 'spec_type':
                gdf[base_field] = list(
                    map((lambda f: f.split('_')[2]), gdf['strip_id']))

            elif base_field == 'catalog_id':
                gdf[base_field] = list(
                    map((lambda f: f.split('_')[3]), gdf['strip_id']))

            elif base_field == 'date':
                gdf[base_field] = list(
                    map((lambda f: f.split('_')[1]), gdf['strip_id']))

            elif base_field == 'year':
                gdf[base_field] = list(map((lambda f: f[0:4]), gdf['date']))

            elif base_field == 'month':
                gdf[base_field] = list(map((lambda f: f[4:6]), gdf['date']))

            elif base_field == 'day':
                gdf[base_field] = list(map((lambda f: f[6:8]), gdf['date']))

            elif base_field == 'acq_time':
                gdf[base_field] = list(
                    map(self._get_acq_time, gdf['xml_path']))

            else:
                print(f'No field map for {base_field} exists yet. Skipping')

        # save to output filename
        gdf.to_file(self.output_filename)

        return gdf

    def _extract_from_xml(self, xml, attr_name):
        tree = ET.parse(xml)
        imgTag = tree.getroot().find('IMD').find('IMAGE')
        return imgTag.find(attr_name).text

    def _get_acq_time(self, xml):
        return self._extract_from_xml(xml, 'FIRSTLINETIME')

    def _add_sst_geometry_fields(self, gdf):

        logging.info(
            f'Adding acquisition geom from data to {self.output_filename}')

        # Add sun/satellite azimuth, elevation, and off-nadir
        for tag, name in self.SST_RENAME_MAP.items():

            gdf[name] = list(
                map(lambda x: float(self._extract_from_xml(x, tag)),
                    gdf['xml_path']))

        # Add sun/sat zenith (90 - elevation)
        gdf['strp_sun_zen'] = 90.0 - gdf['strp_sun_elev'].astype(float)
        gdf['strp_sat_zen'] = 90.0 - gdf['strp_sat_elev'].astype(float)

        # save to output filename
        gdf.to_file(self.output_filename)
        logging.info(f'Updated {self.output_filename}.')

        return gdf
