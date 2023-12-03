import os
import shutil
import logging
import pandas as pd
import dask_geopandas
import geopandas as gpd
import dask.dataframe as ddf
from vhr_composite.model.zonal_stats import ZonalStats


class Metadata(object):

    METADATA_COLUMN_NAMES = {
        'soil_moisture': 'soilM',
        'chirps_precip': 'chirps',
        'worldclim_temp': 'wcTemp',
        'worldclim_precip': 'wcPrecip',
        'ecoregion': 'ecoregion',
        'data_density': 'dataDensity',
        'crop_cover': 'pct_cropCo',
        'cloud_cover': 'pct_cloud',
    }

    MONTHLY_SOIL_MOISTURE_FMT = '%y%m'

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(
                self,
                conf: dict,
                input_filename: str,
                output_filename: str
            ):

        # Configuration file intialization
        self.conf = conf

        # create output directory
        os.makedirs(self.conf.output_dir, exist_ok=True)

        # set input object
        self.input_filename = input_filename

        # set output_filename
        self.output_filename = output_filename

        # read footprints dataframe
        self.fp_gdf = gpd.read_file(self.input_filename)

        # set datetime columns for zonal stats
        self.fp_gdf['datetime'] = pd.to_datetime(
            self.fp_gdf[self.conf.date_column_field],
            format=self.conf.date_column_field_format, exact=False
        )

        print(self.fp_gdf.columns)

    # -------------------------------------------------------------------------
    # extract_metadata
    # -------------------------------------------------------------------------
    def extract_metadata(self):

        # set all_touched field
        all_touched = False
        if 'all_touched' in self.conf.metadata_fields:
            all_touched = True

        # TODO: TEMP until Jordan fixes cloud mask
        self.fp_gdf = self.fp_gdf[
            self.fp_gdf['strip_id'] != 'WV02_20110223_M1BS_1030010009935100']

        # List to hold outputs from zonal stats, TODO: maybe remove
        output_gdfs_list = [self.fp_gdf]

        # Metadata: soil_moisture
        if 'soil_moisture' in self.conf.metadata_fields:
            sm = self.get_monthly_soil_moisture(
                stats=self.conf.stats_list,
                all_touched=all_touched,
                subset_cols=self.conf.join_cols
            )
            output_gdfs_list.append(sm)

        # print(sm)

        """
        # Metadata: 
        if args['chirpsPrecip']:
            print("\n  Getting CHIRPS precip...")
            mp = footprints.getMonthlyPrecip(stats = statsList, 
                            allTouched = allTouched, subsetCols = joinCols.copy())
            outDfs.append(mp)
            #writeShapefile(p, 'test/testZS-chirps.shp')

        if args['worldclimTemp']:
            print("\n  Getting WorldClim temp...")
            wt = footprints.getWorldClimTemp(stats = statsList, 
                            allTouched = allTouched, subsetCols = joinCols.copy())
            outDfs.append(wt)
            #writeShapefile(wt, 'test/testZS-wct.shp')

        if args['worldclimPrecip']:
            print("\n  Getting WorldClim precip...")
            wp = footprints.getWorldClimPrecip(stats = statsList, 
                            allTouched = allTouched, subsetCols = joinCols.copy())
            outDfs.append(wp)
            #writeShapefile(wp, 'test/TAPPAN-testZS-wcp.shp')

        if args['cropCover']:
            print("\n  Getting percent crop cover...")
            cc = footprints.getPercentCropCover(stats = ['cropCover'], 
                            allTouched = allTouched, subsetCols = joinCols.copy())

            outDfs.append(cc)
            #writeShapefile(cc, 'test/TAPPAN-testZS-cc.shp')

        if args['dataDensity']:
            print("\n  Getting data density...")
            dd = footprints.getDataDensity(stats = statsList, 
                            allTouched = allTouched, subsetCols = joinCols.copy())

            outDfs.append(dd)
            #writeShapefile(dd, 'test/TAPPAN-testZS-datadensity.shp')

        if args['ecoregion']:
            print("\n  Getting ecoregion...")
            er = footprints.getEcoregion(stats = ['majority'], 
                            allTouched = allTouched, subsetCols = joinCols.copy())
            
            outDfs.append(er)
            #writeShapefile(er, 'test/TAPPAN-testZS-ecoregion.shp')
        
        if args['sceneSstGeometry']:
            s = time.time()
            print("\n  Getting scene acquisition geometry...")
        #sg = footprints.getSceneSstGeom(useDask = useDask, nPartitions = nPartitions,
            #                                          subsetCols = joinCols.copy())
            sg = footprints.getSceneSstGeom(subsetCols = joinCols.copy())       
            e = time.time()
            print("\n   Done with scene acquisition geometry")
            print("   ", calculateElapsedTime(s, e, unit = 'seconds'))
            
            outDfs.append(sg)

        if args['cloudCover']:
            s = time.time()
            print("\n  Getting percent cloud cover...")
        #sg = footprints.getSceneSstGeom(useDask = useDask, nPartitions = nPartitions,
            #                                          subsetCols = joinCols.copy())
            pc = footprints.getPercentCloudCover(subsetCols = joinCols.copy())       
            e = time.time()
            print("\n   Done with percent cloud cover")
            print("   ", calculateElapsedTime(s, e, unit = 'seconds'))

            outDfs.append(pc)
        #import pdb; pdb.set_trace()
        """

        print(output_gdfs_list)

        dask_gdfs = [
            dask_geopandas.from_geopandas(gdf, npartitions=1)
            for gdf in output_gdfs_list]
        merged_dask_gdfs = ddf.concat(dask_gdfs, axis=1)
        output_gdf = gpd.GeoDataFrame(
            merged_dask_gdfs.compute(), geometry=self.fp_gdf['geometry'])
        output_gdf = output_gdf.loc[
            :, ~output_gdf.columns.duplicated()].copy()

        output_gdf.to_file(self.output_filename)

        return output_gdf

    def get_monthly_soil_moisture(
                self,
                stats: list = ['median'],
                all_touched: bool = True,
                subset_cols: list = ['tile', 'strip_id']
            ):

        # get date var format
        sm_date_format = self.MONTHLY_SOIL_MOISTURE_FMT

        # SM filenames wa_monthly_fldas_soilmoi00_10cm_tavg_DATE.tif
        sm_filename_path = self.conf.monthly_soil_moisture_filename

        # set layer dictionary
        layer_dict = {1: (self.METADATA_COLUMN_NAMES['soil_moisture'], stats)}

        # Get list of unique dates in the SM date format
        unique_dates = self.fp_gdf['datetime'].dt.strftime(
            sm_date_format).unique()

        # Iterate through month/year combos, get raster, run zs and build gdf
        output_df_list = []
        for ud in unique_dates:

            # get sub_df
            sub_gdf = self.fp_gdf[
                self.fp_gdf['datetime'].dt.strftime(sm_date_format) == ud]

            # Input raster
            sm_filename = self._find_unzip_file(
                sm_filename_path.replace('<date>', ud))

            output_df_list.append(
                ZonalStats.calculate_zonal_stats(
                    sub_gdf,
                    sm_filename,
                    layer_dict,
                    subset_cols=subset_cols,
                    all_touched=all_touched)
            )

        return pd.concat(output_df_list, ignore_index=True)

    def _find_unzip_file(self, input_tif):

        if os.path.isfile(input_tif):
            return input_tif

        # If .tif does not exist, look for .zip and unzip
        else:

            in_zip = input_tif.replace('.tif', '.zip')

            if not os.path.isfile(in_zip):
                logging.info(f"Neither {input_tif} nor {in_zip} exists.")
                return

            shutil.unpack_archive(in_zip, os.path.dirname(in_zip))

            # Check to be sure unzip worked
            if not os.path.isfile(input_tif):
                logging.info(
                    f"Expected input {input_tif} still does not exist.")
                return

            return input_tif
