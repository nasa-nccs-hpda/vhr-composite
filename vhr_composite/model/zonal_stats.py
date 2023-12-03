import numpy as np
from osgeo import gdal, osr
from rasterstats import zonal_stats


class ZonalStats(object):

    DEFAULT_STATS_LIST = ['min', 'max', 'mean', 'median']
    # other options: count, sum, majority, minority, unique, range, nodata

    def check_for_user_stats(stats_list):
        """
        Functions for user-defined statistics
        """
        # If user-defined function is in list,
        # remove from useStats and build addStats dict
        if 'cloud_cover' in stats_list:

            stats_list.remove('cloud_cover')
            # TODO: is this really what Maggie wants?
            add_stats = {'cloud_cover': ZonalStats.cloud_cover}

        elif 'crop_cover' in stats_list:

            stats_list.remove('crop_cover')
            # TODO: is this really what Maggie wants?
            add_stats = {'crop_cover': ZonalStats.crop_cover}

        # If not, return original statsList and None for addStats dict
        else:
            add_stats = None

        return (add_stats, stats_list)

    def percent_cover(arr, val):

        # Count the number of non-NaN (count all valid pixels)
        denom = np.count_nonzero(~np.isnan(arr))

        # Count number of cloud pixels [supposedly faster than sum]
        numer = np.count_nonzero(arr == val)

        return int(round(100 * ((numer * 1.0) / denom)))

    def cloud_cover(arr):
        return ZonalStats.percent_cover(arr, val=1)

    def crop_cover(arr):
        return ZonalStats.percent_cover(arr, val=2)

    def default_layer_dict(properties):
        output_dict = {}
        for i in range(properties['nLayers']):
            output_dict[i+1] = (f'L{i+1}', ZonalStats.DEFAULT_STATS_LIST)
        return output_dict

    # Given an input raster stack, return a dict with information needed for ZS
    def stack_properties(stack_path):

        properties = {}
        ds = gdal.Open(stack_path)
        proj = osr.SpatialReference(wkt=ds.GetProjection())

        properties['epsg'] = proj.GetAttrValue('AUTHORITY', 1)
        properties['noDataVal'] = ds.GetRasterBand(1).GetNoDataValue()
        properties['nLayers'] = ds.RasterCount

        del ds
        return properties

    @staticmethod
    def calculate_zonal_stats(
                gdf,
                raster_filename,
                layer_dict: dict = None,
                all_touched: bool = True,
                subset_cols: list = None
            ):

        raster_properties = ZonalStats.stack_properties(raster_filename)

        # If layerDict was not supplied, create it
        if layer_dict is None:
            layer_dict = ZonalStats.default_layer_dict(
                raster_properties)

        # Convert gdf to raster projection if need be
        src_feature_epsg = gdf.crs.to_epsg()
        if int(src_feature_epsg) != int(raster_properties['epsg']):
            zonal_gdf = gdf.to_crs(epsg=raster_properties['epsg'])
        else:
            zonal_gdf = gdf.copy()
        del gdf

        # Get copy of dataframe (to ignore slicing error),
        # subset columns if supplied
        if subset_cols is not None:
            use_cols = subset_cols.copy()
            use_cols.extend(['datetime', 'geometry'])
            output_gdf = zonal_gdf[use_cols].copy()
        else:
            output_gdf = zonal_gdf.copy()

        # Iterate layers and stats to add columns to the dataframe
        new_columns = []
        for layerN in layer_dict:

            layer_name = layer_dict[layerN][0]

            # Get stats from layerDict
            # Make sure it works with or without stats supplied
            try:
                stats_list = layer_dict[layerN][1]
            except IndexError:
                stats_list = ZonalStats.DEFAULT_STATS_LIST

            # Check for our user-defined stat functions:
            (add_stats, user_stats) = ZonalStats.check_for_user_stats(
                stats_list)

            # Extract zonal stats
            zs_dict = zonal_stats(
                zonal_gdf,
                raster_filename,
                all_touched=all_touched,
                add_stats=add_stats,
                stats=user_stats,
                band=layerN
            )

            # This returns a list of dictionaries, where key=stat & value=layer
            # Iterate through stats and add columns to dataframe
            for stat in stats_list:

                col_name = f'{layer_name}_{stat}'
                new_columns.append(col_name)

                output_gdf[col_name] = [d[stat] for d in zs_dict]

        del zonal_gdf

        # Remove any rows whose columns from the PQ were
        # ALL NaN (IOW don't get rid
        # of row just because one column/layer was NaN),
        # only if they all are NaN
        # outDf = outDf.dropna(how = 'all', subset = newColumns)

        # Replace all NaN with our NoData value
        if raster_properties['noDataVal']:
            output_gdf.fillna(
                raster_properties['noDataVal'],
                inplace=True
            )

        # Lastly convert back to initial projection
        if int(src_feature_epsg) != int(output_gdf.crs.to_epsg()):
            output_gdf.to_crs(epsg=src_feature_epsg, inplace=True)

        # If subsetCols was supplied, assume we only want to return a dataframe
        #  with those columns + any new ones. Most of the work has been done
        #  above, but still need to remove datetime and geometry cols
        if subset_cols:
            output_gdf.drop(
                ['datetime', 'geometry'], axis=1, inplace=True)

        return output_gdf
