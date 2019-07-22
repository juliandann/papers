import numpy as np
import pandas as pd
import geopy.distance
import xarray as xr
import time
from scipy.spatial.distance import cdist

def dist(lat1, long1, lat2, long2):
    """Planar geometric distance function

    Parameters
    ----------
    lat1 : float
        latitude
    long1 : float
        longitude
    lat2 : float
        latitude coordinate two.
    long2 : float
        Longitude coordinate two.

    Returns
    -------
    type
        Description of returned object.

    """

    return np.sqrt((lat1-lat2)**2.0+ (long1-long2)**2.0)

def find_SAR_pixel(df,lat1,long1, lat2, long2,export_columns):
    """function that grabs the closest dataframe measurement using dist() function

    Parameters
    ----------
    df : Pandas Dataframe
        Description of parameter `df`.
    lat1 : string
        column within df for latitude
    long1 : string
        column within df for longitude
    lat2 : Float
        Value for latitude
    long2 : float
        Value for longitude
    export_columns : string
        Names of columns to put into the original dataframe that correspond to the nearest value.

    Returns
    -------
    Dataframe
        Updated dataframe with columns from the nearest value.

    """
    distances = df.apply(
        lambda row: dist(lat, long, row[lat1], row[lat2]),
        axis=1)
    return df.loc[distances.idxmin(), export_columns]

def closest_point_index_orig(point, points):
    """ Find closest point from a list of points. """
    return cdist([point], points).argmin()

def closest_point_index(df1,df2,lat1,lon1,lat2,lon2,*export_columns,prefix='df2'):
    """Grab the closest geodesic point from a list using scipy

    Parameters
    ----------
    point : float
        Can be a list of point pairs or a single pair
    points : float
        array of point pairs

    Returns
    -------
    index value(s)
        index values from the points array

    """

    print('Making tuples')
    #make tuple of lat lon from each dataframe
    df1['point'] = [(x, y) for x,y in zip(df1[lat1], df1[lon1])]
    df2['point'] = [(x, y) for x,y in zip(df2[lat2], df2[lon2])]
    print('Calculating nearest point')
    #calculate the closest point index value between the two arrays coordinates
    index = [cdist([x], list(df2['point'])).argmin() for x in df1['point']]

    #adding new columns to df1
    for column in export_columns:
        df1[prefix+'_'+column] = df2[column].iloc[index].values

    return df1

def csv_combined_file_maker(files,combined_filepath='combined_file.csv'):
    """Function that combines a list of csv files into one csv file that can be imported easily

    Parameters
    ----------
    files : list of strings
        The filepaths of the files you wish to combine. They must all have the same column names
    combined_filepath : string
        Filepath for the combined file.

    Returns
    -------
    csv
        Creates a CSV in the directory python is running in unless combined filepath is specified

    """

    list = []
    for i in files:
        df = pd.read_csv(i,sep=',')
        list.append(df)

    df = pd.concat(list,ignore_index=True)
    df.to_csv(combined_filepath,sep=',')

def hydrosense_data_cruncher(df,site,sar_plot):
    pass

def distancer(lat1,lon1,lat2,lon2):
    """Function meant to be used in df.apply() in order to compute distance in WGS84 coordinate system

    Parameters
    ----------
    lat1 : float
        latitude for pair one.
    lon1 : float
        Longitude for pair one.
    lat2 : float
        Latitude for pair two
    lon2 : float
        Longitude for pair two.

    Returns
    -------
    float
        Distance in meters

    """
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return geopy.distance.VincentyDistance(coords_1, coords_2).m

def distance_calc(df,lat1,lon1,lat2,lon2,new_col_name):
    """Function used in conjunction with distancer to find distance in meters for columns of pandas dataframes that are in WGS84.

    Parameters
    ----------
    df : Pandas Dataframe
        Contains pairs of lat/long
    lat1 : float
        latitude for pair one.
    lon1 : float
        Longitude for pair one.
    lat2 : float
        Latitude for pair two
    lon2 : float
        Longitude for pair two.
    new_col_name : string
        The column name that will be added to the original dataframe

    Returns
    -------
    Pandas Dataframe
        Dataframe with a new column with 'new_col_name' that has distance in meters between the two coordinates.




    """

    df[new_col_name] = df.apply(lambda x: distancer(x[lat1],x[lon1],x[lat2],x[lon2]), axis=1)
    return df

def average_SM_at_pixel(df,pixel_index):

    #find unique pixel index values
    index = np.unique(df[pixel_index].values)

    #groupby pixel index and measurement depth
    pixel_groups = df1.groupby([pixel_index,'VWC_Measurement_Depth','SAR_Plot'])

    #average and std of VWC values
    pixel_groups['VWC'].mean()
    pixel_groups['VWC'].std()

    pixel_groups['above_0.06'].mean()
    pixel_groups['above_0.06'].std()
