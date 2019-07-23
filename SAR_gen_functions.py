import numpy as np
import pandas as pd
import geopy.distance
import xarray as xr
import time
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

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

def average_SM_at_pixel(df,pixel_index,savename='pixel_hydro_vs_above.png'):

    #groupby pixel index and measurement depth
    pixel_group = df.groupby([pixel_index,'VWC_Measurement_Depth','SAR_Plot'])

    #get average of VWC values
    avg_pix = pixel_group['VWC','above_0.06','above_0.12','above_0.2'].agg(np.mean)

    #getting error of VWC
    std_vwc = pixel_group['VWC'].agg(np.std)

    avg_pix['VWC_std'] = std_vwc

    avg_pix.reset_index(level=['VWC_Measurement_Depth','SAR_Plot'], inplace=True)

    #plotting 6cm vwc vs. above 6cm closest pixel
    df_6cm = avg_pix[avg_pix['VWC_Measurement_Depth']==6]
    df_6cm[['VWC','VWC_std']] =df_6cm[['VWC','VWC_std']]/100.0

    df_12cm = avg_pix[avg_pix['VWC_Measurement_Depth']==12]
    df_12cm[['VWC','VWC_std']] =df_12cm[['VWC','VWC_std']]/100.0

    df_20cm = avg_pix[avg_pix['VWC_Measurement_Depth']==20]
    df_20cm[['VWC','VWC_std']] =df_20cm[['VWC','VWC_std']]/100.0

    dfs = [df_6cm,df_12cm,df_20cm]
    y_col = ['above_0.06','above_0.12','above_0.2']

    #plot settings
    symbol =['^','s','o']
    title = ' In-Situ Soil Moisture vs. derived ABoVE SAR Soil Moisture'
    alpha = [1,0.6,0.4,0.2]
    fill_style = ['top','full','bottom',]
    edge_colors =['yellow','purple','orange']
    size = [10,10,10]
    save_name = savename
    fig,ax=plt.subplots(figsize=(15,10))

    set = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628" ,"#F781BF", "#999999"]

    colors = {'Teller':set[2],'Kougarok':set[1],'Council':set[0],'Barrow':set[3]}

    for df1 in dfs:
        grouped = df1.groupby('pixel_index')
        for key,group in grouped:
            key = group['site'].iloc[0]
            print(key)
            print(group[['site','sar_plot']])
            print(col_name,y_col,colors,size,col_err)

            #average groupby
            x_data = group[col_name[i]].mean()
            y_data = group[y_col[i]].mean()
            xerr = group[col_name[i]].std()
            yerr = group[y_col[i]].std()

            #group.plot(ax=ax,x=x_data,y=y_data,xerr=xerr,yerr=yerr,kind='scatter',s=size[i],color=colors[key],edgecolors='k',linewidths=1.5 ,marker=MarkerStyle(symbol[i]))#,fillstyle=fill_style[i]),edgecolors=edge_colors[i])#,edgecolors=edge_colors[i],linewidths=2)
            ax.errorbar(x_data,y_data,xerr=xerr,yerr=yerr,markersize=size[i],color=colors[key],fmt=symbol[i])
            plt.rcParams['lines.linewidth'] = 0.5

    plt.xlim(0,1)
    plt.ylim(0,1)

    #creating custom legends
    legend_elements = [ Line2D([0], [0], marker='^', color='w', label='6cm',
                          markerfacecolor='grey', markersize=10,markeredgecolor='k',markeredgewidth=2),
                      Line2D([0], [0], marker='s', color='w', label='12cm',
                            markerfacecolor='grey', markersize=15,markeredgecolor='k',markeredgewidth=2),
                        Line2D([0], [0], marker='o', color='w', label='20cm',
                            markerfacecolor='grey', markersize=25,markeredgecolor='k',markeredgewidth=2)]

    legend_elements2 = [ Line2D([0], [0], marker='o', color='w', label='Teller',
                          markerfacecolor=set[2], markersize=15,markeredgecolor='k'),
                      Line2D([0], [0], marker='o', color='w', label='Kougarok',
                            markerfacecolor=set[1], markersize=15,markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='w', label='Council',
                            markerfacecolor=set[0], markersize=15,markeredgecolor='k'),
                            Line2D([0], [0], marker='o', color='w', label='Barrow',
                                markerfacecolor=set[3], markersize=15,markeredgecolor='k')]

    leg = plt.legend(handles=legend_elements, loc='upper right',prop={'size':20})
    ax.add_artist(leg)

    ax.legend(handles = legend_elements2, loc='upper left',prop={'size':16})
    #ax.add_artist(leg2)

    #create 1:1 line
    y =np.linspace(0,1,100)
    x = np.linspace(0,1,100)
    plt.plot(x,y,ls='--',c='k')
    plt.xlabel('Volumetric Water Content (%/100)',fontsize=18)
    plt.ylabel('ABoVE SAR P-Band Flight (%/100)',fontsize=18)
    plt.title(title,fontsize=22)
    #plt.savefig(save_name,dpi=500)
    plt.show()
    #plt.close()




    '''ax= df_6cm.plot(x='VWC',y='above_0.06',c='b',label='6 cm',kind='scatter')
    df_12cm.plot( x='VWC',y='above_0.12',color='g',ax = ax,label='12 cm',kind='scatter')
    df_20cm.plot( x='VWC',y='above_0.2',color='r',ax = ax,label='20 cm',kind='scatter')

    x = np.arange(-0.2,1.2,0.1)
    plt.plot(x,x,'k--')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()'''



def adjusting_SAR_Plot_names(df,column,prefix,replacement):
    """Short summary.

    Parameters
    ----------
    df : Pandas Dataframe
        Data
    column : string
        Column within dataframe
    prefix : string
        single element or list of strings that are the start of a value in the column you wish to change
    replacement : string
        Replacement value for the prefix

    Returns
    -------
    Pandas Dataframe
        Dataframe with changed column.

    """
    i = 0
    for pre in prefix:
        df[column][df[column].str.startswith(pre)] = replacement[i]
        i=i+1
    return df
