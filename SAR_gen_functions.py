import numpy as np
import pandas as pd
import geopy.distance
import xarray as xr
import time
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib
import matplotlib.dates as mdates



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

    #get rid of barrow
    df.Locale=df.Locale.str.replace(' ','')
    df = df[df.Locale != 'Barrow']
    print(df.Locale.unique())
    #groupby pixel index and measurement depth
    pixel_group = df.groupby([pixel_index,'VWC_Measurement_Depth','SAR_Plot','Locale'])

    #get average of VWC values
    avg_pix = pixel_group['VWC','above_0.06','above_0.12','above_0.2','lbc_0.06','lbc_0.12','lbc_0.2'].agg(np.mean)

    #getting error of VWC
    std_vwc = pixel_group['VWC'].agg(np.std)

    avg_pix['VWC_std'] = std_vwc
    avg_pix[['VWC','VWC_std']] = avg_pix[['VWC','VWC_std']]/100.0

    avg_pix.reset_index(level=['VWC_Measurement_Depth','SAR_Plot','Locale'], inplace=True)

    #print(avg_pix)
    #plotting 6cm vwc vs. above 6cm closest pixel
    df_6cm = avg_pix[avg_pix['VWC_Measurement_Depth']==6]
    df_12cm = avg_pix[avg_pix['VWC_Measurement_Depth']==12]
    df_20cm = avg_pix[avg_pix['VWC_Measurement_Depth']==20]

    dfs = [df_6cm,df_12cm,df_20cm]
    y_col = ['above_0.06','above_0.12','above_0.2']

    #plot settings
    symbol =['^','s','o']
    title = ' In-Situ Soil Moisture vs. derived ABoVE SAR Soil Moisture Per Pixel'
    alpha = [1,0.6,0.4,0.2]
    fill_style = ['top','full','bottom',]
    edge_colors =['yellow','purple','orange']
    size = [10,10,10]
    save_name = savename
    fig,ax=plt.subplots(figsize=(15,10))
    col_name = 'VWC'
    y_col =['above_0.06','above_0.12','above_0.2']
    depth = ['0.06cm','0.12cm','0.2cm']
    set = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628" ,"#F781BF", "#999999"]

    colors = {'Teller':set[2],'Kougarok':set[1],'Council':set[0],'Barrow':set[3]}

    #fit a linear regression to x,y
    j=0

    for df in dfs:
        x= np.array(df['VWC'].values).reshape((-1, 1))
        y = df[y_col[j]].values
        model = LinearRegression().fit(x,y)
        r_sq = model.score(x,y)
        print(depth[j])
        print('r_sq for linear regression:',r_sq)
        print('intercept:', model.intercept_)
        print('slope:', model.coef_)
        j=j+1

    r2_6cm = r2_score(df_6cm['above_0.06'],df_6cm['VWC'])
    r2_12cm = r2_score(df_12cm['above_0.12'],df_12cm['VWC'])
    r2_20cm = r2_score(df_20cm['above_0.2'],df_20cm['VWC'])
    print('R-Squared values for 1:1 line: ')
    print('r^2 6cm: '+str(r2_6cm))
    print('r^2 12cm: '+str(r2_12cm))
    print('r^2 20cm: '+str(r2_20cm))
    i=0

    #calculating an overarching r squared and line of best fit

    for df1 in dfs:
        #print(list(df1))
        grouped = df1.groupby(pixel_index)
        for key,group in grouped:
            #print('Key: '+str(key))

            #print('Group: '+str(group))
            key = (group['Locale'].iloc[0]).strip()
            depth=group['VWC_Measurement_Depth'].iloc[0]
            #print('Key: '+str(key))
            #print(group[['Locale','SAR_Plot']])
            #print('Here!'+str(col_name)+str(y_col))


            #average groupby
            x_data = group[col_name]
            y_data = group[y_col[i]]
            xerr = group['VWC_std']


            #group.plot(ax=ax,x=x_data,y=y_data,xerr=xerr,yerr=yerr,kind='scatter',s=size[i],color=colors[key],edgecolors='k',linewidths=1.5 ,marker=MarkerStyle(symbol[i]))#,fillstyle=fill_style[i]),edgecolors=edge_colors[i])#,edgecolors=edge_colors[i],linewidths=2)
            ax.errorbar(x_data,y_data,xerr=xerr,markersize=size[i],color=colors[key],fmt=symbol[i])
            plt.rcParams['lines.linewidth'] = 0.5
        i=i+1
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
                            markerfacecolor=set[0], markersize=15,markeredgecolor='k')]

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
    plt.savefig(save_name,dpi=500)
    #plt.show()
    plt.close()




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

def engstrom_SM(df,dielectric):
    """SAR soil moisture calibration from Engstrom et al. (2005)

    Parameters
    ----------
    df : dataframe
        Dataframe of ABoVE
    dielectric : string
        Column name for dielectric constant

    Returns
    -------
    dataframe
        Dataframe with extra column

    """
    df['test_VWC'] = -2.50 + (2.508*df[dielectric]) - (0.03634*(df[dielectric]**2.0)) + (0.0002394*(df[dielectric]**3.0))
    df.to_csv('Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_all_sites_engstromtest.csv')

def td_comparison(df,pixel_index,var1,var2,var2_error,save_folder):
    """Comparison of in-situ thaw depth with ABoVE data using pixel binning.

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe from csv from closest_point_index()
    pixel_index : string
        Column name of thing to use to bin same values
    save_folder : string
        folder to save plots into
    var1 : string
        Column name of first variable to compare with var2
    var2 : string
        Column name of second variable to compare with var1
    var2_error : string
        Column name of error associated with var2

    Returns
    -------
        Plots can be found in save_folder

    """
    '''
    #get rid of -9999 values
    df = df.drop(df[df.Thaw_Depth == -9999].index)

    pixel_group = df.groupby([pixel_index])

    avg_td = pixel_group[var1,var2,var2_error].agg(['mean','count'])
    avg_td.to_csv(save_folder+'avg_td_per_pixel.csv')
    '''

    fig,ax=plt.subplots(figsize=(15,10))

    legend_elements = [ Line2D([0], [0], marker='o', color='red', label='0 - 5',
                          markerfacecolor='red', markersize=10),
                          Line2D([0], [0], marker='o', color='orange', label='6 - 10', markerfacecolor='orange', markersize=10),
                          Line2D([0], [0], marker='o', color='green', label='11 - 15',markerfacecolor='green', markersize=10),
                          Line2D([0], [0], marker='o', color='blue', label='16 -20',
                                                markerfacecolor='blue', markersize=10),]
    #plt.errorbar(df[var1],df[var2]*100.0,yerr=df[var2_error]*100.0,fmt='o',mfc=df['Count'])
    #df.plot(var1,var2,yerr=var2_error,c='Count',kind='scatter',colormap='viridis',ax = ax)
    colors = df.groupby('Color')
    r_score = r2_score(df[var1],df[var2])
    print(r_score)
    for name, group in colors:
        print(name,group)
        #group.plot(var1,var2,yerr=var2_error,c=str(name), kind='scatter')
        plt.errorbar(group[var1],group[var2],yerr=group[var2_error],ecolor=str(name),c=str(name),fmt='o')

    ax.legend(handles=legend_elements,title= 'Counts',fontsize=18)
    plt.text(78,25,'R-Squared ='+str(round(r_score,3)),horizontalalignment='right',fontsize=20)
    x = np.linspace(0,100,100)
    y = x

    plt.plot(x,y,'--k')
    plt.ylim(20,80)
    plt.xlim(20,80)
    plt.xlabel('In-Situ Thaw Depth (cm)',fontsize=18)
    plt.ylabel('ABoVE Pixel ALT (cm)',fontsize=18)
    plt.title('ABoVE ALT vs. In-Situ ALT',fontsize=24)
    plt.savefig(save_folder+'ALT_overall_comparison.png')
    plt.close()

def r_squared_1_1_line(x,y):
    """Compares x,y values with one to one line

    Parameters
    ----------
    x : Pandas column
        x values (in-situ hydrosense)
    y : Pandas Column
        y values (ABoVE values)

    Returns
    -------
    float
        R squared value compared to 1:1 line

    """
    #set up 1:1 line
    y_1 = x
    avg_y = np.mean(y)
    residual_1 = np.sum((y-y_1)**2.0)
    residual_null = np.sum((y-avg_y)**2.0)
    r2 = (residual_null - residual_1)/residual_null
    print('residual 1:1: '+str(residual_1))
    print('residual null: '+str(residual_null))
    return r2

def landcover_boxplots(df,class_cat,classes,mean,std,count):
    """Function written to analyze landcover type by h_value

    Parameters
    ----------
    df : dataframe
        Main data
    class_cat : string
        Overarching category
    classes : string
        Subclasses
    mean : string
        average value
    std : string
        standard deviation
    count : string
        counts

    Returns
    -------
    type
        Description of returned object.

    """
    print(list(df))
    grouped = df.groupby(classes)
    type = []
    fweighted_mean = []
    fweighted_std = []
    i=0
    ax = plt.figure(figsize=(24,12))
    for key,group in grouped:

        type.append(key)
        weighted_mean = []
        weighted_std = []
        j=0
        for row_index,row in group.iterrows():
            if j == 0:
                weighted_mean.append( (row[mean])*(row[count] / group[count].sum()))
                weighted_std.append( (row[std])*(row[count] / group[count].sum()))
            else:
                weighted_mean.append( weighted_mean[-1]+ (row[mean])*(row[count] / group[count].sum()))
                weighted_std.append(weighted_std[-1]+ (row[std])*(row[count] / group[count].sum()))
            j=j+1
        fweighted_mean.append(weighted_mean[-1])
        fweighted_std.append(weighted_std[-1])


        plt.errorbar(x=group[class_cat],y=group[mean],yerr=group[std],capsize=3,ls='',marker='o',label=type[-1])

        for row_index,row in group.iterrows():
            plt.text(i,0.5,str(round(row[count],2)),ha='center',rotation='vertical')
            i=i+1

        #print(group[['Ecosystem_LU','above_h_count','ABoVE_h_mean']])
    zippedlist = list(zip(type,fweighted_mean,fweighted_std))
    y_data = 'lbc_6_mean'
    y_err = 'lbc_6_std'
    df2 = pd.DataFrame(zippedlist, columns = ['Class' , y_data,y_err])

    #plt.tight_layout()


    bottom = df2[y_data]-df2[y_err]
    top = df2[y_data]+df2[y_err]
    df2['height'] = top-bottom
    df2['colors'] = ['blue','orange','green','red','purple','brown']
    #plt.bar(x=df2['Class'],height=df2['height'],bottom=bottom,linewidth=2.0,ecolor='black',edgecolor='black')
    #ax= df2.plot(x='Class',y='height',kind='bar',bottom=bottom,legend=False,ecolor='black',edgecolor='black')
    #df2.plot(x='Class',y=y_data,yerr=y_err,color=df2['colors'],kind='bar',legend=False)
    #plt.errorbar(x=df2['Class'],y=df2['h_mean'],yerr=df2['h_std'],capsize=3,ls='',color=df2['colors'])

    plt.legend(fontsize=15,loc=1)
    plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.48)
    plt.xticks(ha='center',rotation='vertical')
    plt.ylim(0.1,0.75)
    plt.xlabel('Vegetation Classification (GAP/LANDFIRE)',fontsize=18)
    plt.ylabel('ABoVE LBC 6cm (%/100)',fontsize=18)
    plt.title('LBC 6cm vs. Landcover Type',fontsize=24)
    #plt.show()
    plt.savefig('Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/Figures/GapLandfire/lbc_6cm_discrete.png',dpi=500)
    plt.close()
def laura_bc_SM(df,e1,e2,z1,depth):
    #tau values from Hydrosense through R. Chen
    tau_12cm = (4.312 + np.sqrt(df[e1]))/5.354
    tau_20cm = (3.43 + np.sqrt(df[e2]))/3.158

    #Laura bc calibration equations
    mv1 = ((-24.28 * (tau_12cm**2.0)) + 134.55*tau_12cm - 110.245)/100.0
    mv2 = ((7.693 * (tau_20cm**2.0)) + 1.641*tau_20cm - 12.341)/100.0

    depth_tag = 'lbc_'+str(depth)

    #conditions
    df[depth_tag] = np.where(df[z1] > depth, mv1,(mv1 * df[z1] + mv2 *(depth - df[z1]))/depth )
    return df

def laura_bc_indi(e1,e2,z1,depth):
    tau_12cm = (4.312 + np.sqrt(e1))/5.354
    tau_20cm = (3.43 + np.sqrt(e2))/3.158

    #Laura bc calibration equations
    mv1 = ((-24.28 * (tau_12cm**2.0)) + 134.55*tau_12cm - 110.245)/100.0
    mv2 = ((7.693 * (tau_20cm**2.0)) + 1.641*tau_20cm - 12.341)/100.0

    if z1 > depth:
        mv=mv1
    else:
        mv = ((mv1 * z1) + mv2* (depth - z1))/depth

    return mv

def engstrom_SM_mv(df,mv1,mv2,z1,depth):
    depth_tag = 'engstrom_'+str(depth)
    df[depth_tag] = np.where(df[z1] > depth, df[mv1],(df[mv1] * df[z1] + df[mv2] *(depth - df[z1]))/depth )
    return df

def compare_above_vs_hydrosense(df,x,y,xerr=[],yerr=[],xlabel='',ylabel='',title='',save_name='test.png'):
    '''This function is meant to take the ABoVE SAR values with discrete measurements at 6,12,20 cm and compare with the hydrosense data'''

    #use ggplot
    matplotlib.style.use('ggplot')

    #make date into datetime object
    df['Date'] = pd.to_datetime(df['Date'],errors='coerce')

    #take only late dates
    VWC_df_late = df[(df['Date']>pd.Timestamp(2017,7,1))]

    symbol =['^','s','o']
    alpha = [1,0.6,0.4,0.2]

    #divide each column by 100.0
    VWC_df_late[x] = VWC_df_late[x]/100.0
    if len(xerr) > 1:
        VWC_df_late[xerr] = VWC_df_late[xerr]/100.0

    size = [10,10,10]

    fig,ax=plt.subplots(figsize=(15,10))
    #colors = {'Teller':'royalblue','Kougarok':'salmon','Council':'seagreen'}
    set = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628" ,"#F781BF", "#999999"]

    colors = {'Teller':set[2],'Kougarok':set[1],'Council':set[0]}

    grouped =  VWC_df_late.groupby(['site','sar_plot'])
    for i in range(0,len(y)):
        for key,group in grouped:
            key = group['site'].iloc[0]
            if key !='Barrow':
                print(key)
                print(group[['site','sar_plot']])

                #average groupby
                x_data = group[x[i]].mean()
                y_data = group[y[i]].mean()
                xerr = group[x[i]].std()
                yerr = group[y[i]].std()

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
                            markerfacecolor=set[0], markersize=15,markeredgecolor='k')]

    leg = plt.legend(handles=legend_elements, loc='upper right',prop={'size':20})
    ax.add_artist(leg)

    ax.legend(handles = legend_elements2, loc='upper left',prop={'size':16})
    #ax.add_artist(leg2)

    #create 1:1 line
    y =np.linspace(0,1,100)
    x = np.linspace(0,1,100)
    plt.plot(x,y,ls='--',c='k')
    plt.xlabel(xlabel,fontsize=18)
    plt.ylabel(ylabel,fontsize=18)
    plt.title(title,fontsize=22)
    #plt.savefig(save_name,dpi=500)
    plt.show()
    #plt.close()

def weather(met_data,title='Temperature and Precipitation at Kougarok (mile 64)'):

    #make date into datetime object
    met_data['Date'] = pd.to_datetime(met_data['Date'],errors='coerce')
    met_data['Rain'][met_data['Rain'] == 6999] = np.nan
    met_data['Air Temp @ 1.5m'][met_data['Air Temp @ 1.5m'] == 6999] = np.nan
    fig, ax1 = plt.subplots(figsize=(15,10))
    ax1.plot(met_data['Date'],met_data['Air Temp @ 1.5m'])
    ax1.set_xlim(pd.Timestamp('2017-07-01'),pd.Timestamp('2017-09-01'))
    ax1.set_ylim(-20,30)
    ax1.set_ylabel('Air Temperature (Celsius)',color='b',fontsize=18)
    ax1.set_xlabel('Date',fontsize=18)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    #initiate secondary y axis
    ax2 = ax1.twinx()
    ax2.plot(met_data['Date'],met_data['Rain'],color='r')
    ax2.set_ylabel('Rain (cm)',color='r',fontsize=16)
    ax2.tick_params(axis='y', labelcolor='r')

    above= plt.axvline(x=pd.Timestamp('2017-08-17'),color='k',linestyle='dashed',label='P-Band ABoVE SAR Flight')
    #plt.axvline(x=pd.Timestamp('2017-08-16'),color='k',linestyle='dashed',label='In-Situ Measurents')
    insitu = ax2.axvspan(pd.Timestamp('2017-08-17 00:00:00'),pd.Timestamp('2017-08-17 23:59:59'),label='In-Situ Measurents',color='crimson',alpha=0.3)
    plt.legend(handles=[above,insitu],fontsize=18,loc=1)
    plt.title(title,fontsize=24)

    plt.setp(ax1.get_xticklabels(), rotation=30)
    #plt.gcf().autofmt_xdate()
    #plt.show()
    plt.savefig('Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/Figures/Weather/Kougarok_weather.png',dpi=500)
    plt.close()

def linear_regression_plots(df,pixel_index,savename='test.png'):

    #get rid of barrow
    df.Locale=df.Locale.str.replace(' ','')
    df = df[df.Locale != 'Barrow']

    #groupby pixel index and measurement depth
    pixel_group = df.groupby([pixel_index,'VWC_Measurement_Depth','Locale'])

    #get average of VWC values
    avg_pix = pixel_group['VWC','above_0.06','above_0.12','above_0.2','lbc_0.06','lbc_0.12','lbc_0.2'].agg(np.mean)


    #getting error of VWC
    std_vwc = pixel_group['VWC'].agg(np.std)

    avg_pix['VWC_std'] = std_vwc
    avg_pix.reset_index(level=['VWC_Measurement_Depth','Locale'], inplace=True)
    avg_pix[['VWC','VWC_std']] = avg_pix[['VWC','VWC_std']]/100.0

    grouped = avg_pix.groupby(['Locale','VWC_Measurement_Depth'])

    #set up figure
    fig,ax = plt.subplots(3,3,sharex='col',sharey='row',figsize=(15,12))
    ax = ax.flatten()

    #plot settings
    symbol ={'6':'^','12':'s','20':'o'}
    set = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628" ,"#F781BF", "#999999"]

    colors = {'Teller':set[2],'Kougarok':set[1],'Council':set[0],'Barrow':set[3]}


    j=0

    for key,group in grouped:
        site = key[0]
        depth = str(key[1])
        if depth == '6':
            y = 'lbc_0.06'
        if depth == '12':
            y = 'lbc_0.12'
        if depth == '20':
            y = 'lbc_0.2'


        #r-squared analysis
        x= np.array(group['VWC'].values).reshape((-1, 1))
        y = group[y].values
        model = LinearRegression().fit(x,y)
        r_sq = model.score(x,y)
        print('r_sq for linear regression:',r_sq)
        print('intercept:', model.intercept_)
        print('slope:', model.coef_)


        ax[j].errorbar(x,y,fmt=symbol[depth],color=colors[site])
        x_2 = np.linspace(0,1,20)
        ax[j].plot(x_2,model.coef_ *x_2 +model.intercept_,ls='--',c='k',lw=1.5)
        ax[j].set_xlim(0,1)
        ax[j].set_ylim(0,1)
        ax[j].set_title(site+' '+str(depth)+'cm')
        ax[j].text(0.05,0.9,r'r$^{2}$ ='+str(round(r_sq,2)))
        plt.rcParams['lines.linewidth'] = 0.5

        j=j+1

    plt.text(0.5,0.05,'In-Situ VWC (%/100)',horizontalalignment='center',transform=plt.gcf().transFigure,fontsize=18)
    plt.text(0.05,0.5,'P-Band SAR Laura BC Method (%/100)',verticalalignment='center',rotation='vertical',transform=plt.gcf().transFigure,fontsize=18)
    plt.text(0.5,0.95,r'R$^{2}$'+' Analyses',horizontalalignment='center',transform=plt.gcf().transFigure,fontsize=24)
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
                            markerfacecolor=set[0], markersize=15,markeredgecolor='k')]

    leg = fig.legend(handles=legend_elements, loc='upper right',prop={'size':16})
    fig.add_artist(leg)

    fig.legend(handles = legend_elements2, loc='upper left',prop={'size':16})

    plt.savefig(savename,dpi=500)
    plt.close()
    plt.show()

def linear_regression_main(df,pixel_index,savename='test.png'):
    """Plot of all above VWC pixels compared wiith their vwc counterpoints and taking a linear regression and an R^2 value.

    Parameters
    ----------
    df : Pandas Dataframe
        closest ABoVE pixels to VWC
    pixel_index : string column name
        Description of parameter `pixel_index`.
    savename : string
        Save name for plot.

    Returns
    -------
    nothing

    """

    #get rid of barrow
    df.Locale=df.Locale.str.replace(' ','')
    df = df[(df.Locale != 'Barrow') ]

    print(df.Locale.unique())
    #groupby pixel index and measurement depth
    pixel_group = df.groupby([pixel_index,'VWC_Measurement_Depth','Locale'])

    #get average of VWC values
    avg_pix = pixel_group['VWC','above_0.06','above_0.12','above_0.2','lbc_0.06','lbc_0.12','lbc_0.2'].agg(np.mean)

    #getting error of VWC
    std_vwc = pixel_group['VWC'].agg(np.std)

    avg_pix['VWC_std'] = std_vwc
    avg_pix.reset_index(level=['VWC_Measurement_Depth','Locale'], inplace=True)
    avg_pix[['VWC','VWC_std']] = avg_pix[['VWC','VWC_std']]/100.0

    grouped = avg_pix.groupby(['VWC_Measurement_Depth'])

    #set up figure
    fig,ax = plt.subplots(3,1,figsize=(7,15),sharex=True)
    ax= ax.flatten()
    #plot settings
    symbol ={'6':'^','12':'s','20':'o'}
    set = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628" ,"#F781BF", "#999999"]

    colors = {'Teller':set[2],'Kougarok':set[1],'Council':set[0],'Barrow':set[3]}


    rsq = []
    intercepts = []
    coeff = []

    j=0
    for key,group in grouped:
        depth = str(key)
        if depth == '6':
            y = 'lbc_0.06'
        if depth == '12':
            y = 'lbc_0.12'
        if depth == '20':
            y = 'lbc_0.2'
        #r-squared analysis
        x= np.array(group['VWC'].values).reshape((-1, 1))
        y = group[y].values
        model = LinearRegression().fit(x,y)
        r_sq = model.score(x,y)
        print('r_sq for linear regression:',r_sq)
        print('intercept:', model.intercept_)
        print('slope:', model.coef_)
        rsq.append(r_sq)
        intercepts.append(model.intercept_)
        coeff.append(model.coef_)
        x_2 = np.linspace(0,1,20)
        ax[j].plot(x_2,model.coef_ *x_2 +model.intercept_,ls='--',c='k',lw=1.5)
        j=j+1

    grouped = avg_pix.groupby(['Locale','VWC_Measurement_Depth'])

    j=0
    for key,group in grouped:
        site = key[0]
        depth = str(key[1])
        if depth == '6':
            y = 'lbc_0.06'
            j=0
        if depth == '12':
            y = 'lbc_0.12'
            j=1
        if depth == '20':
            y = 'lbc_0.2'
            j=2

        ax[j].errorbar(group['VWC'],group[y].values,fmt=symbol[depth],color=colors[site])
        ax[j].set_xlim(0,1)
        ax[j].set_ylim(0,1)
        ax[j].set_title(str(depth)+'cm' )
        ax[j].text(0.05,0.9,r'r$^{2}$ ='+str(round(rsq[j],2)))
        plt.rcParams['lines.linewidth'] = 0.5



    plt.text(0.5,0.05,'In-Situ VWC (%/100)',horizontalalignment='center',transform=plt.gcf().transFigure,fontsize=18)
    plt.text(0.05,0.5,'P-Band SAR Laura BC Method (%/100)',verticalalignment='center',rotation='vertical',transform=plt.gcf().transFigure,fontsize=18)
    plt.text(0.5,0.95,r'R$^{2}$'+' Analyses',horizontalalignment='center',transform=plt.gcf().transFigure,fontsize=24)
    #creating custom legends
    legend_elements = [ Line2D([0], [0], marker='^', color='w', label='6cm',
                          markerfacecolor='grey', markersize=10,markeredgecolor='k',markeredgewidth=2),
                      Line2D([0], [0], marker='s', color='w', label='12cm',
                            markerfacecolor='grey', markersize=15,markeredgecolor='k',markeredgewidth=2),
                        Line2D([0], [0], marker='o', color='w', label='20cm',
                            markerfacecolor='grey', markersize=25,markeredgecolor='k',markeredgewidth=2)]

    legend_elements2 = [ Line2D([0], [0], marker='o', color='w', label='Kougarok',
                          markerfacecolor=set[1], markersize=15,markeredgecolor='k'),
                      Line2D([0], [0], marker='o', color='w', label='Kougarok',
                            markerfacecolor=set[1], markersize=15,markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='w', label='Council',
                            markerfacecolor=set[0], markersize=15,markeredgecolor='k')]

    leg = fig.legend(handles=legend_elements, loc='upper right',prop={'size':16})
    fig.add_artist(leg)

    fig.legend(handles = legend_elements2, loc='upper left',prop={'size':16})

    plt.savefig(savename,dpi=500)
    plt.close()
    #plt.show()
def sunfactor_func(df,aspect,sunfactor):
    df[sunfactor] = 0.5*(np.cos((df[aspect]-40.0)/360.*2.*np.pi)-np.sin((df[aspect]-40.)/360.*2.*np.pi))
    return df
def boxplots_macrotopology_vwc_plots(df):
    """Function that makes boxplots 3x1 for each lbc depth against aspect,slope,sunfactor, and curvature.

    Parameters
    ----------
    df : pandas dataframe
        Description of parameter `df`.

    Returns
    -------
    type
        Description of returned object.

    """

    #column names for easy changing
    aspect = 'aspect'
    slope = 'Slope'
    sunfactor = 'sunfactor'
    curve = 'curvature'
    pix_index = 'Index'
    six_cm = 'lbc_0.06'
    twelve_cm = 'lbc_0.12'
    twenty_cm = 'lbc_0.2'
    h = 'h'
    h_err = 'h_uncertainty'
    alt = 'alt'
    alt_err = 'alt_uncertainty'
    vwc_depth = [six_cm,twelve_cm,twenty_cm]

    df = sunfactor_func(df,aspect,sunfactor)
    df[curve] = df[curve]/ 100.0

    slope_bins = np.arange(0,95,5)
    slope_names= []
    aspect_names= []
    sunfactor_names = []
    aspect_bins = np.arange(0,390,30)
    list_slope = []
    list_aspect = []
    list_sunfactor = []
    sunfactor_bins = np.arange(-0.8,0.9,0.1)

    curv_bins = np.arange(-1.0,1.1,0.1)
    list_curv =[]
    curv_names = []

    fig,ax = plt.subplots(1,3,figsize=(20,10))
    ax = ax.flatten()
    #separate pandas column into value 0-5 etc through slope
    for i in range(0,len(slope_bins)-1):
        list_slope.append( df[((df[slope] > float(slope_bins[i])) & (df[slope] <= float(slope_bins[i+1])))])
        slope_names.append(str(slope_bins[i])+' - '+str(slope_bins[i+1]))

    for i in range(0,len(aspect_bins)-1):
        list_aspect.append( df[((df[aspect] > float(aspect_bins[i])) & (df[aspect] <= float(aspect_bins[i+1])))])
        aspect_names.append(str(aspect_bins[i])+' - '+str(aspect_bins[i+1]))

    for i in range(0,len(sunfactor_bins)-1):
        list_sunfactor.append( df[((df[sunfactor] > float(sunfactor_bins[i])) & (df[sunfactor] <= float(sunfactor_bins[i+1])))])
        sunfactor_names.append(str(round(sunfactor_bins[i],2))+' - '+str(round(sunfactor_bins[i+1],2)))

    for i in range(0,len(curv_bins)-1):
        list_curv.append( df[((df[curve] > float(curv_bins[i])) & (df[curve] <= float(curv_bins[i+1])))])
        curv_names.append(str(round(curv_bins[i],2))+' - '+str(round(curv_bins[i+1],2)))

    #slope
    for j in range(0,len(vwc_depth)):
        for i in range(0,len(slope_bins)-1):
            ax[j].boxplot(list_slope[i][vwc_depth[j]].values,positions=[i],widths=0.6)
        ax[j].set_title('Slope vs. ' +str(vwc_depth[j]))
        ax[j].set_xticklabels(slope_names,rotation=45, horizontalalignment='right')
        ax[j].set_xlabel('Slope Range (deg)',fontsize=18)
        ax[j].set_ylim(0,1)
        ax[j].set_ylabel('VWC (%/100)',fontsize=18 )

    plt.savefig('Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/Figures/ABoVE_vs_Macrotopology/slope.png',dpi=500)
    plt.close()

    fig,ax = plt.subplots(1,3,figsize=(20,10))
    ax = ax.flatten()

    #aspect
    m = 0
    for k in range(0,len(vwc_depth)):
        for i in range(0,len(aspect_bins)-1):
            ax[k].boxplot(list_aspect[i][vwc_depth[m]].values,positions=[i],widths=0.6)
        ax[k].set_title('Aspect vs. '+str(vwc_depth[m]))
        ax[k].set_xticklabels(aspect_names,rotation=45, horizontalalignment='right')
        ax[k].set_xlabel('Aspect Range (deg)',fontsize=18)
        ax[k].set_ylim(0,1.0)
        ax[k].set_ylabel('VWC (%/100)',fontsize=18 )
        m=m+1
    plt.savefig('Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/Figures/ABoVE_vs_Macrotopology/aspect.png',dpi=500)
    plt.close()


    fig,ax = plt.subplots(1,3,figsize=(20,10))
    ax = ax.flatten()

    #sunfactor
    n = 0
    for l in range(0,len(vwc_depth)):
        for i in range(0,len(sunfactor_bins)-1):
            ax[l].boxplot(list_sunfactor[i][vwc_depth[n]].values,positions=[i],widths=0.6)
        ax[l].set_xlabel('Sun Factor Range',fontsize=18)
        ax[l].set_ylim(0,1)
        ax[l].set_ylabel('VWC (%/100)',fontsize=18 )
        ax[l].set_title('Sun Factor vs. '+str(vwc_depth[n]))
        ax[l].set_xticklabels(sunfactor_names,rotation=45, horizontalalignment='right')
        n=n+1

    plt.savefig('Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/Figures/ABoVE_vs_Macrotopology/sunfactor.png',dpi=500)
    plt.close()
    #plt.show()

    fig,ax = plt.subplots(1,3,figsize=(20,10))
    ax = ax.flatten()
    #curvature
    for j in range(0,len(vwc_depth)):
        for i in range(0,len(curv_bins)-1):
            ax[j].boxplot(list_curv[i][vwc_depth[j]].values,positions=[i],widths=0.6)
        ax[j].set_title('Curvature vs. ' +str(vwc_depth[j]))
        ax[j].set_xticklabels(curv_names,rotation=45, horizontalalignment='right')
        ax[j].set_xlabel('Curvature Range (deg)',fontsize=18)
        ax[j].set_ylim(0,1)
        ax[j].set_ylabel('VWC (%/100)',fontsize=18 )

    plt.savefig('Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/Figures/ABoVE_vs_Macrotopology/curv.png',dpi=500)
    plt.close()

    fig,ax = plt.subplots(1,3,figsize=(20,10))
    ax = ax.flatten()

def alt_above_topo_comp(df):
    slope = 'Slope'
    curvature = 'curvature'
    aspect = 'aspect'

    #make curvature -1 -> 1
    df[curvature] = df[curvature]/100.0
    slope_bins = np.arange(0,95,5)
    aspect_bins = np.arange(0,390,30)
    curvature_bins = np. arange(-1,1,0.1)

    #set up for loop lists
    bins = [slope_bins,aspect_bins,curvature_bins]
    topo_name = [slope,aspect,curvature]


    fig,ax = plt.subplots(1,3,figsize=(20,10))
    ax = ax.flatten()
    #cycle through topo name
    for i in range(0,len(bins)):

        curv_names = []
        for j in range(0,len(bins[i])-1):
            ranged = df[((df[topo_name[i]] > float(bins[i][j])) & (df[topo_name[i]] <= float(bins[i][j+1])))]
            curv_names.append(str(round(bins[i][j],2))+' - '+str(round(bins[i][j+1],2)))
            ax[i].boxplot(ranged['alt'],widths=0.6,positions=[j])
        ax[i].set_title(topo_name[i].capitalize()+' vs. ' +' ALT ')
        ax[i].set_xticklabels(curv_names,rotation=45, horizontalalignment='right')
        ax[i].set_xlabel(topo_name[i].capitalize()+' Range (deg)',fontsize=18)
        ax[i].set_ylim(0,df['alt'].max()+(df['alt'].max()/10.0))
        ax[i].set_ylabel('ALT (m)',fontsize=18 )
    plt.savefig('Z:/AKSeward/')
