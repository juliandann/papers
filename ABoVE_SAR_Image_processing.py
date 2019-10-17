import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import time

start_time = time.time()
def indi_int_depth_calc(x,vwc,dif_array,int_depth):

    for i in range(0,len(x)):
        if i==0:
            int_vwc = (vwc[i] * dif_array[i])/int_depth

        else:
            int_vwc = int_vwc + (vwc[i] * dif_array[i])/int_depth

    return int_vwc
def int_depth_calc(df,col_names,depthdif_array,int_depth):
    '''function to pass into apply lambda in order to calculate the integrated depth'''
    #vwc = sum()

    for i in range(0,len(col_names)):
        if i ==0:
            df['int_vwc_'+str(int_depth)] = (df[col_names[i]] * depthdif_array[i])/(int_depth)
        else:
            df['int_vwc_'+str(int_depth)] = df['int_vwc_'+str(int_depth)]+ (df[col_names[i]] * depthdif_array[i])/(int_depth)
    return df
def calc_int_depth_avg(df,int_depth,rate):
    '''this program takes a pandas dataframe with the data from Richard Chen to calculate volumetric water content at a predefined level that we do a very simple numerical integration for'''
    #create depths to calculate value for for each pixel every 2cm sample a discrete value
    depth_array = np.arange(0.05,int_depth+rate,rate)
    col_names = ['mv1_aug']
    #calculating columns of vwc for each depth
    for depth in depth_array:
        # numpy where STYLE
        df[str(depth)] = np.where(df['z1_aug'] > depth, df['mv1_aug'],(df['mv1_aug'] * df['z1_aug'] + df['mv2'] *(depth - df['z1_aug']))/depth )
        col_names.append(str(depth))

    #get rid of the 2nd value in col_names which is the minimum z1
    del col_names[1]

    int_vwc = []
    depthdif_array = []

    #make array of differences between discrete values
    for j in range(0,len(depth_array)):
        if j ==0:
            depthdif_array.append(depth_array[j]- 0.0)
        else:
            depthdif_array.append(depth_array[j]- depth_array[j-1])

    #assuming that the discrete value applies until the previous value calculate the avg soil moisture
    df = int_depth_calc(df,col_names,depthdif_array,int_depth)


    return df

def discrete_val_exporter(df,depth):
    '''function to produce dataframe with discrete depth value '''
    df[str(depth)] = np.where(df['z1_aug'] > depth, df['mv1_aug'],(df['mv1_aug'] * df['z1_aug'] + df['mv2'] *(depth - df['z1_aug']))/depth )

    return df

def crop_file_to_box(df,northing,easting,radius):
    '''crop the file to region around point in northing and easting and radius in meters '''
    #setting the boundaries
    upper_northing = northing - radius
    lower_northing = northing+radius
    upper_easting = easting + radius
    lower_easting = easting - radius

    #df = df[(df['x']< upper_northing) & (df['x'] > lower_northing) &  (df['y'] < upper_easting )&  (df['y'] > lower_easting )]
    df = df[(df['x']> upper_northing) & (df['x'] < lower_northing)&  (df['y'] < upper_easting )&  (df['y'] > lower_easting )]
    print(df[['x','y']])
    return df
def main():
    sew_aug= 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/P_PolSAR_ALP_seward_170817_171010_01.nc4'
    bw_2017 = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/P_PolSAR_ALP_barrow_170813_171009_01.nc4'

    ds = xr.open_dataset(bw_2017)#,drop_variables=[''])
    df = ds.to_dataframe()
    ds.close()
    print(ds.data_vars)
    #drop columns
    df =df.drop(columns=['mv1_oct_uncertainty','z1_oct_uncertainty','epsilon1_oct','mv1_oct','z1_oct','h','epsilon1_oct_uncertainty','h_uncertainty'])
    df = df[np.isfinite(df['z1_aug'])]
    print(df)
    print(list(df))

    depth = [0.06,0.12,0.20]
    for i in depth:
        df = discrete_val_exporter(df,i)
    df.reset_index(level=df.index.names,inplace=True)
    df = crop_file_to_box(df,-1901823,4410293,5000) #-2660547,4234227
    df.to_csv('Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_6_12_20_BW.csv',sep=',')
    '''
    #del ds
    #sew_aug_12 = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/real_12cm.csv'
    #df = pd.read_csv(sew_aug_12,sep=',')
    #df.plot.scatter(x='x',y='y',c='0.12',colormap='jet_r',s=1,vmin=0.0,vmax=0.5)
    #plt.show()
    #df=df[:10]
    #df2 = calc_int_depth_avg(df,0.12,0.001)
    '

    df2 = pd.read_csv('Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/test_nonint_20.csv',sep=',')
    #plotting soil profile moisture
    col_names = list(df2)
    col_names = col_names[-152:]
    x_axis = [float(i) for i in col_names]

    depthdif_array = []

    #make array of differences between discrete values
    for j in range(0,len(x_axis)):
        if j ==0:
            depthdif_array.append(x_axis[j]- 0.0)
        else:
            depthdif_array.append(x_axis[j]- x_axis[j-1])

    for index, row in df2.iterrows():
        if index < 10:
            arr = np.array(row[col_names])
            plt.plot(x_axis,arr)
            vwc = indi_int_depth_calc(x_axis,arr,depthdif_array,0.2)
            plt.scatter(0.202,vwc)
            print(row[['x','y','z1_aug']],vwc)
        else:
            break
    plt.show()

    #print(df2)

    df2 = df2.filter(['z1_aug','alt','alt_uncertainty','mv1_aug','lat','lon','x','y','int_vwc_0.12'])
    df2 = calc_int_depth_avg(df,0.06,0.001)
    df2 = df2.filter(['z1_aug','alt','alt_uncertainty','mv1_aug','lat','lon','x','y','int_vwc_0.06','int_vwc_0.12'])
    df2 = calc_int_depth_avg(df,0.2,0.001)
    df2 = df2.filter(['z1_aug','alt','alt_uncertainty','mv1_aug','lat','lon','x','y','int_vwc_0.06','int_vwc_0.12','int_vwc_0.2'])


    df2 = pd.read_csv('Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_6_12_20_layers.csv',sep=',')
    df = crop_file_to_box(df2,-2656185,4297055,3000)
    df.to_csv('Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_6_12_20_layers_KG.csv',sep=',')
    '''
    #ds = df2.to_xarray()
    #print(ds)
    #ds.to_netcdf('Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/integrated_version.nc4',format='NETCDF3_64BIT')
    #df2.to_csv('Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/all_6_12_20_alt_v2.csv',sep=',')
    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == "__main__":
    main()
