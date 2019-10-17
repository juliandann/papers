import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import matplotlib
from matplotlib.pyplot import figure

plt.style.use('bmh')
def reindex_and_interpolate(df, new_index):
    return df.reindex(df.index | new_index).interpolate(method='index', limit_direction='both').loc[new_index]

def plotting_elev_w_intensity(data,name,offset=0.0):
    '''This function is meant to be used in a loop with the lidar intensity and elevation data'''
    #grabbing a colorbar
    cm = plt.cm.get_cmap('RdYlBu_r')
    plt.scatter(data.index,(data['elev']-data['elev'].mean())+offset,c=data['intensity'],cmap =cm,s=15,vmin=10,vmax=40,edgecolors='none')

    #inserting text one the side with the id name
    avg_elev = data['elev'].mean()
    plt.text(68,offset,name,size=12)
    plt.xlim(-5,80)
def main():

    #grab all files
    path = 'Z:/AKSeward/2017_SAR/CJW_Phase_3_Poster/data/'
    files = [f for f in glob.glob(path + "TL_*.csv")]#, recursive=True)]

    SAR_plots = ['TL_SAR_4','TL_SAR_8','TL_SAR_7']
    i=0
    df_array=[]
    df_id = []

    #read the dtm values and merge with the intensity on index
    for f in files:
        for plot in SAR_plots:
            if plot in f:
                if 'intensity' not in f:
                    df1 = pd.read_csv(f,sep=',')
                    df1.columns =['X','elev']
                    df1_name = plot
                    if '_T1_' in f:
                        df1_tran = 'T1'
                    else:
                        df1_tran='T2'
                else:
                    df2 = pd.read_csv(f,sep=',')
                    df2.columns = ['Y','int']
                    df2_name = plot
                    if '_T1_' in f:
                        df2_tran = 'T1'
                    else:
                        df2_tran='T2'
        i = i+1
        #if it is an even number
        if i %2 ==0:
            print(df1_name+'_'+df1_tran)

            #interpolating the results to the same index
            df2=df2.apply(pd.to_numeric,args=('coerce',))
            #df2.fillna(np.nan)
            df2=df2.dropna(axis=0)
            df2.to_csv('Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/test2.csv',sep=',')
            df1=df1.set_index('X')
            df2=df2.set_index('Y')


            z = np.arange(0.0,60.2,0.2)
            newindex = pd.Float64Index(z)
            df1_reindexed = reindex_and_interpolate(df1, newindex)
            df2_reindexed = reindex_and_interpolate(df2, newindex)
            '''
            plt.plot(df2['int'])
            plt.plot(df2_reindexed['int'])
            plt.show()
            '''
            df3 = pd.concat([df1_reindexed,df2_reindexed],ignore_index=True,axis=1)

            df3.columns =['elev','intensity']
            df_array.append(df3)
            df_id.append(df1_name+'_'+df1_tran)
    l=0
    figure(figsize=(15,10))
    for j in range(0,len(df_id)):
        plotting_elev_w_intensity(df_array[j],df_id[j],offset=l)
        l=l+1.5



    plt.title('Elevation and Intensity along SAR Transects',size=16)
    plt.ylabel('Relative Elevation w/ offset (m) ',size=14)
    plt.xlabel('Distance along Transect (m)',size=14)
    cb=plt.colorbar()
    cb.set_label(label='Lidar Intensity (W/m)',size=14)
    plt.savefig('Z:/AKSeward/2017_SAR/CJW_Phase_3_Poster/figures/lidar_elev_int.pdf',dpi=500)
    plt.show()
    #plt.close()
if __name__ == "__main__":
    main()
