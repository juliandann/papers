import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import time
import matplotlib
from matplotlib.markers import MarkerStyle
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from SAR_gen_functions import *

start_time = time.time()
def compare_discrete_vs_hydrosense(ABoVE_file,VWC_file,depth='20',combine=False):
    '''This function is meant to take the ABoVE SAR values with discrete measurements at 6,12,20 cm and compare with the hydrosense data'''

    #use ggplot
    matplotlib.style.use('ggplot')

    #uncomment to recalculate closest Above pixels

    '''
    #read in data
    ABoVE_df = pd.read_csv(ABoVE_file,sep=',')
    VWC_df = pd.read_csv(VWC_file,sep=',')

    #make date into datetime object
    VWC_df['Date'] = pd.to_datetime(VWC_df['Date'],errors='coerce')

    #take only late dates
    VWC_df_late = VWC_df[(VWC_df['Date']>pd.Timestamp(2017,7,1))]


    #find the closest above pixel to each hydrosense measurement.
    VWC_df_late[['above_6','above_12','above_20','above_lat','above_lon']] = VWC_df_late.apply(lambda row: find_SAR_pixel(ABoVE_df,row['lat'],row['lon'],['0.06','0.12','0.2','lat','lon']),axis=1)


    VWC_df_late.to_csv('Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/everything.csv')
    '''
    VWC_df_late = pd.read_csv('Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/everything.csv',sep=',')


    #defining columns
    col_name = ['avg_'+depth[0],'avg_'+depth[1],'avg_'+depth[2]]
    y_col = ['above_'+depth[0],'above_'+depth[1],'above_'+depth[2]]

    col_err = ['std_'+depth[2],'std_'+depth[1],'std_'+depth[0]]
    symbol =['^','s','o']
    title = ' In-Situ Soil Moisture vs. derived ABoVE SAR Soil Moisture'
    alpha = [1,0.6,0.4,0.2]
    #divide each column by 100.0
    VWC_df_late[col_name] = VWC_df_late[col_name]/100.0
    VWC_df_late[col_err] = VWC_df_late[col_err]/100.0
    fill_style = ['top','full','bottom',]
    edge_colors =['yellow','purple','orange']
    size = [10,10,10]

    save_name = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/2017_aug_hydro_vs_above.png'
    fig,ax=plt.subplots(figsize=(15,10))
    #colors = {'Teller':'royalblue','Kougarok':'salmon','Council':'seagreen'}
    set = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628" ,"#F781BF", "#999999"]

    colors = {'Teller':set[2],'Kougarok':set[1],'Council':set[0],'Barrow':set[3]}

    grouped =  VWC_df_late.groupby(['site','sar_plot'])

    for i in range(0,len(depth)):
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

def main():
    #read in all cropped files
    TL = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_6_12_20_TL.csv'
    KG ='Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_6_12_20_KG.csv'
    CN = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_6_12_20_CN.csv'
    BW ='Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_6_12_20_BW.csv'
    ABoVE_data = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_all_sites.csv'

    #uncomment for making combined file
    '''
    sites = [TL,KG,CN,BW]
    list = []
    for i in sites:
        df = pd.read_csv(i)
        list.append(df)

    df = pd.concat(list,ignore_index=True)
    print(df)
    df.to_csv('Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_all_sites.csv')
    '''
    VWC_data= 'Z:/AKSeward/Data/Excel/SAR/discrete_data_final_20190306.csv'
    #compare_discrete_vs_hydrosense(ABoVE_data,VWC_data,depth=['6','12','20'],combine=True)

    df1 = pd.read_csv(ABoVE_data,sep=',')

    df2 = pd.read_csv(VWC_data,sep=',')
    df1['point'] = [(x, y) for x,y in zip(df1['lat'], df1['lon'])]
    df2['point'] = [(x, y) for x,y in zip(df2['lat'], df2['lon'])]
    df2['above_lat'],df2['above_lon'] = df1.loc[[closest_point_index(x, list(df1['point'])) for x in df2['point']],['lat','lon']]
    df2.to_csv('Z:/JDann/Documents/Documents/Julian_Python/SAR_Programs_20181003/test.csv',sep=',')

    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == "__main__":
    main()
