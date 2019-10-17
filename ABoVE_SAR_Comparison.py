import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib.markers import MarkerStyle
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

def dist(lat1, long1, lat2, long2):
    '''distance calculator on plane'''
    return np.sqrt((lat1-lat2)**2.0+ (long1-long2)**2.0)

def find_SAR_pixel(df,lat, long,export_rows):
    '''function that grabs the closest SAR pixel to the hydrosense measurement'''
    distances = df.apply(
        lambda row: dist(lat, long, row['lat'], row['lon']),
        axis=1)
    return df.loc[distances.idxmin(), export_rows]

def SAR_comparison(ABoVE_file,VWC_file,depth='20',combine=False):
    '''Function meant to read in file containing avg in situ hydrosense values and SAR raster statistics for comparison in plots'''
    #use ggplot
    matplotlib.style.use('ggplot')

    #read in data
    ABoVE_df = pd.read_csv(ABoVE_file,sep=',')
    VWC_df = pd.read_csv(VWC_file,sep=',')

    #make date into datetime object
    VWC_df['Date'] = pd.to_datetime(VWC_df['Date'],errors='coerce')

    #take only late dates
    VWC_df_late = VWC_df[(VWC_df['Date']>pd.Timestamp(2017,7,1)) & (VWC_df['site'] != 'Barrow')]

    #matching and merging ABoVE data with the hydrosense based on whether it is per flag or per SAR Plot
    if combine == False:
        VWC_df_late['int_vwc_6'] = VWC_df_late.apply(lambda row: find_SAR_pixel(ABoVE_df,row['lat'],row['lon'],'int_vwc_6'),axis=1)
        VWC_df_late['int_vwc_12'] = VWC_df_late.apply(lambda row: find_SAR_pixel(ABoVE_df,row['lat'],row['lon'],'int_vwc_12'),axis=1)
        VWC_df_late['int_vwc_20'] = VWC_df_late.apply(lambda row: find_SAR_pixel(ABoVE_df,row['lat'],row['lon'],'int_vwc_20'),axis=1)
    else:
        #group the ABoVE data by SAR_Plot
        gb_SAR = ABoVE_df.groupby('SAR_Plot')
        ABoVE_mean_gb = (gb_SAR[['int_vwc_6','int_vwc_12','int_vwc_20']].mean())
        ABoVE_std_gb = (gb_SAR[['int_vwc_6','int_vwc_12','int_vwc_20']].std())
        ABoVE_std_gb.rename(columns={'int_vwc_6':'int_std_6','int_vwc_12':'int_std_12','int_vwc_20':'int_std_20'},inplace=True)
        ABoVE_data = ABoVE_mean_gb.merge(ABoVE_std_gb,how='left',on='SAR_Plot')


        #merge the dataframes by SAR_plot
        #VWC_df_late.merge(ABoVE_df,left_on='sar_plot',right_on='SAR_Plot')
        gb_VWC= VWC_df_late.groupby('sar_plot').agg({'site':'first','avg_6':np.mean,'avg_12':np.mean,'avg_20':np.mean,'std_6':np.mean,'std_12':np.mean,'std_20':np.mean})
        print(gb_VWC)

        VWC_df_late = gb_VWC.join(ABoVE_data)
        print(list(gb_VWC))
        #x =0/0
    #make single plot or all three depths in one
    if len(depth) == 1:
        #use column name for hydrosense specification
        col_names = 'avg_'+depth
        col_err = 'std_'+depth
        symbol='o'
        title='Comparing Hydrosense II Soil Moisture with derived ABoVE SAR Soil Moisture ('+depth+'cm)'
        #divide each column by 100.0
        VWC_df_late[col_name[0]] = VWC_df_late[col_name[0]]/100.0
        VWC_df_late[col_err[0]] = VWC_df_late[col_err[0]]/100.0
        save_name = 'Z:/AKSeward/2017_SAR/CJW_Phase_3_Poster/figures/2017_aug_mv1_'+depth+'cm.png'
    if len(depth)>1:
        #defining columns
        col_name = ['avg_'+depth[2],'avg_'+depth[1],'avg_'+depth[0]]
        y_col = ['int_vwc_'+depth[2],'int_vwc_'+depth[1],'int_vwc_'+depth[0]]
        if combine == True:
            y_col_err = ['int_std_'+depth[2],'int_std_'+depth[1],'int_std_'+depth[0]]
        else:
            y_col_err = []
        col_err = ['std_'+depth[2],'std_'+depth[1],'std_'+depth[0]]
        symbol =['o','s','^']
        title = ' In-Situ Soil Moisture vs. derived ABoVE SAR Soil Moisture'
        alpha = [1,0.6,0.2]
        #divide each column by 100.0
        VWC_df_late[col_name] = VWC_df_late[col_name]/100.0
        VWC_df_late[col_err] = VWC_df_late[col_err]/100.0
        fill_style = ['top','full','bottom',]
        edge_colors =['yellow','purple','orange']
        size = [650,250,100]
        if combine == False:
            save_name = 'Z:/AKSeward/2017_SAR/CJW_Phase_3_Poster/figures/2017_aug_final_flags_20190502.png'
        else:
            save_name = 'Z:/AKSeward/2017_SAR/CJW_Phase_3_Poster/figures/2017_aug_final_plot_avg_20190502.png'
    fig,ax=plt.subplots(figsize=(15,10))
    #colors = {'Teller':'royalblue','Kougarok':'salmon','Council':'seagreen'}
    set = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628" ,"#F781BF", "#999999"]

    colors = {'Teller':set[2],'Kougarok':set[1],'Council':set[0]}

    if combine == True:
        #print(list(VWC_df_late))
        grouped =  VWC_df_late.groupby('site')
    else:
        grouped =  VWC_df_late.groupby('site')

    for i in range(0,len(depth)):
        for key,group in grouped:
            print(key)
            print(col_name,y_col,colors,size,col_err,y_col_err)
            #print(group[[col_name[i],y_col[i],'flag','sar_plot']])

            group.plot(ax=ax,x=col_name[i],y=y_col[i],xerr=col_err[i],yerr=y_col_err[i],kind='scatter',s=size[i],color=colors[key],edgecolors='k',linewidths=1.5 ,marker=MarkerStyle(symbol[i]))#,fillstyle=fill_style[i]),edgecolors=edge_colors[i])#,edgecolors=edge_colors[i],linewidths=2)
            plt.rcParams['lines.linewidth'] = 0.5

    plt.xlim(0,1)
    plt.ylim(0,0.4)

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
    #plt.savefig(save_name,dpi=500)
    plt.show()
    #plt.close()


def main():
    ABoVE_data = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/all_int_6_12_20_clipped_wgs_1984_V2.csv'
    VWC_data= 'Z:/AKSeward/Data/Excel/SAR/discrete_data_final_20190306.csv'
    discrete_above = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_6_12_20_layers.csv'

    compare_discrete_vs_hydrosense(discrete_above,VWC_data)
    #discrete plots


    #SAR_comparison(ABoVE_data,VWC_data,depth=['6','12','20'],combine=False)

if __name__ == "__main__":
    main()
