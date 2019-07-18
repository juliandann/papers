import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data
#import arcpy,arcinfo
#arcpy.CheckOutExtension("Spatial")
#from arcpy import env
#arcpy.env.overwriteOutput = True
#from scipy.stats import *
from matplotlib.lines import Line2D
import matplotlib
#from data.Python import pillow # register PIL
from PIL import Image
from PIL import PngImagePlugin
from matplotlib.markers import MarkerStyle
from matplotlib.legend import Legend

def get_flag():
    '''program meant to grab a picture of a flag in order to put on SAR plots'''

    path = 'Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/red_flag.png'
    im = plt.imread(path)
    return im
def avg_per_flag(vwc_may_fname,vwc_aug_fname,b_vwc_may_fname,b_vwc_aug_fname):

    '''program written to read in data and average by SAR plot

    Return: dictionary with averages at each site for each depth'''
    #reading in files and creating dataframes
    vwc_1 = pd.read_csv(vwc_may_fname,sep=',')
    b_vwc_1 = pd.read_csv(b_vwc_may_fname,sep=',')
    vwc_1 = pd.concat([vwc_1,b_vwc_1],sort=True)

    vwc_2 = pd.read_csv(vwc_aug_fname,sep=',')
    b_vwc_2 = pd.read_csv(b_vwc_aug_fname,sep=',')
    vwc_2 = pd.concat([vwc_2,b_vwc_2],sort=True)

    vwc_1['TimeStamp'] = pd.to_datetime(vwc_1['TimeStamp'],format='%m/%d/%Y %H:%M')
    vwc_2['TimeStamp'] = pd.to_datetime(vwc_2['TimeStamp'],format='%m/%d/%Y %H:%M')
    vwc_com = [vwc_1,vwc_2]


    sar_plotvwc_names = ['TL_SAR_4_','TL_SAR_8','TL_SAR_7','TL_SAR_41','KG_SAR_2','KG_SAR_5','KG_SAR_6','CN_SAR_1','CN_SAR_2','Barrow_SAR_HC','Barrow_SAR_LC','Barrow_SAR_FC']

    flag_arr = ['Flag_1','Flag_2','Flag_3','Flag_4','Flag_5','Flag_6']

    #setting up for a for loop to go through each site and transect
    site,lon,lat= [],[],[]
    sar_plot = []
    transect = []
    date = []
    title = []
    avg_6 =[]
    avg_12 = []
    avg_20 = []
    std_6 =[]
    std_12 = []
    std_20 = []
    flag = []
    avg_td = []
    std_td = []
    for i in range(0,len(sar_plotvwc_names)):
        for k in range(0,6):
            for h in range(0,2):
                vwc_data = vwc_com[h]
                #print vwc_data[['Measurement_ID','TimeStamp']]
                sub_vwc_data = vwc_data[(vwc_data['Measurement_ID'].str.contains(sar_plotvwc_names[i])) & (vwc_data['SAR_Plot_Flag'] == flag_arr[k])]

                if (sub_vwc_data.empty == False):

                    #separating by depth
                    sub_vwc_6 = sub_vwc_data[sub_vwc_data['VWC_Measurement_Depth'] == 6]
                    sub_vwc_12 = sub_vwc_data[sub_vwc_data['VWC_Measurement_Depth'] == 12]
                    sub_vwc_20 = sub_vwc_data[sub_vwc_data['VWC_Measurement_Depth'] == 20]

                    #averaging and appending soil moisture
                    if (sub_vwc_6.empty == False):
                        avg_6.append(sub_vwc_6['VWC'].mean())
                        std_6.append(sub_vwc_6['VWC'].std())
                    else:
                        avg_6.append(-9999)
                        std_6.append(-9999)

                    if (sub_vwc_12.empty == False):
                        avg_12.append(sub_vwc_12['VWC'].mean())
                        std_12.append(sub_vwc_12['VWC'].std())
                    else:
                        avg_12.append(-9999)
                        std_12.append(-9999)
                    if (sub_vwc_20.empty == False):
                        avg_20.append(sub_vwc_20['VWC'].mean())
                        std_20.append(sub_vwc_20['VWC'].std())
                    else:
                        avg_20.append(-9999)
                        std_20.append(-9999)

                    site.append(sub_vwc_data['Locale'].iloc[0])
                    sar_plot.append(sub_vwc_data['SAR_Plot'].iloc[0])
                    date.append((sub_vwc_data['TimeStamp'].iloc[0]).strftime("%Y-%m-%d"))
                    flag.append(sub_vwc_data['SAR_Plot_Flag'].iloc[0])
                    lon.append(sub_vwc_data['Longitude'].mean())
                    lat.append(sub_vwc_data['Latitude'].mean())
                else:
                    avg_6.append(-9999)
                    avg_12.append(-9999)
                    avg_20.append(-9999)
                    std_6.append(-9999)
                    std_12.append(-9999)
                    std_20.append(-9999)

                    site.append(-9999)
                    sar_plot.append(sar_plotvwc_names[i])
                    if h == 0:
                        dated = 'early season'
                    else:
                        dated = 'late season'
                    date.append(dated)
                    flag.append(flag_arr[k])
                    lon.append(-9999)
                    lat.append(-9999)


    #making dictionary
    data = {'site':site,'flag':flag,'sar_plot':sar_plot,'Date':date,'avg_6':avg_6,'avg_12':avg_12,'avg_20':avg_20,'std_6':std_6,'std_12':std_12,'std_20':std_20,'lat':lat,'lon':lon}
    return data
def combine_data_with_concavity_info(data,concav_fname_1m,concav_fname_2m):
    '''Purpose:combines concavity information with the data at sar plots

    Inputs: data dictionary, excel files of 1m and 2m resolution concavity

    Outputs: merged data and concavity
    '''
    data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.iteritems() ]))
    data.to_csv('Z:/')
    concav_1 = pd.read_excel(concav_fname_1m)
    concav_2 = pd.read_excel(concav_fname_2m)

    mergedstuff = pd.merge(data,concav_1,on=['sar_plot','flag'],how='left')
    print mergedstuff
    print mergedstuff[mergedstuff.grid_code.notnull()]
def plotting_mechanism(fig,ax,td,vwc,sar_plot,date,transect,title,catcol='Thaw_Depth_Comments'):
    '''
    Purpose:

    Input:

    Output:

    '''
    #print td[['Measurement_ID','Date']],vwc[['Measurement_ID','TimeStamp']]
    #plotting thaw depth first
    td.loc[td[catcol].str.contains('FT'),catcol] = 'FT'
    td.loc[td[catcol].str.contains('PR'),catcol] = 'PR'
    td.loc[(td[catcol].str.contains('R') & ~td[catcol].str.contains('PR')),catcol] = 'R'
    td.loc[td[catcol].str.contains('WT'),catcol] = 'WT'
    categories = np.unique(td[catcol])
    str_arr = ['SN','SH','WT','IC']
    colors1 = []
    for i in range(0,len(categories)):
        if ('R' in categories[i]) and ('PR' not in categories[i]):
            colors1.append('r')
            categories[i] = 'R'
        if 'FT' in categories[i]:
            colors1.append('b')
            categories[i] = 'FT'
        if 'PR' in categories[i]:
            colors1.append('purple')
            categories[i] = 'PR'
        '''
        if ('SN' in categories[i]) and ('PR' not in categories[i]):
            colors1.append('k')
        if ('SH' in categories[i]) and ('PR' not in categories[i]):
            colors1.append('k')
        if ('WT' in categories[i]):
            colors1.append('k')
        '''
        if any(x in categories[i] for x in str_arr ):
            colors1.append('k')

    colordict = dict(zip(categories, colors1))
    #print categories,colordict
    td["Color"] = td[catcol].apply(lambda x: colordict[x])
    #plt.scatter(td['Distance'],-1.0*td['Thaw_Depth'],c=td.Color,s=20.0,edgecolors='face')

    #making lines to the thaw depth points
    tot_color = np.unique(td['Color'])

    colors= []
    for j in range(0,len(tot_color)):
        if 'b' in tot_color:
            blue = td[td.Color == 'b']
            colors.append(blue)
        if 'r' in tot_color:
            red = td[td.Color == 'r']
            colors.append(red)
        if 'purple' in tot_color:
            purple = td[td.Color == 'purple']
            colors.append(purple)
        if 'k' in tot_color:
            black = td[td.Color == 'k']
            colors.append(black)


    for i in range(0,len(colors)):
        colors[i] = colors[i][colors[i].Thaw_Depth != -9999]
        if colors[i].empty:
            break
        else:
            markerline, stemlines, baseline = ax.stem(colors[i]['Distance'],-1.0*colors[i]['Thaw_Depth'])
            #extend to end of plot
            baseline.set_xdata([0,60])


            plt.setp(baseline, color='saddlebrown',linewidth=7,axes=ax)
            if colors[i].Color.iloc[0] == 'r' :
                plt.setp(stemlines,linestyle='--', color='gray',axes=ax)
                plt.setp(markerline,color='',marker='',axes=ax)
            if colors[i].Color.iloc[0] == 'b' :
                plt.setp(stemlines,linestyle='-', color='k',axes=ax)
                plt.setp(markerline,color='k',marker='_',axes=ax)
            if colors[i].Color.iloc[0] == 'purple' :
                plt.setp(stemlines,linestyle='-', color='k',axes=ax)
                plt.setp(markerline,color='k',marker=7,axes=ax)
            if colors[i].Color.iloc[0] == 'k' :
                plt.setp(stemlines,linestyle='--', color='gray',axes=ax)
                plt.setp(markerline,color='k',marker='o',axes=ax)
        #making distances for flags

    flags = np.unique(vwc['SAR_Plot_Flag'])
    dist_flags = []
    for i in range(0,len(flags)):
        if '1' in flags[i] or '4' in flags[i]:
            dist_flags.append(15.0)
        if '2' in flags[i] or '5' in flags[i]:
            dist_flags.append(30.0)
        if '3' in flags[i] or '6' in flags[i]:
            dist_flags.append(45.0)
    distdict = dict(zip(flags, dist_flags))
    vwc["Distance"] = vwc['SAR_Plot_Flag'].apply(lambda x: distdict[x])
    if 'TL1' in transect:
        avg_1 = vwc[vwc['SAR_Plot_Flag'] == 'Flag_1']
        avg_2 = vwc[vwc['SAR_Plot_Flag'] == 'Flag_2']
        avg_3 = vwc[vwc['SAR_Plot_Flag'] == 'Flag_3']
        avgs = [avg_1,avg_2,avg_3]
    if 'TL2' in transect:
        avg_1 = vwc[vwc['SAR_Plot_Flag'] == 'Flag_4']
        avg_2 = vwc[vwc['SAR_Plot_Flag'] == 'Flag_5']
        avg_3 = vwc[vwc['SAR_Plot_Flag'] == 'Flag_6']
        avgs = [avg_1,avg_2,avg_3]

    depth_6cm_avg = []
    depth_12cm_avg = []
    depth_20cm_avg = []
    for i in range(0,len(avgs)):
        cm_6 = avgs[i][avgs[i]['VWC_Measurement_Depth'] == 6]
        cm_12 = avgs[i][avgs[i]['VWC_Measurement_Depth'] == 12]
        cm_20 = avgs[i][avgs[i]['VWC_Measurement_Depth'] == 20]

        depth_6cm_avg.append(np.nanmean(cm_6['VWC']))
        depth_12cm_avg.append(np.nanmean(cm_12['VWC']))
        depth_20cm_avg.append(np.nanmean(cm_20['VWC']))

    vwc_arr = [depth_6cm_avg,depth_12cm_avg,depth_20cm_avg]
    dist_flag = [15,30,45]

    cm = plt.cm.get_cmap('RdBu')

    for m in range(0,len(vwc_arr)):
        if m == 0:
            depth_arr = [-6.0,-6.0,-6.0]
        if m == 1:
            depth_arr = [-12.0,-12.0,-12.0]
        if m == 2:
            depth_arr = [-20.0,-20.0,-20.0]
        cp = ax.scatter(dist_flag,depth_arr, c = vwc_arr[m],s=150,marker='p',vmin=0,vmax=100, cmap = cm,zorder=10,alpha=0.9)

    #adding flags
    im = get_flag()
    extent1 = [15,18,3,19]
    extent2 = [30,33,3,19]
    extent3 = [45,48,3,19]
    text1 = (16,13)
    text2 = (31,13)
    text3 = (46,13)
    text = [text1,text2,text3]
    textx,texty = zip(*text)
    extent = [extent1,extent2,extent3]
    str_flags = ['Flag_1','Flag_2','Flag_3']
    for i in range(0,3):
        ax.imshow(im,extent=extent[i],aspect='auto')
        if any(x in td['Measurement_ID'] for x in str_flags):
            ax.text(textx[i],texty[i],str(i+1),fontsize = 10,fontstyle = 'oblique',color='white')
        else:
            ax.text(textx[i],texty[i],str(i+4),fontsize = 10,fontstyle = 'oblique',color='white')

    ax.set_ylim(-120,20)
    ax.set_xlim(-5,65)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (cm)')
    ax.set_title(title)
    return cp
def td_vwc_plotter(td_fname,vwc_may_fname,vwc_aug_fname):
    #reading in files
    td = pd.read_csv(td_fname,sep=',')
    vwc_1 = pd.read_csv(vwc_may_fname,sep=',')
    vwc_2 = pd.read_csv(vwc_aug_fname,sep=',')
    vwc_com = [vwc_1,vwc_2]
    #making column date_time
    td['Date'] = pd.to_datetime(td['Date'],format='%m/%d/%Y')

    #separating thaw depth into early and late season
    td_1 = td[td.Date < pd.Timestamp(2017,8,1)]
    td_2 = td[td.Date > pd.Timestamp(2017,8,1)]
    td_com = [td_1,td_2]

    #is this seward or barrow? ONLY FOR thaw depth
    if 'seward_' in vwc_may_fname:
        sar_plot_names = ['TL_SAR_4-','TL_SAR_8-','TL_SAR_7-','TL_SAR_41-','KG_SAR_2-','KG_SAR_5-','KG_SAR_6-','CN_SAR_1-','CN_SAR_2-']
        sar_plotvwc_names = ['TL_SAR_4_','TL_SAR_8_','TL_SAR_7_','TL_SAR_41_','KG_SAR_2_','KG_SAR_5_','KG_SAR_6_','CN_SAR_1_','CN_SAR_2_']
    if 'barrow_' in vwc_may_fname:
        sar_plot_names = ['BW_SAR_HC-','BW_SAR_LC-','BW_SAR_FC-']
        sar_plotvwc_names = ['Barrow_SAR_HC_','Barrow_SAR_LC_','Barrow_SAR_FC_']

    #transect names for thaw depth measurements
    tran_names = ['TL1','TL2']
    tran_names_vwc = ['T1_','T2_']

    #setting up for a for loop to go through each site and transect
    sar_plot = []
    transect = []
    date = []
    title = []

    for i in range(0,len(sar_plot_names)):
        for j in range(0,len(tran_names)):
            fig,ax=plt.subplots(1,2,figsize=(20,5))
            for h in range(0,2):
                td_data = td_com[h]
                vwc_data = vwc_com[h]
                #print vwc_data[['Measurement_ID','TimeStamp']]

                sub_vwc_data = vwc_data[(vwc_data['Measurement_ID'].str.contains(sar_plotvwc_names[i])) & (vwc_data['Measurement_ID'].str.contains(tran_names_vwc[j]))]
                sub_td_data = td_data[(td_data['Measurement_ID'].str.contains(sar_plot_names[i])) & (td_data['Measurement_ID'].str.contains(tran_names[j]))]

                sar_plot.append(sub_td_data['SAR_Plot'].iloc[0])
                transect.append(sub_td_data['SAR_Plot_Transect'].iloc[0])
                date.append((sub_td_data['Date'].iloc[0]).strftime("%Y-%m-%d"))
                #print date[-1],sar_plot[-1],transect[-1]
                title=str(sar_plot[-1])+str(transect[-1])+' on '+str(date[-1])

                leg = plotting_mechanism(fig,ax[h],sub_td_data,sub_vwc_data,sar_plot[-1],date[-1],transect[-1],title)
            cbar_ax = fig.add_axes([0.925, 0.15, 0.02, 0.7])
            fig.subplots_adjust(right=0.9)
            fig.colorbar(leg,label='Volumetric Water Content (%)',cax=cbar_ax)
            fig.savefig('Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/'+str(sar_plot[-1])+str(transect[-1])+'.png')
            plt.close('all')


def avg_by_plot(td_fname,vwc_may_fname,vwc_aug_fname,b_td_fname,b_vwc_may_fname,b_vwc_aug_fname):
    #reading in files for seward
    td = pd.read_csv(td_fname,sep=',')
    b_td = pd.read_csv(b_td_fname,sep=',')
    td = pd.concat([td,b_td],sort=True)

    vwc_1 = pd.read_csv(vwc_may_fname,sep=',')
    b_vwc_1 = pd.read_csv(b_vwc_may_fname,sep=',')
    vwc_1 = pd.concat([vwc_1,b_vwc_1],sort=True)

    vwc_2 = pd.read_csv(vwc_aug_fname,sep=',')
    b_vwc_2 = pd.read_csv(b_vwc_aug_fname,sep=',')
    vwc_2 = pd.concat([vwc_2,b_vwc_2],sort=True)

    vwc_1['TimeStamp'] = pd.to_datetime(vwc_1['TimeStamp'],format='%m/%d/%Y %H:%M')
    vwc_2['TimeStamp'] = pd.to_datetime(vwc_2['TimeStamp'],format='%m/%d/%Y %H:%M')
    vwc_com = [vwc_1,vwc_2]

    #making column date_time
    td['Date'] = pd.to_datetime(td['Date'],format='%m/%d/%Y')

    #separating thaw depth into early and late season
    td_1 = td[td.Date < pd.Timestamp(2017,8,1)]
    td_2 = td[td.Date > pd.Timestamp(2017,8,1)]
    td_com = [td_1,td_2]

    #is this seward or barrow? ONLY FOR thaw depth
    if 'seward_' in vwc_may_fname:
        sar_plot_names = ['TL_SAR_4-','TL_SAR_8-','TL_SAR_7-','TL_SAR_41-','KG_SAR_2-','KG_SAR_5-','KG_SAR_6-','CN_SAR_1-','CN_SAR_2-','BW_SAR_HC-','BW_SAR_LC-','BW_SAR_FC-']
        sar_plotvwc_names = ['TL_SAR_4_','TL_SAR_8_','TL_SAR_7_','TL_SAR_41_','KG_SAR_2_','KG_SAR_5_','KG_SAR_6_','CN_SAR_1_','CN_SAR_2_','Barrow_SAR_HC_','Barrow_SAR_LC_','Barrow_SAR_FC_']
    '''
    if 'barrow_' in vwc_may_fname:
        sar_plot_names = ['BW_SAR_HC-','BW_SAR_LC-','BW_SAR_FC-']
        sar_plotvwc_names = ['Barrow_SAR_HC_','Barrow_SAR_LC_','Barrow_SAR_FC_']
    '''

    #setting up for a for loop to go through each site and transect
    site,lon,lat= [],[],[]
    sar_plot = []
    transect = []
    date = []
    title = []
    avg_6 =[]
    avg_12 = []
    avg_20 = []
    std_6 =[]
    std_12 = []
    std_20 = []

    avg_td = []
    std_td = []
    for i in range(0,len(sar_plot_names)):

        for h in range(0,2):
            td_data = td_com[h]
            vwc_data = vwc_com[h]
            #print vwc_data[['Measurement_ID','TimeStamp']]

            sub_vwc_data = vwc_data[(vwc_data['Measurement_ID'].str.contains(sar_plotvwc_names[i]))]
            sub_td_data = td_data[(td_data['Measurement_ID'].str.contains(sar_plot_names[i])) & (td_data['Thaw_Depth_Comments'].str.contains('FT'))]

            if (sub_td_data.empty == False):
                #thaw_depth:
                avg_td.append(sub_td_data['Thaw_Depth'].mean())
                std_td.append(sub_td_data['Thaw_Depth'].std())

            else:
                avg_td.append(float('NaN'))
                std_td.append(float('NaN'))
            if (sub_vwc_data.empty == False):
                #separating by depth
                sub_vwc_6 = sub_vwc_data[sub_vwc_data['VWC_Measurement_Depth'] == 6]
                sub_vwc_12 = sub_vwc_data[sub_vwc_data['VWC_Measurement_Depth'] == 12]
                sub_vwc_20 = sub_vwc_data[sub_vwc_data['VWC_Measurement_Depth'] == 20]

                #averaging and appending soil moisture
                avg_6.append(sub_vwc_6['VWC'].mean())
                avg_12.append(sub_vwc_12['VWC'].mean())
                avg_20.append(sub_vwc_20['VWC'].mean())
                std_6.append(sub_vwc_6['VWC'].std())
                std_12.append(sub_vwc_12['VWC'].std())
                std_20.append(sub_vwc_20['VWC'].std())
            else:
                avg_6.append(float('NaN'))
                avg_12.append(float('NaN'))
                avg_20.append(float('NaN'))
                std_6.append(float('NaN'))
                std_12.append(float('NaN'))
                std_20.append(float('NaN'))

            if sub_td_data.empty == False:
                site.append(sub_td_data['Locale'].iloc[0])
                sar_plot.append(sub_td_data['SAR_Plot'].iloc[0])
                date.append((sub_td_data['Date'].iloc[0]).strftime("%Y-%m-%d"))
                title=str(sar_plot[-1])+' on '+str(date[-1])
                lon.append(sub_vwc_data['Longitude'].mean())
                lat.append(sub_vwc_data['Latitude'].mean())

            if (sub_td_data.empty == True) & (sub_vwc_data.empty == False):
                site.append(sub_vwc_data['Locale'].iloc[0])
                lon.append(sub_vwc_data['Longitude'].mean())
                lat.append(sub_vwc_data['Latitude'].mean())
                sar_plot.append(sub_vwc_data['SAR_Plot'].iloc[0])
                date.append((sub_vwc_data['TimeStamp'].iloc[0]).strftime("%Y-%m-%d"))

    #making dictionary
    data = {'site':site,'sar_plot':sar_plot,'Date':date,'avg_6':avg_6,'avg_12':avg_12,'avg_20':avg_20,'std_6':std_6,'std_12':std_12,'std_20':std_20,'avg_td':avg_td,'std_td':std_td,'lon':lon,'lat':lat}
    return data
def avg_plots(data,slope_file):
    df = pd.DataFrame.from_dict(data)
    #print data['sar_plot'],len(data['sar_plot'])
    slopes = pd.read_csv(slope_file,sep=',')

    df.merge(slopes,how='left',left_on='sar_plot',right_on='SAR_Name')
    df_com = pd.merge(df,slopes,left_on='sar_plot',right_on='SAR_Name')

    ylabel = 'Thaw Depth (cm)'
    xlabel_2 = 'Aspect (deg.)'
    #ylabel = 'Volumetric Water Content (%)'
    xlabel = 'Slope (deg.)'
    #plot_y_axis = ['avg_6','avg_12','avg_20']
    #plot_yerror = ['std_6','std_12','std_20']

    plot_y_axis = ['avg_td']
    plot_yerror = ['std_td']

    plot_x_axis = ['Avg_slope','Avg_aspect']
    plot_xerror = ['STD_slope','STD_aspect']
    ylim =(0,150)
    xlim = (-2,16)
    #xlim_2 = (-2,16)
    xlim_2 = (0,360)
    #COLORING plots
    fig, ax = plt.subplots(len(plot_x_axis),len(plot_y_axis),figsize=(15,10))
    categories = np.unique(df_com['site'])

    for i in range(0,len(categories)):
        df_temp_1 = df_com[(df_com['site'] == categories[i]) & (pd.to_datetime(df_com['Date']) < pd.Timestamp(2017,8,1))]
        df_temp_2 = df_com[(df_com['site'] == categories[i]) & (pd.to_datetime(df_com['Date']) > pd.Timestamp(2017,8,1))]
        df_arr = [df_temp_1,df_temp_2]
        for j in range(0,2):
            if categories[i] == 'Teller':
                color='b'
            if categories[i] == 'Kougarok':
                color='g'
            if categories[i] == 'Council':
                color='r'
            if categories[i] == 'Barrow':
                color='purple'
            for l in range(0,len(plot_x_axis)):
                for k in range(0,len(plot_y_axis)):
                    print i,j,l,k
                    if len(plot_x_axis) == 1:
                        t = k
                    elif len(plot_y_axis) == 1:
                        t = l
                    else:
                        t=(l,k)
                    if j == 0:
                        ax[t].errorbar(df_arr[j][plot_x_axis[l]],df_arr[j][plot_y_axis[k]],xerr=df_arr[j][plot_xerror[l]],yerr=df_arr[j][plot_yerror[k]], linestyle="None",c=color,label=categories[i],markersize=7,marker='o',markeredgecolor=color)
                        ax[t].set_ylim(ylim)
                        ax[t].set_ylabel(ylabel)

                        if l == 0:
                            ax[t].set_xlim(xlim)
                            ax[t].set_xlabel(xlabel)
                        else:
                            ax[t].set_xlim(xlim_2)
                            ax[t].set_xlabel(xlabel_2)
                    else:
                        ax[t].errorbar(df_arr[j][plot_x_axis[l]],df_arr[j][plot_y_axis[k]],xerr=df_arr[j][plot_xerror[l]],yerr=df_arr[j][plot_yerror[k]], linestyle="None",c=color,alpha=0.3,markersize=7,marker='d',markeredgecolor=color,label=None)
                        ax[t].set_ylim(ylim)
                        ax[t].set_ylabel(ylabel)
                        if l == 0:
                            ax[t].set_xlim(xlim)
                            ax[t].set_xlabel(xlabel)
                        else:
                            ax[t].set_xlim(xlim_2)
                            ax[t].set_xlabel(xlabel_2)

            ax[0].set_title('Thaw Depth vs. Slope')
            ax[1].set_title('Thaw Depth vs. Aspect')
            #ax[2].set_title('Hydrosense Measurements at 20cm')
            #ax[1,0].set_title('Hydrosense Measurements at 6cm')
            #ax[1,1].set_title('Hydrosense Measurements at 12cm')
            #ax[1,2].set_title('Hydrosense Measurements at 20cm')

            custom_lines_2 = [Line2D([0],[0],marker='o',color='w',label='Teller',markerfacecolor='b'),
                Line2D([0],[0],marker='o',color='w',label='Kougarok',markerfacecolor='g'),
                Line2D([0],[0],marker='o',color='w',label='Council',markerfacecolor='r'),
                Line2D([0],[0],marker='o',color='w',label='Barrrow',markerfacecolor='purple')]
            leg1 = fig.legend( custom_lines_2,['Teller','Kougarok','Council','Barrow'],loc=1)
            custom_lines = [Line2D([0], [0], color='k', lw=4),
                    Line2D([0], [0], color='k', lw=4,alpha=0.3)]
            fig.legend(custom_lines, ['Early Season','Late Season'],loc=2)
            plt.gca().add_artist(leg1)

    #plt.show()
    plt.savefig('Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/slope_aspect_vs_TD.png')
    plt.close('all')
def make_aspect(sar_shp,dem,name):
    #clipping dem to sar plot
    out_clip = arcpy.Clip_management(dem,"#",'C:/Users/323053/Documents/ArcGIS/scratch/'+name+'.tif',sar_shp,"#","ClippingGeometry","MAINTAIN_EXTENT")

    #make aspects
    out_aspect = arcpy.sa.Aspect(out_clip)
    out_aspect.save('C:/Users/323053/Documents/ArcGIS/scratch/'+str(name)+'_SAR_clip_aspect.tif')
    out_slope = arcpy.sa.Slope(out_clip)
    out_slope.save('C:/Users/323053/Documents/ArcGIS/scratch/'+str(name)+'_SAR_clip_slope.tif')

    #raster to points
    out_pts = arcpy.RasterToPoint_conversion(out_aspect,'C:/Users/323053/Documents/ArcGIS/scratch/'+str(name)+'_SAR_clip_aspect_pts.shp',"VALUE")
    out_pts_slope = arcpy.RasterToPoint_conversion(out_slope,'C:/Users/323053/Documents/ArcGIS/scratch/'+str(name)+'_SAR_clip_slope_pts.shp',"VALUE")

    #read in table of shapefile
    point_val = arcpy.da.TableToNumPyArray(out_pts,'grid_code')
    slope_point_val = arcpy.da.TableToNumPyArray(out_pts_slope,'grid_code')

    #statistics
    avg_asp,std_asp = circ_calc(point_val)
    avg_slp,std_slp = avg_std_calc(slope_point_val)

    #print name,'Aspect: ',avg_asp,std_asp,'Slope: ',avg_slp,std_slp

    return name,avg_asp,std_asp,avg_slp,std_slp
def circ_calc(points):
    points = [item[0] for item in points]
    points_df = {'grid_code':points}
    print len(points_df['grid_code'])
    #std
    std = circstd(np.deg2rad(points_df['grid_code']))
    avg = circmean(np.deg2rad(points_df['grid_code']))


    np.set_printoptions(suppress=True)

    return np.rad2deg(avg),np.rad2deg(std)

def avg_std_calc(points):
    points = [item[0] for item in points]
    points_df = {'grid_code':points}

    return np.mean(points_df['grid_code']),np.std(points_df['grid_code'])

def discrete_calculation(csv_file):
    #read data into pandas file
    data = pd.read_csv(csv_file,sep=',')

    #making column date_time
    data['Date'] = pd.to_datetime(data['Date'],format='%Y-%m-%d',errors='ignore')

    data['davg_6_12'] = 100.0 * ((data['avg_12']/100.0 * 12.0) - (data['avg_6']/100.0 * 6.0))/6.0
    data['davg_12_20'] = 100.0 * ((data['avg_20']/100.0 * 20.0) - (data['avg_12']/100.0 * 12.0))/8.0
    data['davg_6_20'] = 100.0 * ((data['avg_20']/100.0 * 20.0) - (data['avg_6']/100.0 * 6.0))/14.0

    #making into dictionary then pandas column
    #data = {'site':site,'lon':,'sar_plot':sar_plot,'Date':date,'avg_6':avg_6,'std_6':std_6,'davg_6_12':davg_6_12,'dstd_6_12':dstd_6_12, 'davg_12_20':davg_12_20,'dstd_12_20':dstd_12_20,'davg_6_20':davg_6_20,'dstd_6_20':dstd_6_20}
    #data = pd.DataFrame.from_dict(data)
    return data

def discrete_plotting(csv_file,slope_file):

    #read data into pandas file
    data = pd.read_csv(csv_file,sep=',')
    slopes = pd.read_csv(slope_file,sep=',')

    #making column date_time
    data['Date'] = pd.to_datetime(data['Date'],format='%Y-%m-%d',errors='ignore')
    #print data['Date']
    sar_plotvwc_names = ['TL_SAR_4','TL_SAR_8','TL_SAR_7','TL_SAR_41','KG_SAR_2','KG_SAR_5','KG_SAR_6','CN_SAR_1','CN_SAR_2','BW_SAR_HC','BW_SAR_LC','BW_SAR_FC']

    site = []
    sar_plot =[]
    date = []
    avg_6 = []
    davg_6_12 =[]
    davg_12_20 = []
    davg_6_20 = []
    std_6 = []
    dstd_6_12 =[]
    dstd_12_20 = []
    dstd_6_20 = []

    for i in range(0,len(sar_plotvwc_names)):
        data_temp_1 = data[(data['sar_plot'] == sar_plotvwc_names[i]) & (pd.to_datetime(data['Date']) < pd.Timestamp(2017,8,1))]
        data_temp_2 = data[(data['sar_plot'] == sar_plotvwc_names[i]) & (pd.to_datetime(data['Date']) > pd.Timestamp(2017,8,1))]
        data_arr = [data_temp_1,data_temp_2]
        for j in range(0,2):

            #replace -9999 with nans
            data_arr[j] = data_arr[j].replace(float(-9999),np.nan)

            #putting into arrays
            site.append(data_arr[j]['site'].iloc[0])
            sar_plot.append(data_arr[j]['sar_plot'].iloc[0])
            date.append(data_arr[j]['Date'].iloc[0])

            avg_6.append(data_arr[j]['avg_6'].mean())
            std_6.append(data_arr[j]['std_6'].sum()/len(data_arr[j]['std_6']))

            davg_6_12.append(data_arr[j]['davg_6_12'].mean())
            dstd_6_12.append(data_arr[j]['derr_6_12'].sum()/len(data_arr[j]['derr_6_12']))

            davg_12_20.append(data_arr[j]['davg_12_20'].mean())
            dstd_12_20.append(data_arr[j]['derr_12_20'].sum()/len(data_arr[j]['derr_12_20']))

            davg_6_20.append(data_arr[j]['davg_6_20'].mean())
            dstd_6_20.append(data_arr[j]['derr_6_20'].sum()/len(data_arr[j]['derr_6_20']))


    #making into dictionary then pandas column
    data = {'site':site,'sar_plot':sar_plot,'Date':date,'avg_6':avg_6,'std_6':std_6,'davg_6_12':davg_6_12,'dstd_6_12':dstd_6_12, 'davg_12_20':davg_12_20,'dstd_12_20':dstd_12_20,'davg_6_20':davg_6_20,'dstd_6_20':dstd_6_20}
    data = pd.DataFrame.from_dict(data)

    #merging slope with other information
    data.merge(slopes,how='left',left_on='sar_plot',right_on='SAR_Name')
    data_com = pd.merge(data,slopes,left_on='sar_plot',right_on='SAR_Name')

    ylabel = 'Volumetric Moisture Content (%)'
    xlabel_2 = 'Aspect (deg.)'
    #ylabel = 'Volumetric Water Content (%)'
    xlabel = 'Slope (deg.)'
    plot_y_axis = ['avg_6','davg_6_12','davg_12_20','davg_6_20']
    plot_yerror = ['std_6','dstd_6_12','dstd_12_20','dstd_6_20']

    #plot_y_axis = ['avg_td']
    #plot_yerror = ['std_td']


    plot_x_axis = ['Avg_slope','Avg_aspect']
    plot_xerror = ['STD_slope','STD_aspect']
    ylim =(0,160)
    xlim = (-2,16)
    #xlim_2 = (-2,16)
    xlim_2 = (0,360)
    #COLORING plots
    fig, ax = plt.subplots(len(plot_x_axis),len(plot_y_axis),figsize=(15,10))

    #get rid of spaces in column
    data_com.site = data_com.site.str.replace(' ', '')

    categories = np.unique(data_com['site'])

    for i in range(0,len(categories)):
        data_temp_1 = data_com[(data_com['site'] == categories[i]) & (pd.to_datetime(data_com['Date']) < pd.Timestamp(2017,8,1)) ]
        data_temp_2 = data_com[(data_com['site'] == categories[i]) & (pd.to_datetime(data_com['Date']) > pd.Timestamp(2017,8,1)) ]
        data_arr = [data_temp_1,data_temp_2]
        data_pd = pd.concat(data_arr)

        for j in range(0,2):
            if categories[i] == 'Teller':
                color='b'
            if categories[i] == 'Kougarok':
                color='g'
            if categories[i] == 'Council':
                color='r'
            if categories[i] == 'Barrow':
                color='purple'
            for l in range(0,len(plot_x_axis)):
                for k in range(0,len(plot_y_axis)):
                    if len(plot_x_axis) == 1:
                        t = k
                    elif len(plot_y_axis) == 1:
                        t = l
                    else:
                        t=(l,k)
                    if j == 0:
                        ax[t].errorbar(data_arr[j][plot_x_axis[l]],data_arr[j][plot_y_axis[k]],xerr=data_arr[j][plot_xerror[l]],yerr=data_arr[j][plot_yerror[k]], linestyle="None",c=color,label=categories[i],markersize=7,marker='o',markeredgecolor=color)
                        ax[t].set_ylim(ylim)
                        ax[t].set_ylabel(ylabel)

                        if l == 0:
                            ax[t].set_xlim(xlim)
                            ax[t].set_xlabel(xlabel)
                        else:
                            ax[t].set_xlim(xlim_2)
                            ax[t].set_xlabel(xlabel_2)
                    else:
                        ax[t].errorbar(data_arr[j][plot_x_axis[l]],data_arr[j][plot_y_axis[k]],xerr=data_arr[j][plot_xerror[l]],yerr=data_arr[j][plot_yerror[k]], linestyle="None",c=color,alpha=0.3,markersize=7,marker='d',markeredgecolor=color,label=None)
                        ax[t].set_ylim(ylim)
                        ax[t].set_ylabel(ylabel)
                        if l == 0:
                            ax[t].set_xlim(xlim)
                            ax[t].set_xlabel(xlabel)
                        else:
                            ax[t].set_xlim(xlim_2)
                            ax[t].set_xlabel(xlabel_2)

            #ax[0].set_title('Thaw Depth vs. Slope')
            #ax[1].set_title('Thaw Depth vs. Aspect')
            #ax[2].set_title('Hydrosense Measurements at 20cm')
            ax[0,0].set_title('Hydrosense Measurements from 0 - 6cm')
            ax[0,1].set_title('Hydrosense Measurements from 6 - 12cm')
            ax[0,2].set_title('Hydrosense Measurements from 12 - 20cm')
            ax[0,3].set_title('Hydrosense Measurements from 6 - 20cm')

            custom_lines_2 = [Line2D([0],[0],marker='o',color='w',label='Teller',markerfacecolor='b'),
                Line2D([0],[0],marker='o',color='w',label='Kougarok',markerfacecolor='g'),
                Line2D([0],[0],marker='o',color='w',label='Council',markerfacecolor='r'),
                Line2D([0],[0],marker='o',color='w',label='Barrrow',markerfacecolor='purple')]
            leg1 = fig.legend( custom_lines_2,['Teller','Kougarok','Council','Barrow'],loc=1)
            custom_lines = [Line2D([0], [0], color='k', lw=4),
                    Line2D([0], [0], color='k', lw=4,alpha=0.3)]
            fig.legend(custom_lines, ['Early Season','Late Season'],loc=2)
            plt.gca().add_artist(leg1)

    #adding in lines of best first
    data_temp_1 = data_com[(pd.to_datetime(data_com['Date']) < pd.Timestamp(2017,8,1))]
    data_temp_2 = data_com[(pd.to_datetime(data_com['Date']) > pd.Timestamp(2017,8,1))]
    data_arr = [data_temp_1,data_temp_2]
    data_pd = pd.concat(data_arr)
    for j in range(0,2):
        for l in range(0,len(plot_x_axis)):
            for k in range(0,len(plot_y_axis)):

                #setting up plot key
                if len(plot_x_axis) == 1:
                    t = k
                elif len(plot_y_axis) == 1:
                    t = l
                else:
                    t=(l,k)

                if j == 0:

                    data_used = data_arr[j][data_arr[j][plot_y_axis[k]].notnull()]

                    slope, intercept, r_value, p_value, std_err = stats.linregress(data_used[plot_x_axis[l]],data_used[plot_y_axis[k]])
                    if t[0]== 0:
                        x=np.arange(-20,20)
                    else:
                        x=np.arange(0,360)
                    line = slope*x+intercept
                    ax[t].plot( x, line,linestyle='--',color='k',linewidth=2.0)

                else:
                    data_used = data_arr[j][data_arr[j][plot_y_axis[k]].notnull()]
                    slope, intercept, r_value, p_value, std_err = stats.linregress(data_used[plot_x_axis[l]],data_used[plot_y_axis[k]])
                    if t[0]== 0:
                        x=np.arange(-20,20)
                    else:
                        x=np.arange(0,360)
                    line = slope*x+intercept

                    ax[t].plot( x, line,linestyle='--',color='k',alpha=0.3,linewidth=2.0)


    plt.show()
    #plt.savefig('Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/slope_aspect_vs_TD.png')
    #plt.close('all')

#boxplot color function
def setBoxColors(bp):
    setp(bp['boxes'][0], color='gold')
    setp(bp['caps'][0], color='black')
    setp(bp['caps'][1], color='black')
    setp(bp['whiskers'][0], color='black')
    setp(bp['whiskers'][1], color='black')
    setp(bp['fliers'][0], color='black')
    setp(bp['fliers'][1], color='black')
    setp(bp['medians'][0], color='black',linewidth=1.5)

    setp(bp['boxes'][1], color='darkgreen')
    setp(bp['caps'][2], color='black')
    setp(bp['caps'][3], color='black')
    setp(bp['whiskers'][2], color='black')
    setp(bp['whiskers'][3], color='black')
    #setp(bp['fliers'][2], color='darkgreen')
    #setp(bp['fliers'][3], color='darkgreen')
    setp(bp['medians'][1], color='black',linewidth=1.5)

    """
        setp(bp['boxes'][2], color='gold')
        setp(bp['caps'][4], color='gold')
        setp(bp['caps'][5], color='gold')
        setp(bp['whiskers'][4], color='gold')
        setp(bp['whiskers'][5], color='gold')
        setp(bp['fliers'][4], color='gold')
        setp(bp['fliers'][5], color='gold')
        setp(bp['medians'][2], color='gold')

        setp(bp['boxes'][3], color='darkgreen')
        setp(bp['caps'][6], color='darkgreen')
        setp(bp['caps'][7], color='darkgreen')
        setp(bp['whiskers'][6], color='darkgreen')
        setp(bp['whiskers'][7], color='darkgreen')
        setp(bp['fliers'][6], color='darkgreen')
        setp(bp['fliers'][7], color='darkgreen')
        setp(bp['medians'][3], color='darkgreen')

        setp(bp['boxes'][4], color='gold')
        setp(bp['caps'][8], color='gold')
        setp(bp['caps'][9], color='gold')
        setp(bp['whiskers'][8], color='gold')
        setp(bp['whiskers'][9], color='gold')
        setp(bp['fliers'][8], color='gold')
        setp(bp['fliers'][9], color='gold')
        setp(bp['medians'][4], color='gold')

        setp(bp['boxes'][5], color='darkgreen')
        setp(bp['caps'][10], color='darkgreen')
        setp(bp['caps'][11], color='darkgreen')
        setp(bp['whiskers'][10], color='darkgreen')
        setp(bp['whiskers'][11], color='darkgreen')
        setp(bp['fliers'][10], color='darkgreen')
        setp(bp['fliers'][11], color='darkgreen')
        setp(bp['medians'][5], color='darkgreen')

        setp(bp['boxes'][6], color='gold')
        setp(bp['caps'][12], color='gold')
        setp(bp['caps'][13], color='gold')
        setp(bp['whiskers'][12], color='gold')
        setp(bp['whiskers'][13], color='gold')
        setp(bp['fliers'][12], color='gold')
        setp(bp['fliers'][13], color='gold')
        setp(bp['medians'][6], color='gold')

        setp(bp['boxes'][7], color='darkgreen')
        setp(bp['caps'][14], color='darkgreen')
        setp(bp['caps'][15], color='darkgreen')
        setp(bp['whiskers'][14], color='darkgreen')
        setp(bp['whiskers'][15], color='darkgreen')
        setp(bp['fliers'][14], color='darkgreen')
        setp(bp['fliers'][15], color='darkgreen')
        setp(bp['medians'][7], color='darkgreen')
    """
#thaw_depth barplots
def TD_barplots(fname_seward,fname_barrow):
    td_seward = pd.read_csv(fname_seward,sep=',')
    td_barrow = pd.read_csv(fname_barrow,sep=',')
    data_arr = [td_seward,td_barrow]

    #sar_site_names
    seward = ['Teller','Kougarok','Council']
    date_crux = pd.Timestamp(2017,8,1)
    #sar plot names


    for i in range(0,2):
        #make datetime dataframes
        data_arr[i]['Date'] = pd.to_datetime(data_arr[i]['Date'],format='%Y-%m-%d',errors='ignore')

        #seward plotting
        if i ==0 :

            for k in range(0,len(seward)):
                site_data = data_arr[i][data_arr[i].Locale == seward[k]]
                categories = np.unique(site_data['SAR_Plot'])

                #separating by SAR Plot
                for j in range(0,len(categories)):
                    sar_plot_data = site_data[site_data['SAR_Plot'] == categories[j]]
                    #separating by early and late
                    for l in range(0,2):
                        if l ==0:
                            finalized_box = sar_plot_data[sar_plot_data['Date'] < date_crux]
                        else:
                            finalized_box =sar_plot_data[sar_plot_data['Date'] > date_crux]
            '''
            #sectioning the data combining transects
            TL_4_TA_may = (thaw_data.SAR_plot == 'TL_SAR_4')  & (thaw_data.Date == '5/29/2017')
            TL_4_TA_aug = (thaw_data.SAR_plot == 'TL_SAR_4')  & (thaw_data.Date == '8/16/2017')

            TL_8_TA_may = (thaw_data.SAR_plot == 'TL_SAR_8') & (thaw_data.Date == '5/29/2017')
            TL_8_TA_aug = (thaw_data.SAR_plot == 'TL_SAR_8') & (thaw_data.Date == '8/16/2017')

            TL_7_TA_may = (thaw_data.SAR_plot == 'TL_SAR_7') & (thaw_data.Date == '5/29/2017')
            TL_7_TA_aug = (thaw_data.SAR_plot == 'TL_SAR_7') & (thaw_data.Date == '8/16/2017')

            TL_41_TA_may = (thaw_data.SAR_plot == 'TL_SAR_41') & (thaw_data.Date == '5/29/2017')
            TL_41_TA_aug = (thaw_data.SAR_plot == 'TL_SAR_41') & (thaw_data.Date == '8/16/2017')

            KG_2_TA_may =(thaw_data.SAR_plot == 'KG_SAR_2')  & (thaw_data.Date == '5/29/2017')
            KG_2_TA_aug =(thaw_data.SAR_plot == 'KG_SAR_2')  & (thaw_data.Date == '8/17/2017')

            KG_5_TA_may =(thaw_data.SAR_plot == 'KG_SAR_5')  & (thaw_data.Date == '5/29/2017')
            KG_5_TA_aug =(thaw_data.SAR_plot == 'KG_SAR_5')  & (thaw_data.Date == '8/17/2017')

            KG_6_TA_may =(thaw_data.SAR_plot == 'KG_SAR_6')  & (thaw_data.Date == '5/29/2017')
            KG_6_TA_aug =(thaw_data.SAR_plot == 'KG_SAR_6')  & (thaw_data.Date == '8/17/2017')

            CN_1_TA_may =(thaw_data.SAR_plot == 'CN_SAR_1')  & (thaw_data.Date == '5/30/2017')
            CN_1_TA_aug =(thaw_data.SAR_plot == 'CN_SAR_1')  & (thaw_data.Date == '8/18/2017')

            CN_2_TA_may =(thaw_data.SAR_plot == 'CN_SAR_2')  & (thaw_data.Date == '5/30/2017')
            CN_2_TA_aug =(thaw_data.SAR_plot == 'CN_SAR_2')  & (thaw_data.Date == '8/18/2017')

            #print thaw_data.Depth_cm[TL_4_TA_may]
            #print thaw_data.Distance_m[TL_4_TA_may]

            #making array of boxes
            TL_sites =[thaw_data.Depth_cm[TL_4_TA_may],thaw_data.Depth_cm[TL_7_TA_may],thaw_data.Depth_cm[TL_8_TA_may],thaw_data.Depth_cm[TL_41_TA_may],thaw_data.Depth_cm[TL_4_TA_aug],thaw_data.Depth_cm[TL_7_TA_aug],thaw_data.Depth_cm[TL_8_TA_aug],thaw_data.Depth_cm[TL_41_TA_aug]]
            KG_sites = [thaw_data.Depth_cm[KG_2_TA_may],thaw_data.Depth_cm[KG_5_TA_may],thaw_data.Depth_cm[KG_6_TA_may],thaw_data.Depth_cm[KG_2_TA_aug],thaw_data.Depth_cm[KG_5_TA_aug],thaw_data.Depth_cm[KG_6_TA_aug]]
            CN_sites = [thaw_data.Depth_cm[CN_1_TA_may],thaw_data.Depth_cm[CN_2_TA_may],thaw_data.Depth_cm[CN_1_TA_aug],thaw_data.Depth_cm[CN_2_TA_aug]]

            #filtering TL
            filt_data_4 = TL_sites[0][~np.isnan(TL_sites[0])]
            filt_data_7 = TL_sites[1][~np.isnan(TL_sites[1])]
            filt_data_8 = TL_sites[2][~np.isnan(TL_sites[2])]
            filt_data_41 = TL_sites[3][~np.isnan(TL_sites[3])]

            filt_data_4a = TL_sites[4][~np.isnan(TL_sites[4])]
            filt_data_7a = TL_sites[5][~np.isnan(TL_sites[5])]
            filt_data_8a = TL_sites[6][~np.isnan(TL_sites[6])]
            filt_data_41a = TL_sites[7][~np.isnan(TL_sites[7])]

            #filtering KG
        #    filt_data_2 = KG_sites[0][~np.isnan(KG_sites[0])]
        #    filt_data_5 = KG_sites[1][~np.isnan(KG_sites[1])]
        #    filt_data_6 = KG_sites[2][~np.isnan(KG_sites[2])]

        #    filt_data_2a = KG_sites[3][~np.isnan(KG_sites[3])]
        #    filt_data_5a = KG_sites[4][~np.isnan(KG_sites[4])]
        #    filt_data_6a = KG_sites[5][~np.isnan(KG_sites[5])]

            #filtering CN
        #    filt_data_1 = CN_sites[0][~np.isnan(CN_sites[0])]
        #    filt_data_2 = CN_sites[1][~np.isnan(CN_sites[1])]

        #    filt_data_1a = CN_sites[2][~np.isnan(CN_sites[2])]
        #    filt_data_2a = CN_sites[3][~np.isnan(CN_sites[3])]
            #print filt_data_8
            #print thaw_data.Depth_cm[TL_8_TA_aug]
            filt_data = [filt_data_4,filt_data_4a, filt_data_7,filt_data_7a,filt_data_8, filt_data_8a,filt_data_41,filt_data_41a]
            ig = plt.figure(1, figsize=(9, 6))

            # Create an axes instance
            ax = fig.add_subplot(111)

            hold(True)
            # Create the boxplot
            bp = ax.boxplot(filt_data[0:2], patch_artist=True,positions=[1,2],widths=0.6)
            setBoxColors(bp)

            bp = ax.boxplot(filt_data[2:4], patch_artist=True,positions=[3,4],widths=0.6)
            setBoxColors(bp)

            bp = ax.boxplot(filt_data[4:6], patch_artist=True,positions=[5,6],widths=0.6)
            setBoxColors(bp)

            bp = ax.boxplot(filt_data[6:8], patch_artist=True,positions=[7,8],widths=0.6)
            setBoxColors(bp)
            ## change outline color, fill color and linewidth of the boxes
        #    for box in bp['boxes']:
                # change outline color
        #        box.set( color='black', linewidth=1)
                # change fill color
        #        box.set( facecolor = 'darkgreen' )

            ## change color and linewidth of the whiskers
        #    for whisker in bp['whiskers']:
        #        whisker.set(color='black', linewidth=1)

            ## change color and linewidth of the caps
        #    for cap in bp['caps']:
        #        cap.set(color='black', linewidth=1)

            ## change color and linewidth of the medians
        #    for median in bp['medians']:
        #        median.set(color='black', linewidth=2)

            ## change the style of fliers and their fill
        #    for flier in bp['fliers']:
        #        flier.set(marker='o', color='#e7298a', alpha=0.5)

            plt.xlabel('Sites')
            plt.ylabel('Depth (cm)')
            plt.title('Teller Thaw Depths in 2017')
            plt.xticks([1,2,3,4,5,6,7,8],['TL-4','TL-4','TL-7','TL-7','TL-8','TL-8','TL-41','TL-41'])
            plt.ylim(0,120)
            plt.xlim(0,10)
            hB, = plot([1,1],'gold')
            hR, = plot([1,1],'darkgreen')
            legend((hB, hR),('May/June', 'August'),loc=4)
            hB.set_visible(False)
            hR.set_visible(False)
            #plt.legend(('May','August'),loc=2)
            plt.savefig('C:/Users/323053/Documents/Julian_Python/figures/thaw_depth_bp/bp_TL_v2.png')
            #fig.savefig('C:/Users/323053/Documents/Julian_Python/fig1.png', bbox_inches='tight')
        #    plt.show()
        #plotting barrow
        else:

        '''

def weather_comparison(weather_csv,sar_csv):
    ''' program intended to compare sar data with weather'''

    #read file skipping rows and renaming columns etc
    weather=pd.read_csv(weather_csv,skiprows=[2,3],header=1,names=['date','precip'],na_values=6999)
    sar_df=pd.read_csv(sar_csv)

    #convert both to datetime objects
    weather['date'] = pd.to_datetime(weather['date'])
    sar_df['Date'] = pd.to_datetime(sar_df['Date'],errors='coerce')

    weather.set_index('date')

    weather = weather.groupby(weather['date'].dt.date).sum()

    #take only teller SAR data
    tl_sar_df= sar_df[sar_df['site'] == 'Teller']
    group = tl_sar_df.groupby('Date').sum()


    #plotprecipitation data
    weather.plot(y='precip')

    #plot vertical lines on dates
    for f in group.index.values:
        plt.axvline(x=f,color='r',linestyle='--')
    plt.ylabel('Precipitation (cm)?')
    plt.show()

def assign_coor(data_filename,coor_file):
    coor = pd.read_csv(coor_file,sep=',')
    data = pd.read_csv(data_filename,sep=',')
    sar_sites = ['TL_SAR_4','TL_SAR_7','TL_SAR_8','TL_SAR_41','KG_SAR_2','KG_SAR_5','KG_SAR_6','CN_SAR_1','CN_SAR_2','BW_SAR_LC','BW_SAR_FC','BW_SAR_HC']
    flags = ['Flag_1','Flag_2','Flag_3','Flag_4','Flag_5','Flag_6']
    #for i in range(0,len(sar_sites)):
    #    for j in range(0,len(flags)):
    #        flag_data = data[(data.sar_plot == sar_sites[i]) & (data.flag ==flags[j])]
    #        print flag_data
    print coor
    data2 = pd.merge(data,coor,how = 'left',left_on =['flag','sar_plot'],right_on = ['flag','sar_plot'] )

    data2.to_csv('Z:/AKSeward/Data/Excel/SAR/discrete_data_coor.csv')

def SAR_comparison(data_filename,depth='20',combine=False):
    '''Function meant to read in file containing avg in situ hydrosense values and SAR raster statistics for comparison in plots'''
    #use ggplot
    matplotlib.style.use('ggplot')
    #read in data
    data = pd.read_csv(data_filename,sep=',')

    #make date into datetime object
    data['Date'] = pd.to_datetime(data['Date'])

    #take only late dates
    data_late = data[(data['Date']>pd.Timestamp(2017,7,1)) & (data['site'] != 'Barrow')]

    #make single plot or all three depths in one
    if len(depth) == 1:
        #use column name for hydrosense specification
        col_names = 'avg_'+depth
        col_err = 'std_'+depth
        symbol='o'
        title='Comparing Hydrosense II Soil Moisture with derived ABoVE SAR Soil Moisture ('+depth+'cm)'
        #divide each column by 100.0
        data_late[col_name[0]] = data_late[col_name[0]]/100.0
        data_late[col_err[0]] = data_late[col_err[0]]/100.0
        save_name = 'Z:/AKSeward/2017_SAR/CJW_Phase_3_Poster/figures/2017_aug_mv1_'+depth+'cm.png'
    if len(depth)>1:
        #defining columns
        col_name = ['avg_'+depth[2],'avg_'+depth[1],'avg_'+depth[0]]
        col_err = ['std_'+depth[2],'std_'+depth[1],'std_'+depth[0]]
        symbol =['o','s','^']
        title = ' In-Situ Soil Moisture vs. derived ABoVE SAR Soil Moisture'
        alpha = [1,0.6,0.2]
        #divide each column by 100.0
        data_late[col_name] = data_late[col_name]/100.0
        data_late[col_err] = data_late[col_err]/100.0
        fill_style = ['top','full','bottom',]
        edge_colors =['yellow','purple','orange']
        size = [650,250,100]
        save_name = 'Z:/AKSeward/2017_SAR/CJW_Phase_3_Poster/figures/2017_aug_mv1_all_depths.png'
    fig,ax=plt.subplots(figsize=(15,10))
    #colors = {'Teller':'royalblue','Kougarok':'salmon','Council':'seagreen'}
    set = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628" ,"#F781BF", "#999999"]

    colors = {'Teller':set[2],'Kougarok':set[1],'Council':set[0]}

    grouped =  data_late.groupby('site')

    for i in range(0,len(depth)):
        for key,group in grouped:
            group.plot(ax=ax,x=col_name[i],y='MEAN',xerr=col_err[i],yerr='STD',kind='scatter',s=size[i],color=colors[key],edgecolors='k',linewidths=1.5 ,marker=MarkerStyle(symbol[i]))#,fillstyle=fill_style[i]),edgecolors=edge_colors[i])#,edgecolors=edge_colors[i],linewidths=2)
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
    plt.savefig(save_name,dpi=500)
    #plt.show()


def LIDAR_vs_Hydrosense(lidar_filename,perflag_filename,depth='20'):
    '''function meant to grab the stats from 3m buffer around august hydrosense measurements for lidar intensity mosaic at teller in 2017'''
    lidar = pd.read_csv(lidar_filename,sep=',')
    hydro = pd.read_csv(perflag_filename,sep=',')

    #make date into datetime object
    hydro['Date'] = pd.to_datetime(hydro['Date'],errors='coerce')

    #take only late dates
    hydro = hydro[(hydro['Date']>pd.Timestamp(2017,7,1))]

    #use ggplot
    matplotlib.style.use('ggplot')

    #use column name for hydrosense specification
    col_name = ['avg_'+depth,'std_'+depth]
    print col_name

    #replace -9999 with nan
    hydro[col_name[0]] = hydro[col_name[0]].replace(-9999.0,np.NaN)

    #dividing by 100
    hydro[col_name[0]] = hydro[col_name[0]]/100.0
    hydro[col_name[1]] = hydro[col_name[1]]/100.0

    #merge on the lidar flag, sar plot site
    merged = pd.merge(lidar,hydro,how='left',on=['sar_plot','flag'])

    #create figure
    fig,ax=plt.subplots(figsize=(15,10))
    #colors = {'Teller':'royalblue','Kougarok':'salmon','Council':'seagreen'}
    set = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628" ,"#F781BF", "#999999"]

    #assigning colors to features
    colors = {'TL_SAR_4':set[3],'TL_SAR_7':set[4],'TL_SAR_8':set[6]}
    grouped =  merged.groupby('sar_plot')

    for key,group in grouped:
        group.plot(ax=ax,x=col_name[0],y='MEAN',xerr=col_name[1],yerr='STD',label=key,s=100,color=colors[key],kind='scatter')

    #plt.xlim(0,1)
    #plt.ylim(0,1)
    #create 1:1 line

    plt.xlabel('Hydrosense Volumetric Water Content (%/100)',fontsize=18)
    plt.ylabel('2017 Lidar Intensity Return (W/m$^2$)',fontsize=18)
    plt.title('Comparing Hydrosense II Soil Moisture with 2017 LIDAR Intensity ('+depth+'cm)',fontsize=22)
    plt.ylim(10,45)
    plt.xlim(0,1)
    plt.savefig('Z:/AKSeward/2017_SAR/CJW_Phase_3_Poster/figures/2017Lidar_aug_'+depth+'cm.png',dpi=500)
    plt.close()

def indi_boxplots(filename,sar_plot='Teller'):
    '''produce individual boxplots for each flag at a specific site'''

    #use ggplot
    matplotlib.style.use('ggplot')

    df = pd.read_csv(filename,sep=',')
    #datetime
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'],errors='coerce')

    #narrow selection to Teller sites in august
    teller = df[df['Locale']=='Teller']
    SAR_plots = ['TL_SAR_4','TL_SAR_7','TL_SAR_8','TL_SAR_41']
    flag_num = ['Flag_1','Flag_2','Flag_3','Flag_4','Flag_5','Flag_6']

    for plot in SAR_plots:
        for flag in flag_num:
            sub_data = teller[(teller['SAR_Plot'] == plot) & (teller['SAR_Plot_Flag'] == flag) & (teller['VWC_Measurement_Depth'] == 6)]
            print sub_data['Measurement_ID']
            print sub_data['VWC'].mean(),sub_data['VWC'].std()
            break
        break

def main():
    #filenames for entry in programs
    td_seward = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/thaw_depth_seward_2017.csv'
    td_barrow = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/thaw_depth_barrow_2017.csv'
    vwc_seward_may = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/vwc_seward_may_2017.csv'
    vwc_seward_aug = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/vwc_seward_aug_2017.csv'
    vwc_barrow_sept = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/vwc_barrow_sept_2017.csv'
    vwc_barrow_jun = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/vwc_barrow_june_2017.csv'
    slope_file = 'Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/SAR_plot_parameters_editted.csv'
    concavity_file_1m = 'Z:/AKSeward/Data/Excel/SAR/1m_buffer_info.xls'
    concavity_file_2m = 'Z:/AKSeward/Data/Excel/SAR/2m_buffer_info.xls'
    TD_seward = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/thaw_depth_seward_2017.csv'
    TD_barrow = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/thaw_depth_barrow_2017.csv'

    discrete_data = r'Z:/AKSeward/Data/Excel/SAR/discrete_data2.csv'
    coor_file = 'Z:/AKSeward/Data/Excel/SAR/Coord_Files/all_sar.csv'
    SAR_comparison_file ='Z:/AKSeward/Data/Excel/SAR/plot_avg_w_SARmv1_aug.csv'
    #SAR_comparison(SAR_comparison_file,depth=['6','12','20'])

    lidar_3m_stats = 'Z:/AKSeward/Data/GIS/Teller/SAR/2017_aug_3mbuffer_lidar_intensity_stats.csv'
    avg_per_flag_file = 'Z:/AKSeward/Data/Excel/SAR/discrete_data_final_20190306.csv'
    #LIDAR_vs_Hydrosense(lidar_3m_stats,avg_per_flag_file,depth='20')
    #assign_coor(discrete_data,coor_file)
    #td_vwc_plotter(td_barrow,vwc_barrow_jun,vwc_barrow_sept)
    #td_vwc_plotter(td_seward,vwc_seward_may,vwc_seward_aug)
    #data = pd.DataFrame(avg_by_plot(td_seward,vwc_seward_may,vwc_seward_aug,td_barrow,vwc_barrow_jun,vwc_barrow_sept))
    #data.to_csv('Z:/AKSeward/Data/Excel/SAR/plot_avg.csv')
    #avg_plots(data,slope_file)
    #data = pd.DataFrame(avg_per_flag(vwc_seward_may,vwc_seward_aug,vwc_barrow_jun,vwc_barrow_sept))
    discrete_data = 'Z:/AKSeward/Data/Excel/SAR/avg_flag_real_coor_20190305.csv'
    #data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.iteritems() ]))
    #data.to_csv(discrete_data)
    #combine_data_with_concavity_info(data,concavity_file_1m,concavity_file_2m)
    #dis_data = discrete_calculation(discrete_data)
    indi_boxplots(vwc_seward_aug)
    discrete_data_final= 'Z:/AKSeward/Data/Excel/SAR/discrete_data_final_20190306.csv'
    weather_csv = 'Z:/JDann/Documents/Documents/Julian_Python/Surface_meteorology_Teller_download_20190304/NGA079.summer_precip_Tot.csv'
    #weather_comparison(weather_csv,discrete_data_final)
    #dis_data.to_csv('Z:/AKSeward/Data/Excel/SAR/discrete_data_final_20190306.csv')
    #commands for plotting discrete data
    #discrete_data = 'Z:/AKSeward/Data/Excel/SAR/discrete_data2.csv'
    #discrete_plotting(discrete_data, slope_file)

    #commands for plotting thaw depth TD_barplots
    #TD_barplots(TD_seward,TD_barrow)

if __name__ == "__main__":
    main()
