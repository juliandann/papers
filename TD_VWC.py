import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plotting_mechanism(td,vwc,sar_plot,date,transect,title,catcol='Thaw_Depth_Comments'):
    #print 'td',td[['SAR_Plot','SAR_Plot_Transect','Date']]
    #print 'vwc',vwc[['SAR_Plot','SAR_Plot_Transect','TimeStamp']]
    #plotting thaw depth first
    td.loc[td[catcol].str.contains('FT'),catcol] = 'FT'
    td.loc[td[catcol].str.contains('PR'),catcol] = 'PR'
    td.loc[(td[catcol].str.contains('R') & ~td[catcol].str.contains('PR')),catcol] = 'R'

    categories = np.unique(td[catcol])
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
        if ('SN' in categories[i]) and ('PR' not in categories[i]):
            colors1.append('k')
        if ('SH' in categories[i]) and ('PR' not in categories[i]):
            colors1.append('k')
    colordict = dict(zip(categories, colors1))
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


    for i in range(0,len(colors)):
        colors[i] = colors[i][colors[i].Thaw_Depth != -9999]
        markerline, stemlines, baseline = plt.stem(colors[i]['Distance'],-1.0*colors[i]['Thaw_Depth'])
        #extend to end of plot
        baseline.set_xdata([0,60])


        plt.setp(baseline, color='saddlebrown',linewidth=7)
        if colors[i].Color.iloc[0] == 'r' :
            plt.setp(stemlines,linestyle='--', color='gray')
            plt.setp(markerline,color='',marker='')
        if colors[i].Color.iloc[0] == 'b' :
            plt.setp(stemlines,linestyle='-', color='k')
            plt.setp(markerline,color='k',marker='_')
        if colors[i].Color.iloc[0] == 'purple' :
            plt.setp(stemlines,linestyle='-', color='k')
            plt.setp(markerline,color='k',marker=7)
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
        cp = plt.scatter(dist_flag,depth_arr, c = vwc_arr[m],s=100,marker='p',vmin=0,vmax=100, cmap = cm,zorder=10,alpha=0.9)
    plt.colorbar(cp)

    plt.ylim(-120,20)
    plt.xlim(-5,65)
    plt.xlabel('Distance (m)')
    plt.ylabel('Depth (cm)')
    plt.title(title)
    plt.savefig('Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/'+str(title.strip())+'.png')
    plt.close('all')
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
        sar_plotvwc_names = ['BW_SAR_HC_','BW_SAR_LC_','BW_SAR_FC_']

    #transect names for thaw depth measurements
    tran_names = ['TL1','TL2']
    tran_names_vwc = ['T1_','T2_']

    #setting up for a for loop to go through each site and transect
    sar_plot = []
    transect = []
    date = []
    title = []

    for i in range(0,len(sar_plot_names)):
        for h in range(0,2):
            for j in range(0,len(tran_names)):
                td_data = td_com[h]
                vwc_data = vwc_com[h]


                sub_vwc_data = vwc_data[(vwc_data['Measurement_ID'].str.contains(sar_plotvwc_names[i])) & (vwc_data['Measurement_ID'].str.contains(tran_names_vwc[j]))]
                sub_td_data = td_data[(td_data['Measurement_ID'].str.contains(sar_plot_names[i])) & (td_data['Measurement_ID'].str.contains(tran_names[j]))]

                sar_plot.append(sub_td_data['SAR_Plot'].iloc[0])
                transect.append(sub_td_data['SAR_Plot_Transect'].iloc[0])
                date.append((sub_td_data['Date'].iloc[0]).strftime("%Y-%m-%d"))
                #print date[-1],sar_plot[-1],transect[-1]
                title=str(sar_plot[-1])+str(transect[-1])+' on '+str(date[-1])

                plotting_mechanism(sub_td_data,sub_vwc_data,sar_plot[-1],date[-1],transect[-1],title)


def main():
    #filenames for entry in programs
    td_seward = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/thaw_depth_seward_2017.csv'
    td_barrow = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/thaw_depth_barrow_2017.csv'
    vwc_seward_may = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/vwc_seward_may_2017.csv'
    vwc_seward_aug = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/vwc_seward_aug_2017.csv'
    vwc_barrow_sept = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/vwc_barrow_sept_2017.csv'
    vwc_barrow_jun = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/vwc_barrow_june_2017.csv'

    td_vwc_plotter(td_seward,vwc_seward_may,vwc_seward_aug)
if __name__ == "__main__":
    main()
