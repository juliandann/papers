import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

def dfScatter(df,df2, title, xcol='Distance', ycol='Thaw_Depth', catcol='Thaw_Depth_Comments'):
    fig, ax = plt.subplots(1,2,figsize=(10,10))
    categories = np.unique(df[catcol])
    colors1 = []
    for i in range(0,len(categories)):
        if 'R' in categories[i]:
            colors1.append('r')
        if 'FT' in categories[i]:
            colors1.append('b')
        if 'PR' in categories[i]:
            colors1.append('purple')
    #colors = ['b','r']
    colordict = dict(zip(categories, colors1))
    df["Color"] = df[catcol].apply(lambda x: colordict[x])

    categories2 = np.unique(df2[catcol])
    colors2 = []
    for i in range(0,len(categories2)):
        if 'R' in categories2[i] and 'PR' not in categories2[i] :
            colors2.append('r')
        if 'FT' in categories2[i]:
            colors2.append('b')
        if 'PR' in categories2[i]:
            colors2.append('purple')
    #colors = ['b','r']
    colordict2 = dict(zip(categories2, colors2))
    df2["Color"] = df2[catcol].apply(lambda x: colordict2[x])

    custom_lines = [Line2D([0], [0], marker='o', color='b', label='Frost Table', markerfacecolor='b', markersize=10),
    Line2D([0], [0], marker='o', color='r', label='Rock', markerfacecolor='r',markersize=10),
    Line2D([0], [0], marker='o', color='purple', label='Probe Length', markerfacecolor='purple', markersize=10)]


    ax[0].scatter(df[xcol], df[ycol], c=df.Color,s=20.0,edgecolors='face')
    ax[0].plot(df[xcol], df[ycol],'--k')
    ax[0].set_title(title[0])

    max1 = max(df[ycol])
    max2 = max(df2[ycol])
    ax[0].set_ylim(0,max(max1,max2)+10)
    ax[0].set_xlim(-5,65)
    ax[0].set_ylabel('Thaw Depth (cm)')
    ax[0].set_xlabel('Distance (m)')

    ax[1].scatter(df2[xcol], df2[ycol], c=df2.Color,s=20.0,edgecolors='face')
    ax[1].plot(df2[xcol], df2[ycol],'--k')
    ax[1].set_ylim(0,max(max1,max2)+10)
    ax[1].set_xlim(-5,65)
    ax[1].set_xlabel('Distance (m)')
    ax[1].set_title(title[1])
    ax[0].legend(handles=custom_lines,loc='upper right')

    return fig


def difference_of_thawdepth(fname):
    data = pd.read_csv(fname,sep=',')

    #converting dates to date_time
    data['Date'] = pd.to_datetime(data['Date'],format='%m/%d/%Y')

    #importing transects
    sar_plot_names = ['TL_SAR_4-','TL_SAR_8-','TL_SAR_7-','TL_SAR_41-','KG_SAR_2-','KG_SAR_5-','KG_SAR_6-','CN_SAR_1-','CN_SAR_2-']
    tran_names = ['TL1','TL2']


    sar_plot = []
    transect = []
    date_st = []
    date_end = []
    total_len = len(sar_plot_names)+len(tran_names)
    title = []
    fig, ax = plt.subplots(6,3,figsize=(20,15))
    fig.tight_layout()
    index = 0
    ax = ax.flatten()
    avg = []
    error = []
    counts = []
    for i in range(0,len(sar_plot_names)):
        for j in range(0,len(tran_names)):
            sub_data = data[(data['Measurement_ID'].str.contains(sar_plot_names[i])) & (data['Measurement_ID'].str.contains(tran_names[j])) & (data['Thaw_Depth'] != -9999)]

            #setting up time mask for separating beginning of summer from end of summer data
            date_masks = [((sub_data['Date'] >  pd.Timestamp(2017,5,1)) & (sub_data['Date'] <  pd.Timestamp(2017,6,30))),
            ((sub_data['Date'] >  pd.Timestamp(2017,8,1)) & (sub_data['Date'] <  pd.Timestamp(2017,9,30))) ]

            #further segregate sub_data to have both available for stacked plotting
            sub_sub_data_st = sub_data[date_masks[0]]
            sub_sub_data_end = sub_data[date_masks[1]]
            sar_plot.append(sub_data['SAR_Plot'].iloc[0])
            transect.append(sub_data['SAR_Plot_Transect'].iloc[0])
            date_st.append((sub_sub_data_st['Date'].iloc[0]).strftime("%Y-%m-%d"))
            date_end.append((sub_sub_data_end['Date'].iloc[0]).strftime("%Y-%m-%d"))

            dist = []
            dif = []

            for k in range(0,len(sub_sub_data_st)):
                for m in range(0,len(sub_sub_data_end)):
                    if sub_sub_data_st['Distance'].iloc[k] == sub_sub_data_end['Distance'].iloc[m] and ('NR' not in str(sub_sub_data_st['Thaw_Depth_Flag'].iloc[k])) and ('NR' not in str(sub_sub_data_end['Thaw_Depth_Flag'].iloc[m])):
                        dif.append(sub_sub_data_end['Thaw_Depth'].iloc[m] - sub_sub_data_st['Thaw_Depth'].iloc[k])
                        dist.append(sub_sub_data_st['Distance'].iloc[k])

            #averaging for each transects
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avg.append(np.mean(dif))
                error.append(np.std(dif))
                counts.append(len(dif))

            if 'TL' in sar_plot[-1]:
                color = 'b'
            if 'CN' in sar_plot[-1]:
                color = 'red'
            if 'KG' in sar_plot[-1]:
                color = 'green'
            ax[index].plot(dist,dif,marker = 'o',linestyle='None',color=color,markeredgewidth=0.0)
            ax[index].set_xlim(-5,65)
            ax[index].set_ylim(0,120)
            ax[index].set_title(str(sar_plot[-1])+str(transect[-1]))
            ax[index].axhline(y=avg[-1],c='k',linestyle = '--')
            ax[index].text(0.01,0.9,'Avg: '+str(round(avg[-1],2)),transform=ax[index].transAxes)
            ttl = ax[index].title
            ttl.set_position([.5, 1.02])
            fig.text(0.5,0.005,'Distance (m)',ha='center')
            fig.text(0.005,0.5,'Thaw Depth (cm)',va = 'center',rotation='vertical')
            title.append(str(sar_plot[-1])+str(transect[-1]))

            index=index+1
    plt.savefig('Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/difference_td.png')
    #plt.close('all')
    plt.close('all')

    #barplot of averages
    arr_ind = np.arange(0,len(avg),1.0)
    fig = plt.figure(figsize=(35,5))

    #setting color of bars:
    colors = []
    for b in range(0,len(title)):

        if 'TL_' in title[b]:
            colors.append('b')
        if 'KG_' in title[b]:
            colors.append('g')
        if 'CN_' in title[b]:
            colors.append('r')
    bars = plt.bar(arr_ind,avg,tick_label=title,yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10,color = colors)

    #adding count on top of bars
    ind = 0
    for rect in bars:

        height = rect.get_height()
        if np.isnan(height):
            height = 0
        else:
            height = height + error[ind]
        plt.text(rect.get_x() + rect.get_width()/2.0, height, str(counts[ind]), ha='center', va='bottom')
        ind = ind+1
    plt.xlim(-1,len(avg))
    plt.ylim(0,120)
    plt.tight_layout()
    plt.savefig('Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/difference_barplot.png')
    plt.close('all')
def thaw_depth_plotter(fname):
    data = pd.read_csv(fname,sep=',')

    #converting dates to date_time
    data['Date'] = pd.to_datetime(data['Date'],format='%m/%d/%Y')

    #importing transects
    sar_plot_names = ['TL_SAR_4-','TL_SAR_8-','TL_SAR_7-','TL_SAR_41-','KG_SAR_2-','KG_SAR_5-','KG_SAR_6-','CN_SAR_1-','CN_SAR_2-']
    tran_names = ['TL1','TL2']


    sar_plot = []
    transect = []
    date_st = []
    date_end = []
    for i in range(0,len(sar_plot_names)):
        for j in range(0,len(tran_names)):
            sub_data = data[(data['Measurement_ID'].str.contains(sar_plot_names[i])) & (data['Measurement_ID'].str.contains(tran_names[j])) & (data['Thaw_Depth'] != -9999)]

            #setting up time mask for separating beginning of summer from end of summer data
            date_masks = [((sub_data['Date'] >  pd.Timestamp(2017,5,1)) & (sub_data['Date'] <  pd.Timestamp(2017,6,30))),
            ((sub_data['Date'] >  pd.Timestamp(2017,8,1)) & (sub_data['Date'] <  pd.Timestamp(2017,9,30))) ]

            #further segregate sub_data to have both available for stacked plotting
            sub_sub_data_st = sub_data[date_masks[0]]
            sub_sub_data_end = sub_data[date_masks[1]]
            sar_plot.append(sub_data['SAR_Plot'].iloc[0])
            transect.append(sub_data['SAR_Plot_Transect'].iloc[0])
            date_st.append((sub_sub_data_st['Date'].iloc[0]).strftime("%Y-%m-%d"))
            date_end.append((sub_sub_data_end['Date'].iloc[0]).strftime("%Y-%m-%d"))

            title = [str(sar_plot[-1])+'-'+str(transect[-1])+' on '+str(date_st[-1]), str(date_end[-1])]
            fig = dfScatter(sub_sub_data_st,sub_sub_data_end,title)
            fig.savefig('Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/'+str(sar_plot[-1])+str(transect[-1])+'.png')

def main():
    fname = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/thaw_depth_seward_2017.csv'
    #thaw_depth_plotter(fname)
    difference_of_thawdepth(fname)
if __name__ == "__main__":
    main()
