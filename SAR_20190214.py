import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
import datetime as dt
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns

def rename_SAR_plots(df):
    '''function that takes all SAR_Plot rows from barrow and makes them the same so that I can group by SAR Plot'''
        #rename columns with iterations BW_SAR_HC_1,2,3,4
    HC = df['SAR_Plot'].str.startswith('Barrow_SAR_HC')
    LC =df['SAR_Plot'].str.startswith('Barrow_SAR_LC')
    FC =df['SAR_Plot'].str.startswith('Barrow_SAR_FC')

    df['SAR_Plot'][HC] = 'Barrow_SAR_HC'
    df['SAR_Plot'][LC] = 'Barrow_SAR_LC'
    df['SAR_Plot'][FC] = 'Barrow_SAR_FC'
    return df

def main():
    '''python script meant to better understand the SAR data'''

    #read_files into merged pandas dataframe for VWC and TD
    df_vwc = pd.concat([pd.read_csv(f) for f in glob.glob('Z:/AKSeward/2017_SAR/SAR_download_20181003/data/vwc*.csv')], ignore_index = True,sort=True)
    df_TD = pd.concat([pd.read_csv(f) for f in glob.glob('Z:/AKSeward/2017_SAR/SAR_download_20181003/data/thaw*.csv')], ignore_index = True,sort=True)

    #rename SAR plots
    df_vwc = rename_SAR_plots(df_vwc)

    #pandas datetime the object
    df_vwc['TimeStamp'] = pd.to_datetime(df_vwc['TimeStamp'])
    df_TD['Date'] = pd.to_datetime(df_TD['Date'])

    #rename column to match
    df_TD.rename(columns={'Date':'TimeStamp'},inplace=True)

    #date to split on
    split_data = pd.to_datetime('20170630', format='%Y%m%d', errors='ignore')

    #grab early and late data
    early_vwc = df_vwc[df_vwc['TimeStamp'] < split_data]
    late_vwc = df_vwc[df_vwc['TimeStamp'] > split_data]
    early_td = df_TD[df_TD['TimeStamp'] < split_data]
    late_td = df_TD[df_TD['TimeStamp'] > split_data]
    arr = [early_vwc,late_vwc,early_td,late_td]

    #groupby site
    fig, axs = plt.subplots(2,1)

    #strip leading and trailing spaces
    for i in arr:
        i['Locale'] = i['Locale'].str.strip()

    group = early_vwc.groupby('Locale')
    print(group.groups.keys())

    sns.boxplot(x='Locale',y='VWC',hue='VWC_Measurement_Depth',ax=axs[0],data=early_vwc)
    sns.boxplot(x='Locale',y='VWC',hue='VWC_Measurement_Depth',ax=axs[1],data=late_vwc)
    plt.show()

if __name__ == '__main__':
    main()
