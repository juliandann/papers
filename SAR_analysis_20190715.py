import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import xarray as xr
import time
from matplotlib.markers import MarkerStyle
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from SAR_gen_functions import *
import glob
import geopy.distance

class Paths:
    def __init__(self, figures, data_save,data_load):
    self.figures = figures
    self.data_save = data_save
    self.data_load = data_load

start_time = time.time()


def main():

    #work comp class
    work_paths = Paths('Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/','Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/Figures/','Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/Figures/')

    #personal comp paths
    personal_paths = Paths('/Users/juliandann/Documents/LANL/qgis/Figures/','/Users/juliandann/Documents/LANL/qgis/CSV/','/Users/juliandann/Documents/LANL/qgis/CSV/')

    above_topo = 'all_coor_slope_aspect_curvature.csv'



    #uncomment for filepaths on work computer
    TL = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_6_12_20_TL.csv'
    KG ='Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_6_12_20_KG.csv'
    CN = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_6_12_20_CN.csv'
    BW ='Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_6_12_20_BW.csv'
    ABoVE_data = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/discrete_all_sites.csv'
    VWC_data= 'Z:/AKSeward/Data/Excel/SAR/discrete_data_final_20190306.csv'
    all_vwc_data = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/vwc_all.csv'
    closest_dist = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/closest_dist_w_lbc.csv'
    all_td_SP = 'Z:/AKSeward/2017_SAR/SAR_download_20181003/data/thaw_depth_seward_2017.csv'
    closest_dist_td = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/closest_dist_td.csv'
    td_per_pixel = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/avg_td_per_pixel.csv'
    landcover = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/gaplandfire_pixels.csv'
    all_ABoVE = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/P_PolSAR_ALP_seward_170817_171010_01.nc4'
    landcover_stats = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/GAPLANDFIRE_60m_statistics.csv'
    all_above_vwc = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/all_coor_slope_aspect_curvature.csv'
    closest_above_vwc = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/all_above_closest_gaplandfire.csv'
    barrow = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/P_PolSAR_ALP_barrow_170813_171009_01.nc4'
    tl_bot_rad_file = 'Z:/AKSeward/Data/GIS/Seward_Peninsula/ABoVE_Seward/Weather/NGA079.teller_bottom_rad_2017.csv'
    tl_bot_met_file = 'Z:/AKSeward/Data/GIS/Seward_Peninsula/ABoVE_Seward/Weather/NGA079.teller_bottom_met_2017.csv'
    rad_file_tl_top = 'Z:/AKSeward/Data/GIS/Seward_Peninsula/ABoVE_Seward/Weather/Teller/NGA079.teller_top_rad_2017.csv'
    met_file_tl_top = 'Z:/AKSeward/Data/GIS/Seward_Peninsula/ABoVE_Seward/Weather/Teller/NGA079.teller_top_met_2017.csv'
    kg_met_file = 'Z:/AKSeward/Data/GIS/Seward_Peninsula/ABoVE_Seward/Weather/Kougarok/ngee_kougarok_met_2017.csv'
    above_slope = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/all_coor_slope.csv'

    '''
    # uncomment for filepaths on personal computer
    TL = '/Users/juliandann/Documents/LANL/SAR_DATA_AND_Programs/20190719/discrete_6_12_20_TL.csv'
    KG ='/Users/juliandann/Documents/LANL/SAR_DATA_AND_Programs/20190719/discrete_6_12_20_KG.csv'
    CN = '/Users/juliandann/Documents/LANL/SAR_DATA_AND_Programs/20190719/discrete_6_12_20_CN.csv'
    BW ='/Users/juliandann/Documents/LANL/SAR_DATA_AND_Programs/20190719/discrete_6_12_20_BW.csv'
    ABoVE_data = '/Users/juliandann/Documents/LANL/SAR_DATA_AND_Programs/20190719/discrete_all_sites.csv'
    VWC_data = '/Users/juliandann/Documents/LANL/SAR_DATA_AND_Programs/20190719/all_vwc_combined.csv'
    closest_pts = '/Users/juliandann/Documents/LANL/SAR_DATA_AND_Programs/20190719/closest_points.csv'
    closest_dist = '/Users/juliandann/Documents/LANL/SAR_DATA_AND_Programs/20190719/closest_distance.csv'
    all_vwc_data = '/Users/juliandann/Documents/LANL/SAR_DATA_AND_Programs/20190719/all_vwc_combined.csv'
    '''
    '''
    #uncomment for making combined files
    files = [TL,KG,CN,BW]
    csv_combined_file_maker(files,combined_filepath='/Users/juliandann/Documents/LANL/SAR_DATA_AND_Programs/20190719/discrete_all_sites.csv')


    vwc_files = glob.glob('/Users/juliandann/Documents/LANL/SAR_DATA_AND_Programs/SAR_DATA_AND_Programs/data/vwc*.csv')
    csv_combined_file_maker(vwc_files,combined_filepath=all_vwc_data)
    '''


    #read in the data
    #rad = pd.read_csv(rad_file_top)
    #met = pd.read_csv(kg_met_file)

    #weather(met,title='Temperature and Precipitation at Kougarok (mile 64)')
    '''
    ds = xr.open_dataset(barrow)
    df1 = ds.to_dataframe()
    df1 = engstrom_SM_mv(df1,'mv1_aug','mv2','z1_aug',0.06)
    df1 = engstrom_SM_mv(df1,'mv1_aug','mv2','z1_aug',0.12)
    df1 = engstrom_SM_mv(df1,'mv1_aug','mv2','z1_aug',0.2)
    df1 = laura_bc_SM(df1,'mv1_aug','mv2','z1_aug',0.06)
    df1 = laura_bc_SM(df1,'mv1_aug','mv2','z1_aug',0.12)
    df1 = laura_bc_SM(df1,'mv1_aug','mv2','z1_aug',0.2)
    df1.to_csv('Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/Barrow/2017_aug_barrow.csv',sep=',')
    '''
    #df1 = pd.read_csv(ABoVE_data,sep=',')
    #df2 = pd.read_csv(all_above_vwc,sep=',')
    #df1 = pd.read_csv(all_td_SP,sep=',')
    #engstrom_SM(df2,'epsilon1_aug')
    #df1 = pd.read_csv(td_per_pixel,sep=',')
    #td_comparison(df1,'above_Index','Thaw_Depth_mean','above_alt','above_alt_uncertainty','Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/Figures/ALT/')
    #df1 = pd.read_csv(VWC_data,sep=',')
    df2 = pd.read_csv(all_above_vwc)
    #df = pd.read_csv(closest_above_vwc,sep=',')

    #compare_above_vs_hydrosense(df,['avg_6','avg_12','avg_20'],['above_engstrom_0.06','above_engstrom_0.12','above_engstrom_0.2'],xerr=['std_6','std_12','std_20'],xlabel='In-Situ Volumetric Water Content (%)',ylabel='ABoVE VWC (Engstrom Calibration) (%)',title=' In-Situ Soil Moisture vs. Engstrom ABoVE SAR Soil Moisture',save_name = 'Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/2017_engstrom_vs_above.png')
    #calculate closest point in df2 to the value in df1
    #df1 = closest_point_index(df1,df2,'lat','lon','Latitude','Longitude','Latitude','Longitude','engstrom_0.06','engstrom_0.12','engstrom_0.2','lbc_0.06','lbc_0.12','lbc_0.2','Index','alt','alt_uncertainty','epsilon1_aug','epsilon1_aug_uncertainty','h','h_uncertainty','mv1_aug','mv1_aug_uncertainty','z1_aug','z1_aug_uncertainty','epsilon2','epsilon2_uncertainty','mv2','mv2_uncertainty',prefix='above')
    #df1 = closest_point_index(df1,df2,'lat','lon','Latitude','Longitude','Latitude','Longitude','grid_code','pointid',prefix='lc')
    #df1 = closest_point_index(df1,df2,'lat','lon','lat','lon','lat','lon','engstrom_0.06','engstrom_0.12','engstrom_0.2','lbc_0.06','lbc_0.12','lbc_0.2','Index','alt','alt_uncertainty','epsilon1_aug','epsilon1_aug_uncertainty','h','h_uncertainty','mv1_aug','mv1_aug_uncertainty','z1_aug','z1_aug_uncertainty','epsilon2','epsilon2_uncertainty','mv2','mv2_uncertainty',prefix='above')

    #saving for speed
    #df1.to_csv('Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/all_above_closest_gaplandfire.csv')


    #calulating distance between coordinates
    #df1 = distance_calc(df1,'lat','lon','above_lat','above_lon','dist')

    #df1.to_csv('Z:/AKSeward/2017_SAR/ABoVE_Soil_Moisture_Products/JBD_Products/closest_dist_all_20191001.csv')


    '''
    #uncomment to get replace SAR_Plot names for consecutive days at Barrow

    df1 = adjusting_SAR_Plot_names(df1,'SAR_Plot',['Barrow_SAR_FC','Barrow_SAR_HC','Barrow_SAR_LC'],['BW_SAR_FC','BW_SAR_HC','BW_SAR_LC'])
    df1.to_csv(closest_dist)
    '''

    #df1 = pd.read_csv(closest_dist,sep=',')
    #landcover_boxplots(df2,'Ecosystem_LU','NVC_CLASS','lbc_6_mean','lbc_6_std','lbc_Counts')
    #boxplots_macrotopology_comparison_plots(df2)

    alt_above_topo_comp(df2)
    #df1 = pd.read_csv(closest_dist)
    #average_SM_at_pixel(df1,'above_Index',savename='Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/Figures/above_vs_hydro/per_pixel_2017_aug_engstrom.png')
    #linear_regression_main(df1,'above_Index',savename='Z:/JDann/Documents/Documents/Julian_Python/SAR_programs_20181003/Figures/above_vs_hydro/per_pixel_2017_aug_regression_depthsonly_TL.png')

    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == "__main__":
    main()
