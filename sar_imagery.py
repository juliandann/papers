import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from PIL import Image
import gdal
from osgeo import gdal_array
def main():

    int_tif = 'Z:/AKSeward/DEM/Lidar/2017_UAS/DEMs/DEM_Intensity_Rasters/9-17_top_allpts_0.2m.tif'

    ds = gdal.Open(int_tif)
    print(ds.GetMetadataItem(''))
    print(ds.RasterCount)
    for i in range(1,1+ds.RasterCount):
        print('Band Number: ',i)
        band = ds.GetRasterBand(i)
        arr =band.ReadAsArray()
        print(arr.size)
        arr = pd.DataFrame(arr)
        #arr.to_csv('Z:/JDann/Documents/Documents/arr.csv')
        arr[arr == 100.0]=np.nan
        print(arr.values.shape)
        print('MAX:',np.nanmax(arr.values))
        print('Min:',np.nanmin(arr.values))

        #find datatype to use in GDAL plotting
        image_datatype = ds.GetRasterBand(1).DataType

        #create empty array with pixel row,column,band values
        image = np.zeros((ds.RasterYSize, ds.RasterXSize, ds.RasterCount+1),
                 dtype=gdal_array.GDALTypeCodeToNumericTypeCode(image_datatype))

        # Read in the band's data into the third dimension of our array
        image[:, :, i] = arr

        [cols, rows] = arr.shape
        plt.imshow(image[:, :, i])
        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    main()
