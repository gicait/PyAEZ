"""
PyAEZ version 2.3 (Apr 2025)
2020: N. Lakmal Deshapriya
2023 (Dec): Swun Wunna Htet
2025 (Apr): Swun Wunna Htet

Modifications:
1.  Excel sheet integration is now added to the routine.
2.  Algorithm will check whether daily or monthly preciptation is provided and 
    calculate Fournier Index accordingly.
3.  Terrain Reduction Factor can now be returned as raster map.
4.  Numba enhancements are done to functions available for optimization.
"""

import numpy as np
import pandas as pd
import numba as nb
from pyaez.UtilitiesCalc import averageDailyToMonthly

class TerrainConstraints(object):

    def importTerrainReductionSheet(self, irr_file_path, rain_file_path):
        """
        Upload the terrain reduction factor excel sheets into Module V object class.
        
        Args:
            rain_file_path (String): file path of terrain reduction factor for rainfed conditions (Excel sheet)
            irr_file_path (String): file path of terrain reduction factor for irrigated conditions (Excel sheet)
        Return:
            None.
        """
        # reading each individual excel sheet databases
        rain_df = pd.read_excel(rain_file_path)
        irr_df = pd.read_excel(irr_file_path)

        self.rain_slope_class = np.array([eval(rain_df.columns.to_numpy()[x+1]) for x in range(rain_df.columns.to_numpy()[1:].shape[0])])
        self.irr_slope_class = np.array([eval(irr_df.columns.to_numpy()[x+1]) for x in range(irr_df.columns.to_numpy()[1:].shape[0])])

        self.rain_FI_class = np.array([eval(rain_df['Classes'].to_numpy()[x]) for x in range(rain_df['Classes'].to_numpy().shape[0])])
        self.irr_FI_class = np.array([eval(irr_df['Classes'].to_numpy()[x]) for x in range(irr_df['Classes'].to_numpy().shape[0])])
        # reduction factor look-up table
        self.rain_np = rain_df.to_numpy()[:,1:].astype(np.float16)
        self.irr_np = irr_df.to_numpy()[:,1:].astype(np.float16)

    def setClimateTerrainData(self, precipitation, slope):
        """
        Import precipitation and percent slope data into the object class.
        Args:
            precipitation (3-D NumPy array): daily or monthly precipitation (Unit: mm/day or mm/month)
            slope (2-D NumPy array): percent slope (Unit: %)
        """
        self.im_height = slope.shape[0]
        self.im_width = slope.shape[1]
        leap_year = False
        
        if precipitation.shape[2] == 12:
            self.prec_monthly = precipitation
        elif precipitation.shape[2] in [365, 366]:
            self.prec_monthly = np.zeros((self.im_height,self.im_width,12))

            
            if precipitation.shape[2] == 365:
                pass
            elif precipitation.shape[2] == 366:
                leap_year = True

            for i in range(self.prec_monthly.shape[0]):
                for j in range(self.prec_monthly.shape[1]):
                    self.prec_monthly[i,j,:] = averageDailyToMonthly(precipitation[i,j,:], leap_year)
        else:
            raise ValueError('Time dimension of input wrong. Please check the input.')

        # slope is now 3D NumPy Array (Slope distribution classes)
        self.slope = slope 
        self.slope[np.isnan(self.slope)] = 0 # This suppresses warning with NaN values
        # slope distribution data type reformatting
        self.slope = self.slope.astype(np.float16)

    def calculateFI(self):
        """Calculation of Fournier Index
        Args:
            None.
        Return:
            None.
        """
        # calculation of Fournier index

        sum_Psquare = np.sum(np.square(self.prec_monthly), axis=2)
        sum_P = np.sum(self.prec_monthly, axis=2)

        self.FI = np.multiply(12, (sum_Psquare / sum_P), where= sum_P !=0)
        self.FI[np.isnan(self.FI)] = 0 # This suppresses warning with NaN values

    def getFI(self):
        """Getting the result of Fournier Index.
        
        Args:
            None.
        Return:
            FI (2-D NumPy Array): Fournier Index
        """
        # returning Fournier index

        return self.FI

    def applyTerrainConstraints(self, yield_in, irr_or_rain):

        """
        Apply the terrain reduction factors to the input yield map based on selected water supply setting.
        Based on it, the terrain reduction factor will be calculated to apply yield reduction.
        
        Args:
            yield_in (2-D NumPy Array): input yield, either rainfed or irrigated (Unit: kg/ha)
            irr_or_rain (String): either provide I (Irrigated) or R (Rainfed)
        
        Return:
            final_yield (2-D NumPy Array): terrain-adjusted yield (rainfed or irrigated)
        """

        if irr_or_rain == 'I':
            crop_P = self.irr_np
            FI_class = self.irr_FI_class
            Slope_class = self.irr_slope_class
            Terrain_factor = self.irr_np
        elif irr_or_rain == 'R':
            crop_P = self.rain_np
            FI_class = self.rain_FI_class
            Slope_class = self.rain_slope_class
            Terrain_factor = self.rain_np

        yield_final = np.copy(yield_in)
        self.terrain_fct = np.zeros(yield_in.shape)
        
        FI_iter = list(enumerate(FI_class))
        for i in range(self.im_height):
            for j in range(self.im_width):

                slp_arr = self.slope[i,j,:]

                # find relevant FI-class specific terrain factor for all slope classes
                for k in range(len(FI_iter)):
                    index, intval = FI_iter[k]
                    if np.logical_and([self.FI[i,j] >= intval[0]], [self.FI[i,j] < intval[1]]):
                        fiidx = index
                        tfct_arr = Terrain_factor[fiidx]
                        break
                    else:
                        pass
                
                fc5 = np.divide(tfct_arr, slp_arr, where= slp_arr >0, out = np.zeros(8, dtype = np.float16)) 

                # each terrain factor is adjusted with the slope distribution classes and summed up.
                fc5 = np.sum(fc5)
                yield_final[i,j] =  fc5 * yield_in[i,j]
                self.terrain_fct[i,j] = fc5

        return yield_final
    
    def getTerrainReductionFactor(self):
        """
        Obtain the terrain reduction factor from the previous yield reduction calculation.
        Terrain reduction factor ranges from 0 (Not suitable) to 1 (Most Suitable).
        
        Note: Based on the setting from applyTerrainConstraint function, the reduction factor map
        corresponds to either rainfed or irrigated.
        
        Args:
            None.
        Return:
            fc5 (2-D NumPy Array): Terrain Reduction factor (0 : Unsuitable, 1 = Very Suitable)"""
        
        return self.terrain_fct

#----------------------------------------------End of File-----------------------------------------------#
#--------------------------------------  END OF TERRAIN CONSTRAINTS  ---------------------------------------#
