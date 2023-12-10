"""
PyAEZ version 2.2 (Dec 2023)
2020: N. Lakmal Deshapriya
2023 (Dec): Swun Wunna Htet

Modifications:
1.  Excel sheet integration is now added to the routine.
2.  Algorithm will check whether daily or monthly preciptation is provided and 
    calculate Fournier Index accordingly.
3.  Terrain Reduction Factor can now be returned as raster map.
"""

import numpy as np
import pandas as pd
from pyaez import UtilitiesCalc


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
        self.rain_np = rain_df.to_numpy()[:,1:]
        self.irr_np = irr_df.to_numpy()[:,1:]

    def setClimateTerrainData(self, precipitation, slope):
        """
        Import precipitation and percent slope data into the object class.
        Args:
            precipitation (3-D NumPy array): daily or monthly precipitation (Unit: mm/day or mm/month)
            slope (2-D NumPy array): percent slope (Unit: %)
        """
        self.im_height = slope.shape[0]
        self.im_width = slope.shape[1]
        
        if precipitation.shape[2] == 12:
            self.prec_monthly = precipitation
        else:
            self.prec_monthly = np.zeros((self.im_height,self.im_width,12))
            for i in range(self.prec_monthly.shape[0]):
                for j in range(self.prec_monthly.shape[1]):
                    self.prec_monthly[i,j,:] = UtilitiesCalc.UtilitiesCalc().averageDailyToMonthly(precipitation[i,j,:])


        self.slope = slope # Percentage Slope
        self.slope[np.isnan(self.slope)] = 0 # This suppresses warning with NaN values



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


        FI_count = -1
        for FI_cls1 in FI_class:
            FI_count = FI_count + 1

            slope_count = -1
            for slope_cls1 in Slope_class:
                slope_count = slope_count + 1

                FI_idx = np.logical_and(FI_cls1[0]<=self.FI, self.FI<=FI_cls1[1])
                slope_idx = np.logical_and(slope_cls1[0]<=self.slope, self.slope<=slope_cls1[1])
                temp_idx = np.logical_and(FI_idx, slope_idx)

                yield_final[temp_idx] = yield_in[temp_idx] * (Terrain_factor[FI_count][slope_count] / 100)
                self.terrain_fct[temp_idx] = Terrain_factor[FI_count][slope_count] / 100
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
            fc5 (2-D NumPy Array): Terrain Reduction factor"""
        
        return self.terrain_fct

#----------------------------------------------End of File-----------------------------------------------#
#--------------------------------------  END OF TERRAIN CONSTRAINTS  ---------------------------------------#
