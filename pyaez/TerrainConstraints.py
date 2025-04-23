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
4.  Terrain reduction calculation is now changed with the slope distribution class layers.
"""

import numpy as np
import pandas as pd
import numba as nb
from pyaez.UtilitiesCalc import averageDailyToMonthly

class TerrainConstraints(object):

    def importTerrainReductionSheet(self, irr_file_path, rain_file_path):
        """
        (MANDATORY FUNCTION) Upload the terrain reduction factor excel sheets into Module V object class.
        
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
    
    def setStudyAreaMask(self, admin_mask, no_data_value):
        """
        (OPTIONAL FUNCTION) Set clipping mask of the area of interest.

        Args:
            admin_mask (2D NumPy/Binary): mask to extract only region of interest
            no_data_value (int): pixels with this value will be omitted during PyAEZ calculations
        """    
        self.im_mask = admin_mask
        self.nodata_val = no_data_value
        self.set_mask = True

    def setClimateTerrainData(self, precipitation, slope):
        """
        (MANDATORY FUNCTION) Import precipitation and percent slope data into the object class.
        Args:
            precipitation (3D-NumPy Array): daily or monthly precipitation (Unit: mm/day or mm/month)
            slope (2D-NumPy Array): percent slope (Unit: %)
            mask (2D-NumPy Array): mask layer
            no_val_mask (int): pixel value of mask layer to omit calculation.
        Return:
            None.
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
        self.slope = classifySlopeDistribution(slope) 
        self.slope[np.isnan(self.slope)] = 0 # This suppresses warning with NaN values


    def calculateFI(self):
        """
        (MANDATORY FUNCTION) Calculation of Fournier Index

        Args:
            None.
        Return:
            None.
        """
        # calculation of Fournier index

        sum_Psquare = np.sum(np.square(self.prec_monthly), axis=2)
        sum_P = np.sum(self.prec_monthly, axis=2)

        self.FI = np.multiply(12, (sum_Psquare / sum_P), where= sum_P !=0, out = np.zeros(sum_Psquare.shape))

    def getFI(self):
        """
        Getting the result of Fournier Index.
        
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

        yield_final = np.zeros(yield_in.shape)
        self.terrain_fct = np.zeros(yield_in.shape)

        if irr_or_rain == 'I':
            terrain_factor = self.irr_np
        else:
            terrain_factor = self.rain_np

        # Define FI slope classes
        fi_classes = [(0,1300), (1300, 1800), (1800, 2200), (2200, 2500), (2500, 2700), 'fi>45']

        for i in range(self.im_height):
            for j in range(self.im_width):
                

                if self.set_mask:
                    if self.im_mask[i, j] == self.nodata_val:
                        continue 
                
                if i==0 or i==self.im_height-1 or j ==0 or j== self.im_width-1:
                    continue
                
                # select an array of slope class distribution
                slp_arr = self.slope[i,j,:]

                # get FI class-based terrain reduction factors
                for k in range(len(fi_classes)):

                    if k == 5:
                        terrain_fct_arr = terrain_factor[k]
                    else:
                        if self.FI[i,j] in range(fi_classes[k][0], fi_classes[k][0]):
                            terrain_fct_arr = terrain_factor[k]
                            break
                    
                # Normalize the slope classs percentages as decimals
                normalized_slp = slp_arr/100.

                # Application of weighted average of the terrain ratings from all slope classes
                fc5 = np.sum(np.multiply(normalized_slp, terrain_fct_arr))
                self.terrain_fct[i,j] = fc5

                # Applying the yield reduction due to terrain condition
                yield_final[i,j] = yield_in[i,j] * (fc5 / 100)

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

def classifySlopeDistribution(slp):
    """
    Calculates the eight slope class distribution based on GAEZ framework.
    
    Arg:
        slope [2-D NumPy Array]: Percent slope map (Unit = Percent)
    Return:
        slp_class [3D NumPy Array]: eight slope class distribution.
                                    Class 1: 0 - 0.5 % (very flat)
                                    Class 2: 0.5 - 2 % (flat)
                                    Class 3: 2 - 5 % (gently sloping)
                                    Class 4: 5 - 10 % (undulating)
                                    Class 5: 10 - 15 % (rolling)
                                    Class 6: 15 - 30 % (hilly)
                                    Class 7: 30 - 45 % (steep)
                                    Class 8: > 45 % (very steep)
    """
    row,col = slp.shape

    slp_class = np.zeros((row, col, 8), dtype = int)
    
    for i in range(row):
        for j in range(col):

            if i==0 or i==row-1 or j ==0 or j== col-1:
                continue

            slA = slp[i-1,j-1] # top-left
            slB = slp[i-1,j] # top-middle
            slC = slp[i-1,j+1] # top-right
            slD = slp[i,j-1] # middle-left
            slE = slp[i,j] # middle
            slF = slp[i,j+1] # middle-right
            slG = slp[i+1,j-1] # bottom-left
            slH = slp[i+1,j] # bottom-middle
            slI = slp[i+1,j+1] # bottom-right

            # Start counting the corresponding slope class
            for k in [slA, slB, slC, slD, slE, slF, slG, slH, slI]:
                
                # C1: 0 - 0.5 % (very flat)
                if k >=0 and k< 0.5:
                    slp_class[i,j,0] =  slp_class[i,j,0] + 1
                # C2: 0.5 - 2 % (flat)
                elif k>=0.5 and k<2:
                    slp_class[i,j,1] =  slp_class[i,j,1] + 1
                # C3: 2 - 5 % (gently sloping)
                elif k>=2 and k<5:
                    slp_class[i,j,2] =  slp_class[i,j,2] + 1
                # C4: 0 - 0.5 % (undulating)
                elif k>=5 and k<10:
                    slp_class[i,j,3] =  slp_class[i,j,3] + 1
                # C5: 0 - 0.5 % (rolling)
                elif k>=10 and k<15:
                    slp_class[i,j,4] =  slp_class[i,j,4] + 1
                # C6: 0 - 0.5 % (hilly)
                elif k>=15 and k<30:
                    slp_class[i,j,5] =  slp_class[i,j,5] + 1
                # C7: 0 - 0.5 % (steep)
                elif k>=30 and k<45:
                    slp_class[i,j,6] =  slp_class[i,j,6] + 1
                # Cl8: 0 - 0.5 % (very steep)
                elif k >=45:
                    slp_class[i,j,7] =  slp_class[i,j,7] + 1
    
    total = np.sum(slp_class, axis = 2)
    total = total[:,:,np.newaxis]
    total = np.repeat(total, 8, axis = 2)
    slp_class = np.divide(slp_class, total ,where = total >0, out = np.zeros((row, col, 8)))
    slp_class = np.round(slp_class*100, decimals= 1)
    return slp_class

#----------------------------------------------------END OF FILE --------------------------------------------------------------#