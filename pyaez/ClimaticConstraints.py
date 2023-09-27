""""
PyAEZ version 2.1.0 (June 2023)
2023: Swun Wunna Htet, Kittiphon Boonma

Modification:

1.  Based of GAEZ appendices, the two different lookup tables of reduction factors
    depending on the annual mean temperatures of >= 20 and < 10 deg C are added.
    With this new tables, new fc3 factors are calculated separately for rainfed and
    irrigated conditions.
    The algorithm will check the annual mean temperature and assess the value from 
    the respective look-up fc3 factor to apply to yield.
2. Added missing logic of linear interpolation for pixels with annual mean temperature between
    10 and 20 deg Celsius to extract fc3 constraint factor.
3. Adding missing logic of linear interpolation for wetness-day-specific agro-climatic constraints
4. Excel sheets of agro-climatic constraint factors are required to provide into the system instead of python file.
    
"""

import numpy as np
import pandas as pd
from pyaez import ETOCalc, UtilitiesCalc

class ClimaticConstraints(object):

    def __init__(self, lat_min, lat_max, elevation, mask = None, no_mask_value = None):
        """Calling object class of Climate Constraints. Providing minimum and maximum latitudes, and mask layer.
        
        Args:
            lat_min (float): Minimum latitude [Decimal Degrees]
            lat_max (float) : Maximum latitude [Decimal Degrees]
            elevation (float/integer): elevation [meters]
            mask [integer]: mask layers [binary, 0/1]
        """
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.elevation = elevation
        
        self.im_height = elevation.shape[0]
        self.im_width = elevation.shape[1]
        self.latitude = UtilitiesCalc.UtilitiesCalc().generateLatitudeMap(lat_min, lat_max, self.im_height, self.im_width)

        self.set_mask = False

        if mask is not None:
            self.mask = mask
            self.no_mask_value = no_mask_value
            self.set_mask = True

    def setClimateData(self, min_temp, max_temp, wind_speed, short_rad, rel_humidity, precip):
        """Load the climatic data into Module III
        
        Args:
            min_temp (3D NumPy, float): Minimum temperature [Celsius]
            max_temp (3D NumPy, float): Maximum temperature [Celsius]
            wind_speed (3D NumPy, float): Windspeed at 2m altitude [m/s]
            short_rad (3D NumPy, float): Radiation [W/m2]
            rel_humidity (3D Numpy, float): Relative humidity [decimal percentage]
            precipitation (3D Numpy, float): Precipitation [mm/day]

        """
        mean_temp = (min_temp + max_temp)/2
        self.ann_mean = np.mean(mean_temp, 2)

        rel_humidity[rel_humidity > 0.99] = 0.99
        rel_humidity[rel_humidity < 0.05] = 0.05
        short_rad[short_rad < 0] = 0
        wind_speed[wind_speed < 0] = 0

        shrt_MJ_m2_day = (short_rad * 3600 * 24)/1000000 # conversion from W/m2 to MJ/m2/day

        self.eto_daily = np.zeros((self.im_height, self.im_width, 365))

        for i in range(self.im_height):
            for j in range(self.im_width):
                obj_eto = ETOCalc.ETOCalc(1, 365, self.latitude[i,j], self.elevation[i,j])
                obj_eto.setClimateData(min_temp[i,j,:], max_temp[i,j,:], wind_speed[i,j,:], shrt_MJ_m2_day[i,j,:], rel_humidity[i,j,:])
                self.eto_daily[i,j,:] = obj_eto.calculateETO()
        
        monthly_precip = np.zeros((self.im_height, self.im_width, 12))
        monthly_eto = np.zeros((self.im_height, self.im_width, 12))
        
        for i in range(self.im_height):
            for j in range(self.im_width):
                monthly_precip[i,j,:] = UtilitiesCalc.UtilitiesCalc().averageDailyToMonthly(precip[i,j,:])
                monthly_eto[i,j,:] = UtilitiesCalc.UtilitiesCalc().averageDailyToMonthly(self.eto_daily[i,j,:])
        
        self.months_P_gte_eto = np.zeros((self.im_height, self.im_width), dtype = np.int8)
        self.months_P_gte_eto = np.sum(monthly_precip >= monthly_eto, axis = 2)

        # releasing memory
        del(mean_temp, rel_humidity, short_rad, wind_speed, shrt_MJ_m2_day, obj_eto, monthly_precip, monthly_eto)

        

    def setReductionFactors(self, file_path):
        """ Load the agro-climatic reduction factors for either rainfed or irrigated conditions.

        Args:
            file_path : String.
                The directory file path of excel sheet in xlsx format storing agro-climatic reduction factor.
                The excel must contain three sheets namely: mean>20, mean<10 and lgpt10.
        
        Return: 
            None.
        """

        main = pd.read_excel(file_path, sheet_name=None)

        if main['lgpt10'].isnull().values.any()==True or main['mean>20'].isnull().values.any()==True or  main['mean<10'].isnull().values.any()==True:
            print('Missing values of reduction factor detected. Excel sheets with no null-values required')
            del(main)
        
        else:
            self.gte20 = main['mean>20']
            self.lt10 = main['mean<10']
            self.lgpt10 = main['lgpt10']
            del(main)
    

    def calculateLGPagc(self, lgp, lgp_equv):
        """ Calculation of adjustted LGP for agro-climatic constraints.
        
        Args:
            lgp (Numerical): Length of Growing Period (Days)
            lgp_equv (Numerical): Equivalent Length of Growing Periods (Days)

        Return:
            lgp_agc (Numerical): Adjusted LGP for agro-climatic constraints. 
        """

        # Wetness indicator calculation logic referred to GAEZ v4 Model Documentation Pg. 72

        if lgp <= 120:
            lgp_agc= min(120, max(lgp, lgp_equv))
            return lgp_agc
        
        elif lgp in range(121,210+1):
            lgp_agc = lgp
            return lgp_agc
            
        elif lgp > 210:
            lgp_agc = max(210, min(lgp, lgp_equv))
            return lgp_agc
    
    def applyClimaticConstraints(self, yield_input, lgp, lgp_equv, lgpt10):

        """
        Args:
        ----------
        yield_input (2D NumPy, int or float): Yield map to apply agro-climatic constraint factor.
        lgp (2D NumPy, int): Length of Growing Period (Days)
        lgp_equv (2D NumPy, int): Equivalent Length of Growing Periods (Days)
        lgpt10 (2D NumPy, int): Thermal Growing Periods at 10 degrees (Days)


        Returns
        -------
        None.
        """

        self.adj_yield = np.zeros((self.im_height, self.im_width), dtype = int)
        original_yld = np.copy(yield_input)
        self.lgp_agc = np.zeros((self.im_height, self.im_width), dtype = int)
        self.fc3 = np.zeros((self.im_height, self.im_width), dtype = np.float16)

        # Middle day of year for each agro-climatic constraints (used for linear interpolation purposes)
        mid_doy = np.array([15,  45,  75, 105, 135, 165, 195, 225, 255, 285, 315, 345, 365])

        for i in range(self.im_height):
            for j in range(self.im_width):

                if self.set_mask:
                    if self.mask[i,j] == self.no_mask_value:
                        continue
                

                # for LPG having 365 or 366, either 365+ or 365- will be selected

                self.lgp_agc[i,j] = self.calculateLGPagc(lgp[i,j], lgp_equv[i,j])

                if self.lgp_agc[i,j] >=365 and self.months_P_gte_eto[i,j] == 12:
                    gte20 = self.gte20.drop(columns = ['365-', 'type'])
                    lt10 = self.lt10.drop(columns=['365-', 'type'])
                else:
                    gte20 = self.gte20.drop(columns = ['365+', 'type'])
                    lt10 = self.lt10.drop(columns=['365+', 'type'])
                
                # Annual mean temperature will select the relevant look-up table
                
                # Case I: ann_mean >= 20
                if self.ann_mean[i,j] >= 20:
                    B_row = B_row = 1 - (gte20.iloc[0].to_numpy() / 100)
                    C_row = 1 - (gte20.iloc[1].to_numpy()/100)
                    D_row = 1 - (gte20.iloc[2].to_numpy()/100)
                
                # Case II: ann_mean <= 10:
                elif self.ann_mean[i,j] <= 10:
                    B_row = 1 - (lt10.iloc[0].to_numpy()/100)
                    C_row = 1 - (lt10.iloc[1].to_numpy()/100)
                    D_row = 1 - (lt10.iloc[2].to_numpy()/100)
                
                # Case III: ann_mean between 10 and 20. Linear interpolation is applied.
                else:

                    # 'B' constraint row interpolation
                    B_row_10 = lt10.iloc[0].to_numpy()
                    B_row_20 = gte20.iloc[0].to_numpy()
                    B_row = np.zeros(B_row_10.shape[0])

                    for e in range(B_row_10.shape[0]):
                        B_row[e] = 1 - ((np.interp(self.ann_mean[i,j], [10,20], [B_row_10[e], B_row_20[e]]))/ 100)
                    
                    
                    # 'C' constraint row interpolation
                    C_row_10 = lt10.iloc[1].to_numpy()
                    C_row_20 = gte20.iloc[1].to_numpy()
                    C_row = np.zeros(B_row_10.shape[0])

                    for e in range(C_row_10.shape[0]):
                        C_row[e] = (1 - (np.interp(self.ann_mean[i,j], [10,20], [C_row_10[e], C_row_20[e]]))/100)
                    

                    # 'D' constraint row interpolation
                    D_row_10 = lt10.iloc[2].to_numpy()
                    D_row_20 = gte20.iloc[2].to_numpy()
                    D_row = np.zeros(B_row_10.shape[0])

                    for e in range(D_row_10.shape[0]):
                        D_row[e] = 1 - ((np.interp(self.ann_mean[i,j], [10,20], [D_row_10[e], D_row_20[e]]))/100)
                
                
                # 'E' constraint row interpolation
                E_row = 1 - ((self.lgpt10.drop(columns= 'type').iloc[0].to_numpy())/100)

                # Start calculation of agro-climatic constraints
                # 1: find agro-climatic factors of its corresponding interval of wetness days 
                # 2: select the most limiting factor amongst 'b', 'c', 'd' and 'e' constraints

                B = np.interp(self.lgp_agc[i,j], mid_doy, B_row)
                C = np.interp(self.lgp_agc[i,j], mid_doy, C_row)
                D = np.interp(self.lgp_agc[i,j], mid_doy, D_row)
                E = np.interp(lgpt10[i,j], mid_doy, E_row)

                self.fc3[i,j] = np.round(np.min([B*C*D, E]), 2)

                self.adj_yield[i,j] = int(np.round(original_yld[i,j] * self.fc3[i,j], 0))


    def getClimateAdjustedYield(self):
        """
        Generate yield map adjusted with agro-climatic constraints.

        Returns
        -------
        TYPE: 2-D numpy array.
            Agro-climatic constraint applied yield.

        """
        return self.adj_yield
    
    def getClimateReductionFactor(self):
        """
        Generates agro-climatic constraint map (fc3) applied to unconstrainted 
        yield.

        Returns
        -------
        TYPE : 2-D numpy array.
            Agro-climatic constraint map (fc3).

        """
        return self.fc3

#----------------- End of file -------------------------#
