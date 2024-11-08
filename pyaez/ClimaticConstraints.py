""""
PyAEZ version 2.2(Dec 2023)
2022- 2023: Swun Wunna Htet, Kittiphon Boonma
2023 (Dec) : Swun Wunna Htet

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
from pyaez.ETOCalc import calculateETONumba
from pyaez.UtilitiesCalc import generateLatitudeMap, interpMonthlyToDaily, averageDailyToMonthly
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
        self.set_daily = False
        self.set_monthly = False
        
        self.im_height = elevation.shape[0]
        self.im_width = elevation.shape[1]
        self.latitude = generateLatitudeMap(lat_min, lat_max, self.im_height, self.im_width)

        self.set_mask = False

        if mask is not None:
            self.mask = mask
            self.no_mask_value = no_mask_value
            self.set_mask = True

    def setClimateData(self, min_temp, max_temp, wind_speed, short_rad, rel_humidity, precip):
        """Load the DAILY or MONTHLY climatic data into Module III
        
        Args:
            min_temp (3D NumPy, float): Minimum temperature [Celsius]
            max_temp (3D NumPy, float): Maximum temperature [Celsius]
            wind_speed (3D NumPy, float): Windspeed at 2m altitude [m/s]
            short_rad (3D NumPy, float): Radiation [W/m2]
            rel_humidity (3D Numpy, float): Relative humidity [decimal percentage]
            precipitation (3D Numpy, float): Precipitation [mm/day]

        """
        rel_humidity[rel_humidity > 0.99] = 0.99
        rel_humidity[rel_humidity < 0.05] = 0.05
        short_rad[short_rad < 0] = 0
        wind_speed[wind_speed < 0] = 0

        if np.all(min_temp.shape[2] ==12 and max_temp.shape[2] ==12 and wind_speed.shape[2] ==12
                           and short_rad.shape[2] ==12 and rel_humidity.shape[2] ==12 and precip.shape[2] ==12):

            totalPrec_daily = np.zeros((self.im_height, self.im_width, 365))
            minT_daily = np.zeros((self.im_height, self.im_width, 365))
            maxT_daily = np.zeros((self.im_height, self.im_width, 365))
            radiation_daily = np.zeros((self.im_height, self.im_width, 365))
            rel_humidity_daily = np.zeros((self.im_height, self.im_width, 365))
            wind_daily = np.zeros((self.im_height, self.im_width, 365))

            for i_row in range(self.im_height):
                for i_col in range(self.im_width):

                    if self.set_mask:
                        if self.mask[i_row, i_col] == self.no_mask_value:
                            continue

                    totalPrec_daily[i_row, i_col, :] = interpMonthlyToDaily(precip[i_row, i_col,:], 1, 365, no_minus_values=True)
                    minT_daily[i_row, i_col, :] = interpMonthlyToDaily(min_temp[i_row, i_col,:], 1, 365)
                    maxT_daily[i_row, i_col, :] = interpMonthlyToDaily(max_temp[i_row, i_col,:], 1, 365)
                    radiation_daily[i_row, i_col, :] = (interpMonthlyToDaily(short_rad[i_row, i_col,:], 1, 365, no_minus_values=True)* 3600 * 24)/1000000
                    wind_daily[i_row, i_col, :] = interpMonthlyToDaily(wind_speed[i_row, i_col,:], 1, 365, no_minus_values=True)
                    rel_humidity_daily[i_row, i_col, :] = interpMonthlyToDaily(rel_humidity[i_row, i_col,:], 1, 365, no_minus_values=True)
            
        else:
            totalPrec_daily = precip
            minT_daily = min_temp
            maxT_daily = max_temp
            radiation_daily = (short_rad * 3600 * 24)/1000000 # conversion from W/m2 to MJ/m2/day
            rel_humidity_daily = rel_humidity.copy()
            wind_daily = wind_speed.copy()

            
        mean_temp = (minT_daily + maxT_daily)/2
        
        # The minimum temperature of the 12 monthly mean temperatures
        self.min_T = np.zeros((self.im_height, self.im_width))
        for i in range(self.im_height):
            for j in range(self.im_width):
                month12_temp = averageDailyToMonthly(mean_temp[i,j,:])
                self.min_T[i,j] = np.min(month12_temp)

        self.eto_daily = np.zeros((self.im_height, self.im_width, 365))

        for i in range(self.im_height):
            for j in range(self.im_width):
                self.eto_daily[i,j,:] = calculateETONumba(1, 365, self.latitude[i,j], self.elevation[i,j], minT_daily[i,j,:], maxT_daily[i,j,:],
                                                           wind_daily[i,j,:], radiation_daily[i,j,:], rel_humidity_daily[i,j,:])
        
        monthly_precip = np.zeros((self.im_height, self.im_width, 12))
        monthly_eto = np.zeros((self.im_height, self.im_width, 12))
        
        for i in range(self.im_height):
            for j in range(self.im_width):
                monthly_precip[i,j,:] = averageDailyToMonthly(totalPrec_daily[i,j,:])
                monthly_eto[i,j,:] = averageDailyToMonthly(self.eto_daily[i,j,:])
        
        self.months_P_gte_eto = np.zeros((self.im_height, self.im_width), dtype = np.int8)
        self.months_P_gte_eto = np.sum(monthly_precip >= monthly_eto, axis = 2)

        # releasing memory
        del(mean_temp, totalPrec_daily, minT_daily, maxT_daily, radiation_daily, rel_humidity_daily, wind_daily,
             monthly_eto, monthly_precip)

        

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
            lgp_agc= min(120, max(lgp, lgp_equv)) # correct
            return lgp_agc
        
        elif lgp in range(121,210+1):
            lgp_agc = lgp # correct
            return lgp_agc
            
        elif lgp > 210:
            lgp_agc = max(210, min(lgp, lgp_equv)) # correct
            return lgp_agc
    
    def applyClimaticConstraints(self, yield_input, lgp, lgp_equv, lgpt10, omit_yld_0= False):

        """
        Args:
        ----------
        yield_input (2D NumPy, int or float): Yield map to apply agro-climatic constraint factor.
        lgp (2D NumPy, int): Length of Growing Period (Days)
        lgp_equv (2D NumPy, int): Equivalent Length of Growing Periods (Days)
        lgpt10 (2D NumPy, int): Thermal Growing Periods at 10 degrees (Days)
        omit_yld_0 (Boolean): Any zero yield areas will not be calculated. Default is False.


        Returns
        -------
        None.
        """

        self.adj_yield = np.zeros((self.im_height, self.im_width), dtype = int)
        original_yld = np.copy(yield_input)
        self.lgp_agc = np.zeros((self.im_height, self.im_width), dtype = int)
        self.fc3 = np.zeros((self.im_height, self.im_width), dtype = np.float16)

        # Middle day of year for each agro-climatic constraints (used for linear interpolation purposes)
        mid_doy = np.array([0, 15,  45,  75, 105, 135, 165, 195, 225, 255, 285, 315, 345, 365]) # total 14 interval points

        for i in range(self.im_height):
            for j in range(self.im_width):

                if self.set_mask:
                    if self.mask[i,j] == self.no_mask_value:
                        continue
                
                if omit_yld_0:

                    if original_yld[i,j] == 0:
                        continue

                # for LPG having 365 or 366, either 365+ or 365- will be selected

                self.lgp_agc[i,j] = self.calculateLGPagc(lgp[i,j], lgp_equv[i,j])

                if self.lgp_agc[i,j] >=365 and self.months_P_gte_eto[i,j] == 12:
                    gte20 = (self.gte20.drop(columns = ['365-', 'type'])).to_numpy()
                    lt10 = (self.lt10.drop(columns=['365-', 'type'])).to_numpy()

                else:
                    gte20 = (self.gte20.drop(columns = ['365+', 'type'])).to_numpy()
                    lt10 = (self.lt10.drop(columns=['365+', 'type'])).to_numpy()
                

                # Appending zero reduction factor for zero LGPagc
                # gte20 = np.append(0,gte20)
                # lt10 = np.append(0,lt10)
                
                # Annual mean temperature will select the relevant look-up table
                
                # Case I: ann_mean >= 20
                if self.min_T[i,j] >= 20:
                    B_row = 1 - (np.append(0, gte20[0,:]) / 100)
                    C_row = 1 - (np.append(0, gte20[1,:])/100)
                    D_row = 1 - (np.append(0, gte20[2,:])/100)
                
                # Case II: ann_mean <= 10:
                elif self.min_T[i,j] <= 10:
                    B_row = 1 - (np.append(0, lt10[0,:])/100)
                    C_row = 1 - (np.append(0, lt10[1,:])/100)
                    D_row = 1 - (np.append(0, lt10[2,:])/100)
                
                # Case III: ann_mean between 10 and 20. Linear interpolation is applied.
                else:

                    # 'B' constraint row interpolation
                    B_row_10 = np.append(0, lt10[0,:])
                    B_row_20 = np.append(0, gte20[0,:])
                    B_row = np.zeros(B_row_10.shape[0])

                    for e in range(B_row_10.shape[0]):
                        B_row[e] = 1 - ((np.interp(self.min_T[i,j], [10,20], [B_row_10[e], B_row_20[e]]))/ 100)
                    
                    
                    # 'C' constraint row interpolation
                    C_row_10 = np.append(0, lt10[1,:])
                    C_row_20 = np.append(0, gte20[1,:])
                    C_row = np.zeros(B_row_10.shape[0])

                    for e in range(C_row_10.shape[0]):
                        C_row[e] = (1 - (np.interp(self.min_T[i,j], [10,20], [C_row_10[e], C_row_20[e]]))/100)
                    

                    # 'D' constraint row interpolation
                    D_row_10 = np.append(0, lt10[2,:])
                    D_row_20 = np.append(0, gte20[2,:])
                    D_row = np.zeros(B_row_10.shape[0])

                    for e in range(D_row_10.shape[0]):
                        D_row[e] = 1 - ((np.interp(self.min_T[i,j], [10,20], [D_row_10[e], D_row_20[e]]))/100)
                
                
                # 'E' constraint row interpolation
                E_row = np.append(0, self.lgpt10.drop(columns= 'type').iloc[0].to_numpy())
                E_row = 1 - (E_row/100)

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
    
    # Developer's Note: This code snippet below is to investigate the intermediate values used in Module III.
    #                   Do not remove this code part.

    def getintermediate(self, i, j, yield_input, lgp, lgp_equv, lgpt10):
        """
        Generates intermediate values of Module III

        Returns
        -------
        TYPE : a python list.
            [].

        """
        lgp_agc = self.calculateLGPagc(lgp, lgp_equv)

        # Middle day of year for each agro-climatic constraints (used for linear interpolation purposes)
        mid_doy = np.array([0, 15,  45,  75, 105, 135, 165, 195, 225, 255, 285, 315, 345, 365]) # total 14 interval points


        if lgp_agc >=365 and self.months_P_gte_eto[i,j] == 12:
            gte20 = (self.gte20.drop(columns = ['365-', 'type'])).to_numpy()
            lt10 = (self.lt10.drop(columns=['365-', 'type'])).to_numpy()
            test = '365+'

        else:
            gte20 = (self.gte20.drop(columns = ['365+', 'type'])).to_numpy()
            lt10 = (self.lt10.drop(columns=['365+', 'type'])).to_numpy()
            test = '365-'
        
        
        # Annual mean temperature will select the relevant look-up table
        
        # Case I: ann_mean >= 20
        if self.min_T[i,j] >= 20:
            B_row = 1 - (np.append(0, gte20[0,:]) / 100)
            C_row = 1 - (np.append(0, gte20[1,:])/100)
            D_row = 1 - (np.append(0, gte20[2,:])/100)
        
        # Case II: ann_mean <= 10:
        elif self.min_T[i,j] <= 10:
            B_row = 1 - (np.append(0, lt10[0,:])/100)
            C_row = 1 - (np.append(0, lt10[1,:])/100)
            D_row = 1 - (np.append(0, lt10[2,:])/100)
        
        # Case III: ann_mean between 10 and 20. Linear interpolation is applied.
        else:

            # 'B' constraint row interpolation
            B_row_10 = np.append(0, lt10[0,:])
            B_row_20 = np.append(0, gte20[0,:])
            B_row = np.zeros(B_row_10.shape[0])

            for e in range(B_row_10.shape[0]):
                B_row[e] = 1 - ((np.interp(self.min_T[i,j], [10,20], [B_row_10[e], B_row_20[e]]))/ 100)
            
            
            # 'C' constraint row interpolation
            C_row_10 = np.append(0, lt10[1,:])
            C_row_20 = np.append(0, gte20[1,:])
            C_row = np.zeros(B_row_10.shape[0])

            for e in range(C_row_10.shape[0]):
                C_row[e] = (1 - (np.interp(self.min_T[i,j], [10,20], [C_row_10[e], C_row_20[e]]))/100)
            

            # 'D' constraint row interpolation
            D_row_10 = np.append(0, lt10[2,:])
            D_row_20 = np.append(0, gte20[2,:])
            D_row = np.zeros(B_row_10.shape[0])

            for e in range(D_row_10.shape[0]):
                D_row[e] = 1 - ((np.interp(self.min_T[i,j], [10,20], [D_row_10[e], D_row_20[e]]))/100)
        
        
        # 'E' constraint row interpolation
        E_row = np.append(0, self.lgpt10.drop(columns= 'type').iloc[0].to_numpy())
        E_row = 1 - (E_row/100)

        # Start calculation of agro-climatic constraints
        # 1: find agro-climatic factors of its corresponding interval of wetness days 
        # 2: select the most limiting factor amongst 'b', 'c', 'd' and 'e' constraints

        B = np.interp(lgp_agc , mid_doy, B_row)
        C = np.interp(lgp_agc , mid_doy, C_row)
        D = np.interp(lgp_agc , mid_doy, D_row)
        E = np.interp(lgp_agc , mid_doy, E_row)

        fc3 = np.round(np.min([B*C*D, E]), 2)

        adj_yld  = int(np.round(yield_input * fc3, 0))

        return [self.latitude[i,j], self.elevation[i,j], self.months_P_gte_eto[i,j], self.min_T[i,j], test, B, C, D, E, fc3, adj_yld, mid_doy, B_row, C_row, D_row, E_row, lgp_agc]
#----------------- End of file -------------------------#
