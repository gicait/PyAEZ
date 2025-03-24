""""
PyAEZ version 2.3(Apr 2024)
2022- 2023: Swun Wunna Htet, Kittiphon Boonma
2023 (Dec) : Swun Wunna Htet
2024 (Apr) : Swun Wunna Htet, Dwijendra Das

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
5. A new object class (ProcessExcelWrapper) is added for excel sheet creation for agro-climatic constraints.
    
"""
import numpy as np
import pandas as pd
from pyaez.ETOCalc import calculateETONumba
from pyaez.UtilitiesCalc import generateLatitudeMap, interpMonthlyToDaily, averageDailyToMonthly
import warnings
warnings.filterwarnings('ignore')

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
        self.leap_year = False
        self.set_mask = False

        if mask is not None:
            self.im_mask = mask
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

        # Time dimension checkpoint
        doy = None
        if np.all(min_temp.shape[2] ==12 and max_temp.shape[2] ==12 and wind_speed.shape[2] ==12
                and short_rad.shape[2] ==12 and rel_humidity.shape[2] ==12 and precip.shape[2] ==12):
            self.set_monthly = True
            doy = 365
        elif np.all(min_temp.shape[2] ==365 and max_temp.shape[2] ==365 and wind_speed.shape[2] ==365
                and short_rad.shape[2] ==365 and rel_humidity.shape[2] ==365 and precip.shape[2] ==365):
            doy = 365
        elif np.all(min_temp.shape[2] ==366 and max_temp.shape[2] ==366 and wind_speed.shape[2] ==366
                and short_rad.shape[2] ==366 and rel_humidity.shape[2] ==366 and precip.shape[2] ==366):
            doy = 366
            self.leap_year = True
        else:
            raise ValueError('Time Dimension of climate data must be 12, 365 or 366. Please check your input data.')

        self.meanT_daily = np.zeros((self.im_height, self.im_width, doy))
        self.totalPrec_daily = np.zeros((self.im_height, self.im_width, doy))
        self.minT_daily = np.zeros((self.im_height, self.im_width, doy))
        self.maxT_daily = np.zeros((self.im_height, self.im_width, doy))
        self.shortrad_daily = np.zeros((self.im_height, self.im_width, doy))
        self.wind_daily = np.zeros((self.im_height, self.im_width, doy))
        self.rel_humidity_daily = np.zeros((self.im_height, self.im_width, doy))
        self.eto_daily = np.zeros((self.im_height, self.im_width, doy))
        self.shortrad_daily_MJm2day = np.zeros((self.im_height, self.im_width, doy))

        self.min_T = np.zeros((self.im_height, self.im_width))
        self.months_P_gte_eto = np.zeros((self.im_height, self.im_width))
        monthly_precip = np.zeros((self.im_height, self.im_width, 12))
        monthly_eto = np.zeros((self.im_height, self.im_width, 12))

        if self.set_monthly:

            for i_row in range(self.im_height):
                for i_col in range(self.im_width):

                    if self.set_mask:
                        if self.im_mask[i_row, i_col] == self.no_mask_value:
                            continue
                    
                    Tm = (max_temp[i_row, i_col, :] + min_temp[i_row, i_col, :])/2
                    self.meanT_daily[i_row, i_col, :] = interpMonthlyToDaily(Tm, 1, doy)
                    self.totalPrec_daily[i_row, i_col, :] = interpMonthlyToDaily(precip[i_row, i_col,:], 1, doy, no_minus_values=True)
                    self.minT_daily[i_row, i_col, :] = interpMonthlyToDaily(min_temp[i_row, i_col,:], 1, doy)
                    self.maxT_daily[i_row, i_col, :] = interpMonthlyToDaily(max_temp[i_row, i_col,:], 1, doy)
                    self.shortrad_daily[i_row, i_col, :] = interpMonthlyToDaily(short_rad[i_row, i_col,:], 1, doy, no_minus_values=True)
                    self.wind_daily[i_row, i_col, :] = interpMonthlyToDaily(wind_speed[i_row, i_col,:], 1, doy, no_minus_values=True)
                    self.rel_humidity_daily[i_row, i_col, :] = interpMonthlyToDaily(rel_humidity[i_row, i_col,:], 1, doy, no_minus_values=True)

                    monthly_precip[i_row, i_col, : ] = precip[i_row, i_col, :]
                    self.min_T[i_row, i_col] = np.nanmin(Tm)
                
        else:
            self.meanT_daily = (min_temp+ max_temp)/2
            self.totalPrec_daily = precip
            self.minT_daily = min_temp
            self.maxT_daily = max_temp
            self.shortrad_daily = short_rad
            self.wind_daily = wind_speed
            self.rel_humidity_daily = rel_humidity

            for i_row in range(self.im_height):
                for i_col in range(self.im_width):
                    monthly_precip[i_row, i_col, : ]=  averageDailyToMonthly(self.totalPrec_daily[i_row, i_col, :], self.leap_year)        
                    monthly_Tm = averageDailyToMonthly(self.meanT_daily[i_row, i_col, :], self.leap_year)
                    self.min_T[i_row, i_col] = np.nanmin(monthly_Tm)
        # ET0 calculation
        self.shortrad_daily_MJm2day = (self.shortrad_daily*3600*24)/1000000 # convert w/m2 to MJ/m2/day
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                self.eto_daily[i_row, i_col, :] = calculateETONumba(1, doy, self.latitude[i_row, i_col], self.elevation[i_row, i_col], 
                                                                    self.minT_daily[i_row, i_col, :], self.maxT_daily[i_row, i_col, :], 
                                                                    self.wind_daily[i_row, i_col, :], self.shortrad_daily_MJm2day[i_row, i_col, :],
                                                                      self.rel_humidity_daily[i_row, i_col, :])
                
                monthly_eto[i_row,i_col,:] = averageDailyToMonthly(self.eto_daily[i_row,i_col,:], self.leap_year)
        
        # counting months with monthly precipitation >= monthly ET0
        self.months_P_gte_eto = np.sum(monthly_precip >= monthly_eto, axis = 2)
    

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
        
        elif lgp in range(121,210+1):
            lgp_agc = lgp # correct
            
        elif lgp > 210:
            lgp_agc = max(210, min(lgp, lgp_equv)) # correct
        
        else:
            raise ValueError('Something is wrong.')
        
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
                    if self.im_mask[i,j] == self.no_mask_value:
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

class ProcessExcelWrapper:

    def __init__(self, filename_input, crop_name, crop_cycle_length, input_management, condition, output_path):
        """
        Initialization of ProcessExcelWrapper object class.

        Args:
            filename_input (str): Input file name directory
            crop_name (str): Selected LUT crop name
            crop_cycle_length (float/int): Reference cycle length of the selected LUT crop name
            input_management (str): [Low, Medium or High]
            condition (str): Rainfed ['rainfed','rain-fed','Rainfed','RainFed', 'RAINFED','r','R', 'rf','RF']
                             Irrigated ['irrigated','Irrigated', 'IRRIGATED', 'I','ir']
            output_path(str): folder directory to save the output excel sheet
        Return:
            None.    
        """
        self.filename_input=filename_input
        self.crop_name=crop_name
        self.crop_cycle_length=crop_cycle_length
        self.input_management=input_management
        self.condition=condition
        self.data_dict={}
        self.SHEET_NAMES=['A5-1.1','A5-1.2','A5-1.3','A5-1.4','A5-1.5']
        try:
            self.output_file_name= output_path+'/'+self.crop_name+'_'+self.input_management+'_'+self.condition+'_lst.xlsx'
        except (TypeError, ValueError):
            print ("Wrong operation filname cannot create!!")
        
    def read_input_file(self, sheet_name):
        """
        (Sub-function) Reading the selected sheet of GAEZ Appendix.
        """
        try:
            self.Dframe=pd.read_excel(self.filename_input, header=0, sheet_name=sheet_name)
        except IOError:
            print("Could not open the file "+self.filename_input)

    def process_data(self):
        """
        (Sub-function) Main function to compile all agro-climatic indicators based on user-defined.
        settings.
        """
        try:
            self.hDframe=self.Dframe.iloc[0:2].iloc[0,1]
            if self.condition in ['rainfed','rain-fed','Rainfed','RainFed', 'RAINFED','r','R', 'rf','RF']:
                condition='rain-fed'
            elif self.condition in ['irrigated','Irrigated', 'IRRIGATED', 'I','ir']:
                condition='irrigated'
            self.bDframe=self.Dframe.iloc[3:]
            self.bDframe.columns=self.Dframe.iloc[2]
            self.cropgroups=self.bDframe.groupby('Common name')
            if self.crop_name in self.cropgroups.groups.keys():
                self.cropDframe=self.cropgroups.get_group(self.crop_name)
            elif self.crop_name+' ' in self.cropgroups.groups.keys():
                self.cropDframe=self.cropgroups.get_group(self.crop_name+' ') 
            if ((self.hDframe.find(condition) > 0) and (self.hDframe.find('temperature < 10')>0 or self.hDframe.find('temperature > 20')>0)):
                for i in range(0,self.cropDframe.shape[0]):
                    if i%12==0:
                        if self.cropDframe.iloc[i,2:5][1]=='+':
                            if np.isnan(self.cropDframe.iloc[i,2:5][0])==False and np.isnan(self.cropDframe.iloc[i,2:5][0])==False:
                                if int(self.cropDframe.iloc[i,2:5][0]+self.cropDframe.iloc[i,2:5][2])==self.crop_cycle_length:
                                    self.InputlDframe=self.cropDframe[i:i+12].groupby('Input level').get_group(self.input_management).iloc[1:,6:].fillna(0)
                                    self.InputlDframe1=self.InputlDframe.drop([0], axis=1)
                                    processed=self.process_colemnnames(list(self.InputlDframe1.columns))
                                    self.OutDFrame=self.InputlDframe1.rename(columns=processed, errors='raise')                 
                            else:
                                if self.cropDframe.iloc[i,2:5][2]==self.crop_cycle_length:
                                    self.InputlDframe=self.cropDframe[i:i+12].groupby('Input level').get_group(self.input_management).iloc[1:,6:].fillna(0)
                                    self.InputlDframe1=self.InputlDframe.drop([0], axis=1)
                                    processed=self.process_colemnnames(list(self.InputlDframe1.columns))
                                    self.OutDFrame=self.InputlDframe1.rename(columns=processed, errors='raise')
                        elif np.isnan(self.cropDframe.iloc[i,2:5][0])==True and np.isnan(self.cropDframe.iloc[i,2:5][1])==True:
                            if int(self.cropDframe.iloc[i,2:5][2])==self.crop_cycle_length:
                                self.InputlDframe=self.cropDframe[i:i+12].groupby('Input level').get_group(self.input_management).iloc[1:,6:].fillna(0)
                                self.InputlDframe1=self.InputlDframe.drop([0], axis=1)
                                processed=self.process_colemnnames(list(self.InputlDframe1.columns))
                                self.OutDFrame=self.InputlDframe1.rename(columns=processed, errors='raise')
                        if self.hDframe.find('temperature < 10')>0:   
                            self.data_dict.update({'mean < 10':self.OutDFrame})
                        elif self.hDframe.find('temperature > 20')>0:
                            self.data_dict.update({'mean > 20':self.OutDFrame})
                        else:
                            print('Error Data Not Found Invalid Data to write Dataframe')
            elif self.hDframe.find('frost')>0:
                    try:
                        for i in range(0,self.cropDframe.shape[0]):
                            if self.cropDframe.iloc[i,2:5][1]=='+':
                                if np.isnan(self.cropDframe.iloc[i,2:5][0])==False and np.isnan(self.cropDframe.iloc[i,2:5][2])==False:
                                    if int(self.cropDframe.iloc[i,2:5][0]+self.cropDframe.iloc[i,2:5][2])==self.crop_cycle_length:
                                        self.InputlDframe=self.cropDframe[i:i+2].iloc[0:1,6:].fillna(0)
                                        self.InputlDframe1=self.InputlDframe.drop([0], axis=1)
                                        processed=self.process_colemnnames(list(self.InputlDframe1.columns))
                                        self.OutDFrame=self.InputlDframe1.rename(columns=processed, errors='raise')
                                        self.OutDFrame['type']='lgpt10'
                                        self.OutDFrame['365,366']=0
                                        self.data_dict.update({'lgpt10':self.OutDFrame})
                                else:
                                    if self.cropDframe.iloc[i,2:5][2]==self.crop_cycle_length:
                                        if np.isnan(self.cropDframe.iloc[i,2:5][0])==True and np.isnan(self.cropDframe.iloc[i,2:5][1])==True:
                                            self.InputlDframe=self.cropDframe[i:i+2].iloc[0:1,6:].fillna(0)
                                            self.InputlDframe1=self.InputlDframe.drop([0], axis=1)
                                            processed=self.process_colemnnames(list(self.InputlDframe1.columns))
                                            self.OutDFrame=self.InputlDframe1.rename(columns=processed, errors='raise')
                                            self.OutDFrame['type']='lgpt10'
                                            self.OutDFrame['365,366']=0
                                            self.data_dict.update({'lgpt10':self.OutDFrame})
                            elif np.isnan(self.cropDframe.iloc[i,2:5][0])==True and np.isnan(self.cropDframe.iloc[i,2:5][1])==True:
                                if int(self.cropDframe.iloc[i,2:5][2])==self.crop_cycle_length:
                                    InputlDframe=self.cropDframe[i:i+1].iloc[0:1,6:].fillna(0)
                                    self.InputlDframe1=InputlDframe.drop([0], axis=1)
                                    processed=self.process_colemnnames(list(self.InputlDframe1.columns))
                                    self.OutDFrame=self.InputlDframe1.rename(columns=processed, errors='raise')
                                    self.OutDFrame['type']='lgpt10'
                                    self.OutDFrame['365,366']=0
                                    self.data_dict.update({'lgpt10':self.OutDFrame})
                    except (KeyError,IndexError, ValueError, AttributeError):
                        print(" Index or Key not found or wrong value in the input PLease check: ", self.crop_name, self.crop_cycle_length,self.input_management)
        except (KeyError, IndexError):
            print ("Index or Key not present in the dataframe")
        
    def process_colemnnames(self, columns:list):
        """
        (Sub-function) Output excel column settings.
        """
        processed={}
        for val in columns:
            x=val
            val=val.strip("'")
            if val=='Constraint type':
                processed.update({x:'type'})
            elif len(val.split('-'))==2 and val.split('-')[1].isnumeric()==True:
                if val.split('-')[0]=='1':
                    val="0-29"
                    processed.update({x:val.replace('-',',')})
                elif val.split('-')[0]=='330' and val.split('-')[1]=='365':
                    val="330-364"
                    processed.update({x:val.replace('-',',')})
                else:
                    processed.update({x:val.replace('-',',')})    
            else:
                processed.update({x:val})
        return processed
    
    def write_excel(self):
        """
        (Sub-function) Convert the database dictionary to excel writer to export.
        """
        
        try:
            with pd.ExcelWriter(self.output_file_name) as writer:
                for k in self.data_dict:
                    self.data_dict[k].to_excel(writer, sheet_name=k, index=False)
        except (IOError):
            print("Not a valid Input/Output File")

    def run(self):
        """
        Run the excel wrapper function to export out user-defined crop/LUT agro-climatic
        constraints:
        
        Args:
            None.
        Return:
            Excel sheet of agro-climatic constraint factors.
        """
        for s in self.SHEET_NAMES:
            self.read_input_file(sheet_name=s)
            self.process_data()
            self.write_excel()
        return self.output_file_name
    
#----------------- End of file -------------------------#
