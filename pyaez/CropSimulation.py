"""
PyAEZ version 2.3 (Apr 2025)
This CropSimulation Class simulates all the possible crop cycles to find 
the best crop cycle that produces maximum yield for a particular grid.

2020: N. Lakmal Deshapriya
2022/2023: Swun Wunna Htet, Kittiphon Boonma
2023 (Dec): Swun Wunna Htet
2025 (Apr): Swun Wunna Htet, Kittiphon Boonma

Modifications
1.  Minimum cycle length checking logic added to crop simulation.
2.  New crop parameters: minimum cycle length, maximum cycle length, plant height is added logic added.
3.  Removing unnecessary variables in the algorithm for slight code enhancement.
4.  Added hibernation crop simulation routine, and vernalization calculation.
5.  Revised crop simulation for perennial crops, ***validation is not completed yet.
6.  The workflow of crop simulation is subsetted with sub-function which works like previous core workflow
    which helps in future modification.
"""
import numpy as np
import numba as nb
import pandas as pd
try:
    import gdal
except:
    from osgeo import gdal

# from pyaez import UtilitiesCalc,BioMassCalc,ETOCalc,CropWatCalc,ThermalScreening, LGPCalc
from pyaez.ThermalScreening import getTempTrend,  getTemperatureGrowingPeriod, getInterpolatedTempData, findbeginningLGPT,  \
    getTemperatureProfile, getTemperatureSum, calculateTemperatureProfileClasses, getReductionFactorNumba
from pyaez.BioMassCalc import calculateBiomassNumba
from pyaez.CropWatCalc import calculateMoistureLimitedYieldNumba
from pyaez.ETOCalc import calculateETONumba
from pyaez.UtilitiesCalc import interpMonthlyToDaily, generateLatitudeMap, averageDailyToMonthly
from pyaez.LGPCalc import RefWaterBalanceCalc, psh, rainPeak, islgpt, val10day, search_cycles

class CropSimulation(object):

    def __init__(self):
        """Initiate a Class instance
        """        
        self.set_mask = False
        self.set_tclimate_screening = False
        self.set_Tsum_screening = False
        self.set_Permafrost_screening = False  
        self.setCropSpecificRule = False
        self.set_monthly = False
        self.set_daily = False
        self.leap_year = False
    
    """--------------------- MANDATORY FUNCTIONS START HERE --------------------------"""

    def setMonthlyClimateAndWaterData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity, Sa = 100., D = 1.):
        """
        (MANDATORY FUNCTION)
        Load MONTHLY climate data into the Class and calculate the Reference Evapotranspiration (ETo).
        All climatic variables used in the simulations will be interpolated into daily values with 365 days.
        Users need to run setLocationTerrainData() before running this function.

        Args:
            min_temp (3D-NumPy Array): Monthly minimum temperature [Unit: Degree Celcius]
            max_temp (3D-NumPy Array): Monthly maximum temperature [Unit: Degree Celcius]
            precipitation (3D-NumPy Array): Monthly total precipitation [Unit: mm/day]
            short_rad (3D-NumPy Array): Monthly solar radiation [Unit: W/m2]
            wind_speed (3D-NumPy Array): Monthly windspeed at 2m altitude [Unit: m/s]
            rel_humidity (3D-NumPy Array): Monthly relative humidity [Unit: percentage decimal (0-1)]
            Sa (int/float/2D-NumPy Array): Soil water holding capacity [Unit: mm]. Default value to 100.
            D (int/float): rooting depth [Unit: m]. Default value to 1.
        Return:
            None.
        """
        doy = None

        self.Sa = Sa
        self.D = D

        if np.all(min_temp.shape[2] ==12 and max_temp.shape[2] ==12 and wind_speed.shape[2] ==12
            and short_rad.shape[2] ==12 and rel_humidity.shape[2] ==12 and precipitation.shape[2] ==12):
            doy = 365
        else:
            raise Exception('The monthly time dimension of climate data is not uniform. Please modify.')
        
        # Empty array creation
        self.meanT_daily = np.zeros((self.im_height, self.im_width, doy))
        self.totalPrec_daily = np.zeros((self.im_height, self.im_width, doy))
        self.pet_daily = np.zeros((self.im_height, self.im_width, doy))
        self.minT_daily = np.zeros((self.im_height, self.im_width, doy))
        self.maxT_daily = np.zeros((self.im_height, self.im_width, doy))
        self.shortRad_daily = np.zeros((self.im_height, self.im_width, doy))
        self.wind2m_daily = np.zeros((self.im_height, self.im_width, doy))
        self.rel_humidity_daily = np.zeros((self.im_height, self.im_width, doy))

        # curtailing extreme value ranges for calculation purposes.
        self.rel_humidity_daily[self.rel_humidity_daily > 0.99] = 0.99
        self.rel_humidity_daily[self.rel_humidity_daily < 0.05] = 0.05
        self.shortRad_daily[self.shortRad_daily < 0] = 0
        self.wind2m_daily[self.wind2m_daily < 0] = 0

        mean_temp = (min_temp + max_temp)/2
        # Interpolate monthly to daily data
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):

                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue

                self.meanT_daily[i_row, i_col, :] = interpMonthlyToDaily(mean_temp[i_row, i_col, :], 1, doy)
                self.minT_daily[i_row, i_col, :] = interpMonthlyToDaily(min_temp[i_row, i_col, :], 1, doy)
                self.maxT_daily[i_row, i_col, :] = interpMonthlyToDaily(max_temp[i_row, i_col, :], 1, doy)
                self.totalPrec_daily[i_row, i_col, :] = interpMonthlyToDaily(precipitation[i_row, i_col, :], 1, doy, no_minus_values=True)
                self.shortRad_daily[i_row, i_col, :] = interpMonthlyToDaily(short_rad[i_row, i_col, :], 1, doy, no_minus_values=True)
                self.wind2m_daily[i_row, i_col, :] = interpMonthlyToDaily(wind_speed[i_row, i_col, :], 1, doy, no_minus_values=True)
                self.rel_humidity_daily[i_row, i_col, :] = interpMonthlyToDaily(rel_humidity[i_row, i_col, :], 1, doy, no_minus_values=True)

                # calculation of reference evapotranspiration (ETo)
                # convert w/m2 to MJ/m2/day
                shortrad_daily_MJm2day = (self.shortRad_daily[i_row, i_col, :] * 3600 * 24)/1000000
            
                self.pet_daily[i_row, i_col, :] = calculateETONumba(1, doy, self.latitude[i_row, i_col], self.elevation[i_row, i_col],  
                                                                    self.minT_daily[i_row, i_col, :], self.maxT_daily[i_row, i_col, :], 
                                                                    self.wind2m_daily[i_row, i_col, :], shortrad_daily_MJm2day,  self.rel_humidity_daily[i_row, i_col, :],
                                                                    self.leap_year)
        

        #============================
        # Calculation of Reference Water Balance to estimate ETa, ETm
        #============================
        # kc_list = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
        #============================
        Txsnm = 0.  # Txsnm - snow melt temperature threshold
        Fsnm = 5.5  # Fsnm - snow melting coefficient
        Sb_old = 0.
        Wb_old = 0.
        #============================
        Tx365 = self.maxT_daily.copy()
        Ta365 = self.meanT_daily.copy()
        Pcp365 = self.totalPrec_daily.copy()
        self.Eto365 = self.pet_daily.copy()  # Eto
        self.Etm365 = np.zeros(Tx365.shape)
        self.Eta365 = np.zeros(Tx365.shape)
        self.Sb365 = np.zeros(Tx365.shape)
        self.Wb365 = np.zeros(Tx365.shape)
        self.Wx365 = np.zeros(Tx365.shape)
        self.kc365 = np.zeros(Tx365.shape)
        self.maxT_daily_new = np.zeros(Tx365.shape)
        #============================
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):

                lgpt5_point = np.sum(self.meanT_daily[i_row, i_col,:]>=5)

                # totalPrec_monthly = averageDailyToMonthly(self.totalPrec_daily[i_row, i_col, :], self.leap_year)
                meanT_daily_point = Ta365[i_row, i_col, :]
                istart0, istart1 = rainPeak( meanT_daily_point, lgpt5_point)
                
                # calculate the increasing/decreasing temperature trend
                istup = getTempTrend(self.meanT_daily[i_row, i_col, :])
                #----------------------------------
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue

                for d in range(0, doy):
                    p = psh(0., self.Eto365[i_row, i_col, d])
                    Eta_new, Etm_new, Wb_new, Wx_new, Sb_new, kc_new = RefWaterBalanceCalc(
                        np.float64(Tx365[i_row, i_col, d]), np.float64(
                            Ta365[i_row, i_col, d]),
                            # Ta365[i_row, i_col, doy]),
                        np.float64(Pcp365[i_row, i_col, d]), Txsnm, Fsnm, np.float64(
                            self.Eto365[i_row, i_col, d]),
                        Wb_old, Sb_old, d, istart0, istart1,
                        self.Sa, self.D, p, lgpt5_point, istup[d])

                    if Eta_new <0.: Eta_new = 0.

                    self.Eta365[i_row, i_col, d] = Eta_new
                    self.Etm365[i_row, i_col, d] = Etm_new
                    self.Wb365[i_row, i_col, d] = Wb_new
                    self.Wx365[i_row, i_col, d] = Wx_new
                    self.Sb365[i_row, i_col, d] = Sb_new
                    self.kc365[i_row, i_col, d] = kc_new

                    Wb_old = Wb_new
                    Sb_old = Sb_new

        self.set_monthly=True
    
    def setDailyClimateAndWaterData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity, Sa = 100., D = 1.):
        """
        (MANDATORY FUNCTION)
        Load DAILY climate data into the Class and calculate the Reference Evapotranspiration (ETo).
        Users need to run setLocationTerrainData() before running this function.

        Args:
            min_temp (3D-NumPy Array): Monthly minimum temperature [Unit: Degree Celcius]
            max_temp (3D-NumPy Array): Monthly maximum temperature [Unit: Degree Celcius]
            precipitation (3D-NumPy Array): Monthly total precipitation [Unit: mm/day]
            short_rad (3D-NumPy Array): Monthly solar radiation [Unit: W/m2]
            wind_speed (3D-NumPy Array): Monthly windspeed at 2m altitude [Unit: m/s]
            rel_humidity (3D-NumPy Array): Monthly relative humidity [Unit: percentage decimal (0-1)]
            Sa (int/float/2D-NumPy Array): Soil water holding capacity [Unit: mm]. Default value to 100.
            D (int/float): rooting depth [Unit: m]. Default value to 1.
        Return:
            None.
        """
        doy = None

        self.Sa = Sa
        self.D = D

        if np.all(min_temp.shape[2] ==365 and max_temp.shape[2] ==365 and wind_speed.shape[2] ==365
                    and short_rad.shape[2] ==365 and rel_humidity.shape[2] ==365 and precipitation.shape[2] ==365):
            pass
        elif np.all(min_temp.shape[2] ==366 and max_temp.shape[2] ==366 and wind_speed.shape[2] ==366
                    and short_rad.shape[2] ==366 and rel_humidity.shape[2] ==366 and precipitation.shape[2] ==366):
            self.leap_year = True
        else:
            raise Exception('The daily time dimension of climate data is not uniform. Please modify.')

        doy = 366 if self.leap_year else 365

        # setting the daily temperature
        self.minT_daily = min_temp.copy()
        self.maxT_daily = max_temp.copy()
        self.meanT_daily = (self.minT_daily + self.maxT_daily)/2
        self.totalPrec_daily = precipitation.copy()
        self.shortRad_daily = short_rad.copy()
        self.wind2m_daily = wind_speed.copy()
        self.rel_humidity_daily = rel_humidity.copy()

        # curtailing extreme value ranges for calculation purposes.
        self.rel_humidity_daily[self.rel_humidity_daily > 0.99] = 0.99
        self.rel_humidity_daily[self.rel_humidity_daily < 0.05] = 0.05
        self.shortRad_daily[self.shortRad_daily < 0] = 0
        self.wind2m_daily[self.wind2m_daily < 0] = 0

        self.pet_daily = np.zeros((self.im_height, self.im_width, doy))

        for i_row in range(self.im_height):
            for i_col in range(self.im_width):

                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                # calculation of reference evapotranspiration (ETo)
                shortrad_daily_MJm2day = (self.shortRad_daily[i_row, i_col,:] * 3600 * 24)/1000000
                self.pet_daily[i_row, i_col, :] = calculateETONumba(1, doy, self.latitude[i_row, i_col], self.elevation[i_row, i_col],  
                                                    self.minT_daily[i_row, i_col, :], self.maxT_daily[i_row, i_col, :], 
                                                    self.wind2m_daily[i_row, i_col, :], shortrad_daily_MJm2day,  self.rel_humidity_daily[i_row, i_col, :],
                                                    self.leap_year)
        
        #============================
        # Calculation of Reference Water Balance to estimate ETa, ETm
        #============================
        # kc_list = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
        #============================
        Txsnm = 0.  # Txsnm - snow melt temperature threshold
        Fsnm = 5.5  # Fsnm - snow melting coefficient
        Sb_old = 0.
        Wb_old = 0.
        #============================
        Tx365 = self.maxT_daily.copy()
        Ta365 = self.meanT_daily.copy()
        Pcp365 = self.totalPrec_daily.copy()
        self.Eto365 = self.pet_daily.copy()  # Eto
        self.Etm365 = np.zeros(Tx365.shape)
        self.Eta365 = np.zeros(Tx365.shape)
        self.Sb365 = np.zeros(Tx365.shape)
        self.Wb365 = np.zeros(Tx365.shape)
        self.Wx365 = np.zeros(Tx365.shape)
        self.kc365 = np.zeros(Tx365.shape)
        self.maxT_daily_new = np.zeros(Tx365.shape)
        #============================
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):

                lgpt5_point = np.sum(self.meanT_daily[i_row, i_col,:]>=5)

                # totalPrec_monthly = averageDailyToMonthly(self.totalPrec_daily[i_row, i_col, :], self.leap_year)
                meanT_daily_point = Ta365[i_row, i_col, :]
                istart0, istart1 = rainPeak(meanT_daily_point, lgpt5_point)
                
                istup = getTempTrend(self.meanT_daily[i_row, i_col, :])
                #----------------------------------
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue

                for d in range(0, doy):
                    p = psh(0., self.Eto365[i_row, i_col, d])
                    Eta_new, Etm_new, Wb_new, Wx_new, Sb_new, kc_new = RefWaterBalanceCalc(
                        np.float64(Tx365[i_row, i_col, d]), np.float64(
                            Ta365[i_row, i_col, d]),
                            # Ta365[i_row, i_col, doy]),
                        np.float64(Pcp365[i_row, i_col, d]), Txsnm, Fsnm, np.float64(
                            self.Eto365[i_row, i_col, d]),
                        Wb_old, Sb_old, d, istart0, istart1,
                        self.Sa, self.D, p, lgpt5_point, istup[d])

                    if Eta_new <0.: Eta_new = 0.

                    self.Eta365[i_row, i_col, d] = Eta_new
                    self.Etm365[i_row, i_col, d] = Etm_new
                    self.Wb365[i_row, i_col, d] = Wb_new
                    self.Wx365[i_row, i_col, d] = Wx_new
                    self.Sb365[i_row, i_col, d] = Sb_new
                    self.kc365[i_row, i_col, d] = kc_new

                    Wb_old = Wb_new
                    Sb_old = Sb_new
        self.set_daily = True
    
    def setLocationTerrainData(self, lat_min, lat_max, elevation):
        """
        (MANDAOTRY FUNCTION)
        Load geographical extents and elevation data in to the Class, 
        and create a latitude map.

        Args:
            lat_min (float): the minimum latitude of the AOI [Unit: Decimal Degrees]
            lat_max (float): the maximum latitude of the AOI [Unit: Decimal Degrees]
            elevation (2D-NumPy Array): elevation map [Unit: meters]
        
        Return:
            None.
        """
        self.elevation = elevation
        self.im_height = elevation.shape[0]
        self.im_width = elevation.shape[1]
        self.latitude = generateLatitudeMap(lat_min, lat_max, self.im_height, self.im_width)
    
    def readCropandCropCycleParameters(self, file_path, crop_name):
        """
        (MANDATORY FUNCTION)
        Importing the excel sheet of crop-specific parameters,
        crop water requirements, management info, perennial adjustment parameters,
        and TSUM screening thresholds.

        Args:
            file_path (String): The file path of the external excel sheet in xlsx format.
            crop_name (String): Unique name of crop for crop simulation.

        Return:
            None.
        """

        self.crop_name = crop_name
        df = pd.read_excel(file_path)

        crop_df_index = df.index[df['Crop_name'] == crop_name].tolist()[0]
        crop_df = df.loc[df['Crop_name'] == crop_name]

        self.setCropParameters(LAI=crop_df['LAI'][crop_df_index], HI=crop_df['HI'][crop_df_index], legume=crop_df['legume'][crop_df_index], adaptability=int(crop_df['adaptability'][crop_df_index]), cycle_len=int(crop_df['cycle_len'][crop_df_index]), D1=crop_df['D1']
                               [crop_df_index], D2=crop_df['D2'][crop_df_index], min_temp=crop_df['min_temp'][crop_df_index], aLAI=crop_df['aLAI'][crop_df_index], bLAI=crop_df['bLAI'][crop_df_index], aHI=crop_df['aHI'][crop_df_index], bHI=crop_df['bHI'][crop_df_index],
                               min_cycle_len=crop_df['min_cycle_len'][crop_df_index], max_cycle_len=crop_df['max_cycle_len'][crop_df_index], plant_height = crop_df['height'][crop_df_index], SDG = crop_df['SDG'][crop_df_index])
        self.setCropCycleParameters(stage_per=[crop_df['stage_per_1'][crop_df_index], crop_df['stage_per_2'][crop_df_index], crop_df['stage_per_3'][crop_df_index], crop_df['stage_per_4'][crop_df_index]], kc=[crop_df['kc_0'][crop_df_index], crop_df['kc_1'][crop_df_index], crop_df['kc_2']
                                    [crop_df_index]], kc_all=crop_df['kc_all'][crop_df_index], yloss_f=[crop_df['yloss_f0'][crop_df_index], crop_df['yloss_f1'][crop_df_index], crop_df['yloss_f2'][crop_df_index], crop_df['yloss_f3'][crop_df_index]], yloss_f_all=crop_df['yloss_f_all'][crop_df_index])
        # perennial = 1, annual = 0
        if crop_df['annual/perennial flag'][crop_df_index] == 1:
            self.perennial = True
        else:
            self.perennial = False

        # hibernation = 1, non-hibernation = 0
        if crop_df['HB_flag'][crop_df_index] == 1:
            self.hibernate = True
        else:
            self.hibernate = False

        # If users provide all TSUM thresholds, TSUM screening will be done. Otherwise, TSUM screening will not be activated.
        if np.all([crop_df['LnS'][crop_df_index] != np.nan, crop_df['LsO'][crop_df_index] != np.nan, crop_df['LO'][crop_df_index] != np.nan, crop_df['HnS'][crop_df_index] != np.nan, crop_df['HsO'][crop_df_index] != np.nan, crop_df['HO'][crop_df_index] != np.nan]):
            self.setTSumScreening(LnS=crop_df['LnS'][crop_df_index], LsO=crop_df['LsO'][crop_df_index], LO=crop_df['LO'][crop_df_index],
                                  HnS=crop_df['HnS'][crop_df_index], HsO=crop_df['HsO'][crop_df_index], HO=crop_df['HO'][crop_df_index])

        # releasing memory
        del (crop_df_index, crop_df)



    def setCropParameters(self, LAI, HI, legume, adaptability, cycle_len, D1, D2, min_temp, aLAI, bLAI, aHI, bHI, min_cycle_len, max_cycle_len, plant_height, SDG):
        """
        (NESTED FUNCTION)
        This function allows users to set up the main crop parameters necessary for PyAEZ.

        Parameters
        ----------
        LAI (float): Leaf Area Index
        HI (float): Harvest Index
        legume (binary, yes=1, no=0): Is the crop legume?
        adaptability (int): Crop adaptability clases (1-4)
        cycle_len (int): Length of crop cycle
        D1 (float): Rooting depth at the beginning of the crop cycle [m]
        D2 (float): Rooting depth after crop maturity [m]
        min_temp (int or float): minimum temperature requirement of the crop [deg C]
        aLAI (int or float): alpha LAI adjustment parameter
        bLAI (int or float): beta LAI adjustment parameter
        min_cycle_len (int): minimum cycle length [days]
        max_cycle_len (int): maximum cycle length [days]
        plant_height (int or float): plant height [m]
        SDG (int): soil water depletion factor group
        
        Returns
        -------
        None.
        """
        self.LAi = LAI  # leaf area index
        self.HI = HI  # harvest index
        self.legume = legume  # binary value
        self.adaptability = adaptability  # one of [1,2,3,4] classes
        self.cycle_len = cycle_len  # length of growing period
        self.D1 = D1  # rooting depth 1 (m)
        self.D2 = D2  # rooting depth 2 (m)
        self.min_temp = min_temp  # minimum temperature
        self.aLAI = aLAI
        self.bLAI = bLAI
        self.aHI = aHI
        self.bHI = bHI
        self.min_cycle_len = min_cycle_len
        self.max_cycle_len = max_cycle_len
        self.plant_height= plant_height
        self.crop_group = SDG

    def setCropCycleParameters(self, stage_per, kc, kc_all, yloss_f, yloss_f_all):
        """
        (NESTED FUNCTION)
        This function allows users to set up crop-specific growth sta.

        Parameters
        ----------
        stage_per (list of integers): percentages for D1, D2, D3, D4 growth stages
        kc (float): crop water requirements for initial, reproductive and mature stages of crop development
        kc_all (float): crop water requirements for the entire growth cycle
        yloss_f (list of floats): yield loss factor for D1, D2, D3, D4 growth stages
        yloss_f_all (float): yield loss factor for the entire growth period

        Returns
        -------
        None.
        """
        self.d_per = np.array(stage_per)  # Percentage for D1, D2, D3, D4 stages
        self.kc = np.array(kc)  # 3 crop water requirements for initial, reproductive, the end of the maturation stages
        self.kc_all = kc_all  # crop water requirements for entire growth cycle
        self.yloss_f = np.array(yloss_f)  # yield loss for D1, D2, D3, D4
        self.yloss_f_all = yloss_f_all  # yield loss for entire growth cycle
    
    
    
    def ImportLGPandLGPT(self, lgp, lgpt5, lgpt10):
        """
        (MANDATORY FUNCTION)
        Importing LGP and temperature growing period data.

        Args:
            lgp (2D NumPy Array): Length of Growing Period [Unit: Days].
            lgpt5 (2D NumPy Array): Temperature Growing Period at 5℃ threshold [Unit: Days]. 
            lgpt10 (2D NumPy Array): Temperature Growing Period at 10℃ threshold [Unit: Days].

        Return:
            None.
        """
        self.LGP = lgp
        self.LGPT5 = lgpt5
        self.LGPT10 = lgpt10
    
    """----------------------------  MANDATORY FUNCTIONS END HERE   --------------------------"""
    """---------------------THERMAL SCREENING FUNCTIONS STARTS HERE (OPTIONAL)--------------------------"""

    def setThermalClimateScreening(self, t_climate, no_t_climate):
        """
        (OPTIONAL FUNCTION) The thermal screening function omit out user-specified thermal climate classes
        not suitable for a particular crop for crop simulation. Using this optional 
        function will activate application of thermal climate screening in crop cycle simulation.
    

        Args:
            t_climate (2-D NumPy Array): Thermal Climate.
            no_t_climate (list): A list of thermal climate classes not suitable for crop simulation.

        Return:
            None.
        """
        self.t_climate = t_climate
        self.no_t_climate = no_t_climate  # list of unsuitable thermal climate
        self.set_tclimate_screening = True

    def setTSumScreening(self, LnS, LsO, LO, HnS, HsO, HO):
        """
        This thermal screening corresponds to Type A constraint (TSUM Screeing) of GAEZ which
        uses six TSUM thresholds for optimal, sub-optimal and not suitable conditions. Using 
        this optional function will activate application of TSUM screening in crop cycle simulation.
        

        Args:
            LnS (int): Lower boundary of not-suitable accumulated heat unit range.
            LsO (int): Lower boundary of sub-optimal accumulated heat unit range.
            LO (int): Lower boundary of optimal accumulated heat unit range.
            HnS (int):Upper boundary of not-suitable accumulated heat range.
            HsO (int): Upper boundary of sub-optimal accumulated heat range.
            HO (int): Upper boundary of not-suitable accumulated heat range.

        Return:
            None.
        """
        self.LnS = int(LnS)  # Lower boundary/ not suitable
        self.LsO = int(LsO)  # Lower boundary/ sub optimal
        self.LO = int(LO)  # Lower boundary / optimal
        self.HnS = int(HnS)  # Upper boundary/ not suitable
        self.HsO = int(HsO)  # Upper boundary / sub-optimal
        self.HO = int(HO)  # Upper boundary / optimal
        self.set_Tsum_screening = True

    def setPermafrostScreening(self, permafrost_class):
        """
        (OPTIONAL FUNCTION) This thermal screening corresponds to permafrost characteristics screening.  
        Using this optional function will activate  permafrost screening in crop cycle simulation.
        
        Args:
            permaforst_class (2-D NumPy Array): Permafrost class (Obtained from Module I: Climate Regime).
        Return:
            None.
        """
        self.permafrost_class = permafrost_class  # permafrost class 2D numpy array
        self.set_Permafrost_screening = True

    def setupCropSpecificRule(self, file_path, crop_name):
        """
        (OPTIONAL FUNCTION) Initiates the Crop Specific Rule (Temperature Profile 
        Constraint) on the existing crop based on user-specified constraint rules.

        Parameters
        ----------
        file_path (String): The file path of excel sheet where the Type B constraint rules are provided as xlsx.format.
        crop_name (String): Unique name of crop to consider. The name must be the corresponding to the Crop_name of crop
                            parameter sheet.

        Returns
        -------
        None.

        """

        data = pd.read_excel(file_path)
        self.crop_name = crop_name

        self.data = data.loc[data['Crop'] == self.crop_name]

        self.setCropSpecificRule = True

        # releasing data
        del (data)
    
    """---------------------THERMAL SCREENING FUNCTIONS END HERE (OPTIONAL)--------------------------"""
    """---------------------------- OPTIONAL FUNCTIONS STARTS HERE  ---------------------------------"""  

    def setStudyAreaMask(self, admin_mask, no_data_value):
        """(OPTIONAL FUNCTION) Set clipping mask of the area of interest.

        Args:
            admin_mask (2D-NumPy Array): mask to extract only region of interest.
            no_data_value (int): pixels with this value will be omitted during PyAEZ calculations.
        Return:
            None.
        """
        self.im_mask = admin_mask
        self.nodata_val = no_data_value
        self.set_mask = True
    
    """---------------------------- OPTIONAL FUNCTIONS END HERE  ---------------------------------"""
    """------------------   MAIN FUNCTION OF CROP SIMULATION STARTS HERE  ------------------------"""
    def getEstimatedYieldRainfed(self):
        """Estimation of Maximum Yield for Rainfed scenario.

        Args:
            None.
        Return:
            yld_rain (2D-NumPy Array): the maximum attainable yield under the provided climate conditions, 
                                        under rain-fed conditions [Unit: kg/ha].
        """        
        return self.final_yield_rain

    def getEstimatedYieldIrrigated(self):
        """Estimation of Maximum Yield for Irrigated scenario.

        Args:
            None.
        Return:
            yld_irr (2D-NumPy Array): the maximum attainable yield under the provided climate conditions, 
                                      under irrigated conditions [Unit: kg/ha].
        """
        return self.final_yield_irrig

    def getOptimumCycleStartDateIrrigated(self):
        """
        Function for optimum starting date for irrigated condition.

        Args:
            None.
        Return:
            ccb_irr (2D-NumPy Array): Optimum starting date for irrigated condition. [Unit: DOY].
        """
        return self.crop_calender_irr

    def getOptimumCycleStartDateRainfed(self):
        """
        Function for optimum starting date for rainfed condition.

        Args:
            None.
        Return:
            ccb_rain (2D-NumPy Array): Optimum starting date for rainffed condition. [Unit: DOY].
        """
        return self.crop_calender_rain

    def getThermalReductionFactorRainfed(self):
        """
        Function for thermal reduction factor (fc1) map for rainfed conditions.

        Args:
            None.
        Return:
            fc1_rain (2D_NumPy Array): Thermal reduction factor map (fc1) for rainfed conditions.
        """
        return self.fc1_rain
    
    def getThermalReductionFactorIrrigated(self):
        """
        Function for thermal reduction factor (fc1) map for irrigated conditions.

        Args:
            None.
        Return:
            fc1_irr (2D_NumPy Array): crop specific thermal reduction factor map (fc1) for irrigated conditions.
        """
        return self.fc1_irr

    def getMoistureReductionFactorRainfed(self):
        """
        Function for reduction factor map due to moisture deficit (fc2) for 
        rainfed condition.
        
        Args:
            None.
        Return:
            fc2_rain (2D_NumPy Array):crop-specifc moisture reduction factor map (fc2) for rainfed conditions.
        """
        return self.fc2_rain
    
    def getMoistureReductionFactorIrrigated(self):
        """
        Function for reduction factor map due to moisture deficit (fc2) for 
        irrigated condition.
        
        Args:
            None.
        Return:
            fc2_irr (2D-NumPy Array): crop-specifc moisture reduction factor map (fc2) for irrigated conditions.
        """
        return self.fc2_irr
    
    def getETAIrrigated(self):
        """
        Function for total actual crop evapotranspiration from precipitation (excluding irrigation)
        for irrigated conditon simulation.
        
        Args:
            None.
        Return:
            eta_irr (2D-NumPy Array): crop-specifc total irrigation requirement for irrigated conditions [Unit: mm]
        """
        return self.eta_irr
    
    def getETARainfed(self):
        """
        Function for total actual crop evapotranspiration from precipitation (excluding irrigation)
        for Rainfed conditon simulation.
        
        Args:
            None.
        Return:
            eta_rain (2D-NumPy Array): crop-specifc total irrigation requirement for rainfed conditions [Unit: mm]
        """
        return self.eta_rain
    
    def getWDEIrrigated(self):
        """
        Function for crop-specific total water deficit/net irrigation requirement during crop cycle
        for irrigated conditions.
        
        Args:
            None.
        Return:
            wde_irr (2D-NumPy Array): crop-specifc total water deficit for irrigated conditon [Unit: mm].
        """
        return self.wde_irr
    
    def getWDERainfed(self):
        """
        Function for crop-specific total water deficit/net irrigation requirement during crop cycle
        for irrigated conditions.
        
        Args:
            None.
        Return:
            wde_rain (2D-NumPy Array): crop-specifc total water deficit for rainfed condition [Unit: mm].
        """
        return self.wde_rain
    
#"""------------------   MANDATORY/OPTIONAL FUNCTIONS OF CROP SIMULATION ENDS HERE  ------------------------"""
#"""------------------       MAJOR CROP SIMULATION ROUTINE STARTS HERE    ----------------------------------"""
    def simulateIrrigatedCropCycle(self, start_doy:int =1, end_doy:int= 365, step_doy:int = 1, leap_year:bool = False):
        """Running the Irrigated crop cycle calculation/simulation.

        Args:
            start_doy (int, optional): Starting Julian day for simulating period. Defaults to 1.
            end_doy (int, optional): Ending Julian day for simulating period. Defaults to 365.
            step_doy (int, optional): Spacing (in days) between 2 adjacent crop simulations. Defaults to 1.
            leap_year (bool, optional): whether or not the simulating year is a leap year. Defaults to False.
        Return:
            None.
        """
        bar = '-' * 25
        msg = {True:'Activated', False:'Deactivated'}
        msg2 = {0: 'Annuals', 1:'Perennial'}
        print(f'EXECUTING {self.crop_name} Irrigated Crop Simulation\n{bar}', end = '\n')
        print(f'Crop Type {msg2[self.perennial]}\n{bar}', end = '\n')
        print(f'Masking\t\t\t\t={msg[self.set_mask]}\nThermal Climate Screening\t={msg[self.set_tclimate_screening]}', end= '\n')
        print(f'TSUM Screening\t\t\t={msg[self.set_Tsum_screening]}\nPermafrost Screening\t\t={msg[self.set_Permafrost_screening]}', end= '\n')
        print(f'Crop-specific Rule Screening\t={msg[self.setCropSpecificRule]}\n{bar}', end= '\n')
        print(f'Hibernation Principle\t\t={msg[self.hibernate]}\n{bar}', end= '\n')

        # just a counter to keep track of progress
        count_pixel_completed = 0
        total = self.im_height * self.im_width


        # this stores final result
        self.final_yield_irrig = np.zeros((self.im_height, self.im_width))
        self.crop_calender_irr = np.zeros((self.im_height, self.im_width), dtype=int)
        self.fc2_irr = np.zeros((self.im_height, self.im_width))
        self.fc1_irr = np.zeros((self.im_height, self.im_width))
        self.wde_irr = np.zeros((self.im_height, self.im_width))
        self.eta_irr =  np.zeros((self.im_height, self.im_width))


        for i in range(self.im_height):
            for j in range(self.im_width):
                
                # init_suit_chk_data = getInitialSuitabilityCheckData(self.set_mask, self.im_mask[i,j], self.nodata_val, self.set_Permafrost_screening, self.permafrost_class[i,j], 
                #                                                     self.set_tclimate_screening, self.t_climate[i,j], self.no_t_climate)
                
                # An initial suitability check
                if InitialSuitabilityCheck(self.set_mask, self.im_mask[i,j], self.nodata_val, 
                                self.set_Permafrost_screening, self.permafrost_class[i,j], self.set_tclimate_screening,
                                self.t_climate[i,j], self.no_t_climate):

                    count_pixel_completed = count_pixel_completed + 1
                    print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')
                    continue

                if CycleLengthChecking(self.LGPT5[i,j], self.LGPT10[i,j], self.LGP[i,j], self.min_cycle_len, 'I', self.perennial, self.min_temp):
                    count_pixel_completed = count_pixel_completed + 1
                    print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')
                    continue

                climate_data = DuplicateOneYearClimateData(self.minT_daily[i,j,:], self.maxT_daily[i,j,:], self.meanT_daily[i,j,:], 
                                                           self.shortRad_daily[i,j,:], self.wind2m_daily[i,j,:], self.totalPrec_daily[i,j,:], 
                                                           self.rel_humidity_daily[i,j,:], self.pet_daily[i,j,:])
                
                cycle_len_check_data = getCycleLengthCheckingData(self.LGPT5[i,j], self.LGPT10[i,j], self.LGP[i,j],
                                                                self.min_cycle_len, 'I', self.perennial, self.min_temp, 
                                                                self.max_cycle_len, self.cycle_len, self.hibernate)
                
                LAI_HI_data = getLAIandHIdata(self.LAi, self.HI, self.aLAI, self.bLAI, self.aHI, self.bHI)

                if len(np.array(self.Sa).shape) == 2:
                    Sa_temp = self.Sa[i, j]
                else:
                    Sa_temp = self.Sa

                values = simulateCropCycleOneLocation(start_doy, end_doy, step_doy, leap_year, cycle_len_check_data, LAI_HI_data, climate_data,
                                    self.latitude[i,j], self.elevation[i,j], self.plant_height, self.set_Tsum_screening, self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO,
                                    self.setCropSpecificRule, self.data, self.legume, self.adaptability,
                                    self.kc, self.d_per, Sa_temp, self.D1, self.D2, self.crop_group, self.yloss_f_all, self.yloss_f, 'I', self.crop_name)
                
                self.final_yield_irrig[i,j] = values[0]
                self.wde_irr[i,j] = values[1]
                self.eta_irr[i,j]= values[2]
                self.fc1_irr[i,j] = values[3]
                self.fc2_irr[i,j] = values[4]
                self.crop_calender_irr[i,j] = values[5]

                count_pixel_completed = count_pixel_completed + 1
                print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')

        
        print('\nIrrigated Crop Simulation Completed')

    def simulateRainfedCropCycle(self, start_doy:int =1, end_doy:int= 365, step_doy:int = 1, leap_year:bool = False):
        """Running the Rainfed crop cycle calculation/simulation.

        Args:
            start_doy (int, optional): Starting Julian day for simulating period. Defaults to 1.
            end_doy (int, optional): Ending Julian day for simulating period. Defaults to 365.
            step_doy (int, optional): Spacing (in days) between 2 adjacent crop simulations. Defaults to 1.
            leap_year (bool, optional): whether or not the simulating year is a leap year. Defaults to False.
        Return:
            None.
        """
        bar = '-' * 25
        msg = {True:'Activated', False:'Deactivated'}
        msg2 = {0: 'Annuals', 1:'Perennial'}
        print(f'EXECUTING {self.crop_name} Rainfed Crop Simulation\n{bar}', end = '\n')
        print(f'Crop Type {msg2[self.perennial]}\n{bar}', end = '\n')
        print(f'Masking\t\t\t\t={msg[self.set_mask]}\nThermal Climate Screening\t={msg[self.set_tclimate_screening]}', end= '\n')
        print(f'TSUM Screening\t\t\t={msg[self.set_Tsum_screening]}\nPermafrost Screening\t\t={msg[self.set_Permafrost_screening]}', end= '\n')
        print(f'Crop-specific Rule Screening\t={msg[self.setCropSpecificRule]}\n{bar}', end= '\n')
        print(f'Hibernation Principle\t\t={msg[self.hibernate]}\n{bar}', end= '\n')
        
        # just a counter to keep track of progress
        count_pixel_completed = 0
        total = self.im_height * self.im_width

        # this stores final result
        self.final_yield_rain = np.zeros((self.im_height, self.im_width))
        self.crop_calender_rain = np.zeros((self.im_height, self.im_width), dtype=int)
        self.fc2_rain = np.zeros((self.im_height, self.im_width))
        self.fc1_rain = np.zeros((self.im_height, self.im_width))
        self.wde_rain = np.zeros((self.im_height, self.im_width))
        self.eta_rain =  np.zeros((self.im_height, self.im_width))


        for i in range(self.im_height):
            for j in range(self.im_width):
                
                
                # init_suit_chk_data = getInitialSuitabilityCheckData(self.set_mask, self.im_mask[i,j], self.nodata_val, self.set_Permafrost_screening, self.permafrost_class[i,j], 
                #                                                     self.set_tclimate_screening, self.t_climate[i,j], self.no_t_climate)
                
                if InitialSuitabilityCheck(self.set_mask, self.im_mask[i,j], self.nodata_val, 
                                self.set_Permafrost_screening, self.permafrost_class[i,j], self.set_tclimate_screening,
                                self.t_climate[i,j], self.no_t_climate):

                    count_pixel_completed = count_pixel_completed + 1
                    print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')
                    continue

                if CycleLengthChecking(self.LGPT5[i,j], self.LGPT10[i,j], self.LGP[i,j], self.min_cycle_len, 'R', self.perennial, self.min_temp):
                    count_pixel_completed = count_pixel_completed + 1
                    print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')
                    continue

                climate_data = DuplicateOneYearClimateData(self.minT_daily[i,j,:], self.maxT_daily[i,j,:], self.meanT_daily[i,j,:], 
                                                            self.shortRad_daily[i,j,:], self.wind2m_daily[i,j,:], self.totalPrec_daily[i,j,:], 
                                                            self.rel_humidity_daily[i,j,:], self.pet_daily[i,j,:])
                
                cycle_len_check_data = getCycleLengthCheckingData(self.LGPT5[i,j], self.LGPT10[i,j], self.LGP[i,j],
                                                                self.min_cycle_len, 'R', self.perennial, self.min_temp, 
                                                                self.max_cycle_len, self.cycle_len, self.hibernate)
                
                LAI_HI_data = getLAIandHIdata(self.LAi, self.HI, self.aLAI, self.bLAI, self.aHI, self.bHI)

                if len(np.array(self.Sa).shape) == 2:
                    Sa_temp = self.Sa[i, j]
                else:
                    Sa_temp = self.Sa

                values = simulateCropCycleOneLocation(start_doy, end_doy, step_doy, leap_year, cycle_len_check_data, LAI_HI_data, climate_data,
                                    self.latitude[i,j], self.elevation[i,j], self.plant_height, self.set_Tsum_screening, self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO,
                                self.setCropSpecificRule, self.data, self.legume, self.adaptability,
                                self.kc, self.d_per, Sa_temp, self.D1, self.D2, self.crop_group, self.yloss_f_all, self.yloss_f, 'R', self.crop_name)
                
                self.final_yield_rain[i,j] = values[0]
                self.wde_rain[i,j] = values[1]
                self.eta_rain[i,j]= values[2]
                self.fc1_rain[i,j] = values[3]
                self.fc2_rain[i,j] = values[4]
                self.crop_calender_rain[i,j] = values[5]

                count_pixel_completed = count_pixel_completed + 1
                print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')

        
        print('\nRainfed Crop Simulation Completed')
    
    def simulateIrrigatedSugarcane(self):
        """Running the Irrigated Sugarcane calculation/simulation.

        Args:
            None.
        Return:
            None.
        """
        bar = '-' * 25
        msg = {True:'Activated', False:'Deactivated'}
        print(f'EXECUTING {self.crop_name} Irrigated Crop Simulation\n{bar}', end = '\n')
        print(f'Masking\t\t\t\t={msg[self.set_mask]}\nThermal Climate Screening\t={msg[self.set_tclimate_screening]}', end= '\n')
        print(f'TSUM Screening\t\t\t={msg[self.set_Tsum_screening]}\nPermafrost Screening\t\t={msg[self.set_Permafrost_screening]}', end= '\n')
        print(f'Crop-specific Rule Screening\t={msg[self.setCropSpecificRule]}\n{bar}', end= '\n')
        
        # just a counter to keep track of progress
        count_pixel_completed = 0
        total = self.im_height * self.im_width

        # this stores final result
        self.final_yield_irrig = np.zeros((self.im_height, self.im_width))
        self.crop_calender_irr = np.zeros((self.im_height, self.im_width), dtype=int)
        self.fc2_irr = np.zeros((self.im_height, self.im_width))
        self.fc1_irr = np.zeros((self.im_height, self.im_width))
        self.wde_irr = np.zeros((self.im_height, self.im_width))
        self.eta_irr =  np.zeros((self.im_height, self.im_width))


        for i in range(self.im_height):
            for j in range(self.im_width):
                
                
                # init_suit_chk_data = getInitialSuitabilityCheckData(self.set_mask, self.im_mask[i,j], self.nodata_val, self.set_Permafrost_screening, self.permafrost_class[i,j], 
                #                                                     self.set_tclimate_screening, self.t_climate[i,j], self.no_t_climate)
                
                if InitialSuitabilityCheck(self.set_mask, self.im_mask[i,j], self.nodata_val, 
                                self.set_Permafrost_screening, self.permafrost_class[i,j], self.set_tclimate_screening,
                                self.t_climate[i,j], self.no_t_climate):

                    count_pixel_completed = count_pixel_completed + 1
                    print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')
                    continue

                if CycleLengthChecking(self.LGPT5[i,j], self.LGPT10[i,j], self.LGP[i,j], self.min_cycle_len, 'R', self.perennial, self.min_temp):
                    count_pixel_completed = count_pixel_completed + 1
                    print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')
                    continue

                self.final_yield_irrig[i,j], self.fc1_irr[i,j], self.fc2_irr[i,j], self.eta_irr[i,j], self.wde_irr[i,j], self.crop_calender_irr[i,j] = self.simulateIrrigatedSugarcaneCropCycleOneTest(i, j)

                count_pixel_completed = count_pixel_completed + 1
                print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')

        
        print('\nIrrigated Sugarcane Simulation Completed')
    
    def simulateRainfedSugarcane(self):
        """Running the Rainfed Sugarcane calculation/simulation.

        Args:
            None.
        Return:
            None.
        """
        bar = '-' * 25
        msg = {True:'Activated', False:'Deactivated'}
        print(f'EXECUTING {self.crop_name} Rainfed Crop Simulation\n{bar}', end = '\n')
        print(f'Masking\t\t\t\t={msg[self.set_mask]}\nThermal Climate Screening\t={msg[self.set_tclimate_screening]}', end= '\n')
        print(f'TSUM Screening\t\t\t={msg[self.set_Tsum_screening]}\nPermafrost Screening\t\t={msg[self.set_Permafrost_screening]}', end= '\n')
        print(f'Crop-specific Rule Screening\t={msg[self.setCropSpecificRule]}\n{bar}', end= '\n')
        
        # just a counter to keep track of progress
        count_pixel_completed = 0
        total = self.im_height * self.im_width

        # this stores final result
        self.final_yield_rain = np.zeros((self.im_height, self.im_width))
        self.crop_calender_rain = np.zeros((self.im_height, self.im_width), dtype=int)
        self.fc2_rain = np.zeros((self.im_height, self.im_width))
        self.fc1_rain = np.zeros((self.im_height, self.im_width))
        self.wde_rain = np.zeros((self.im_height, self.im_width))
        self.eta_rain =  np.zeros((self.im_height, self.im_width))


        for i in range(self.im_height):
            for j in range(self.im_width):
                
                
                # init_suit_chk_data = getInitialSuitabilityCheckData(self.set_mask, self.im_mask[i,j], self.nodata_val, self.set_Permafrost_screening, self.permafrost_class[i,j], 
                #                                                     self.set_tclimate_screening, self.t_climate[i,j], self.no_t_climate)
                
                if InitialSuitabilityCheck(self.set_mask, self.im_mask[i,j], self.nodata_val, 
                                self.set_Permafrost_screening, self.permafrost_class[i,j], self.set_tclimate_screening,
                                self.t_climate[i,j], self.no_t_climate):

                    count_pixel_completed = count_pixel_completed + 1
                    print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')
                    continue

                if CycleLengthChecking(self.LGPT5[i,j], self.LGPT10[i,j], self.LGP[i,j], self.min_cycle_len, 'R', self.perennial, self.min_temp):
                    count_pixel_completed = count_pixel_completed + 1
                    print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')
                    continue

                self.final_yield_rain[i,j], self.fc1_rain[i,j], self.fc2_rain[i,j], self.eta_rain[i,j], self.wde_rain[i,j], self.crop_calender_rain[i,j] = self.simulateRainfedSugarcaneCropCycleOneTest(i, j)

                count_pixel_completed = count_pixel_completed + 1
                print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')

        
        print('\nRainfed Sugarcane Simulation Completed')
    
    # def simulateRainfedCropCycle(self, start_doy:int =1, end_doy:int= 365, step_doy:int = 1, leap_year:bool = False):
    #     """Running the Rainfed crop cycle calculation/simulation.

    #     Args:
    #         start_doy (int, optional): Starting Julian day for simulating period. Defaults to 1.
    #         end_doy (int, optional): Ending Julian day for simulating period. Defaults to 365.
    #         step_doy (int, optional): Spacing (in days) between 2 adjacent crop simulations. Defaults to 1.
    #         leap_year (bool, optional): whether or not the simulating year is a leap year. Defaults to False.

    #     """
    #     bar = '-' * 25
    #     msg = {True:'Activated', False:'Deactivated'}
    #     print(f'EXECUTING {self.crop_name} Irrigated Crop Simulation\n{bar}', end = '\n')
    #     print(f'Masking\t\t\t\t={msg[self.set_mask]}\nThermal Climate Screening\t={msg[self.set_tclimate_screening]}', end= '\n')
    #     print(f'TSUM Screening\t\t\t={msg[self.set_Tsum_screening]}\nPermafrost Screening\t\t={msg[self.set_Permafrost_screening]}', end= '\n')
    #     print(f'Crop-specific Rule Screening\t={msg[self.setCropSpecificRule]}\n{bar}', end= '\n')
        
    #     # just a counter to keep track of progress
    #     count_pixel_completed = 0
    #     total = self.im_height * self.im_width

    #     # this stores final result
    #     self.final_yield_rain = np.zeros((self.im_height, self.im_width))
    #     self.crop_calender_rain = np.zeros((self.im_height, self.im_width), dtype=int)
    #     self.fc2_rain = np.zeros((self.im_height, self.im_width))
    #     self.fc1_rain = np.zeros((self.im_height, self.im_width))
    #     self.wde_rain = np.zeros((self.im_height, self.im_width))
    #     self.eta_rain =  np.zeros((self.im_height, self.im_width))


    #     for i in range(self.im_height):
    #         for j in range(self.im_width):
                
                
    #             # init_suit_chk_data = getInitialSuitabilityCheckData(self.set_mask, self.im_mask[i,j], self.nodata_val, self.set_Permafrost_screening, self.permafrost_class[i,j], 
    #             #                                                     self.set_tclimate_screening, self.t_climate[i,j], self.no_t_climate)
                
    #             if InitialSuitabilityCheck(self.set_mask, self.im_mask[i,j], self.nodata_val, 
    #                             self.set_Permafrost_screening, self.permafrost_class[i,j], self.set_tclimate_screening,
    #                             self.t_climate[i,j], self.no_t_climate):

    #                 count_pixel_completed = count_pixel_completed + 1
    #                 print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')
    #                 continue

    #             if CycleLengthChecking(self.LGPT5[i,j], self.LGPT10[i,j], self.LGP[i,j], self.min_cycle_len, 'R', self.perennial, self.min_temp):
    #                 count_pixel_completed = count_pixel_completed + 1
    #                 print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')
    #                 continue

    #             climate_data = DuplicateOneYearClimateData(self.minT_daily[i,j,:], self.maxT_daily[i,j,:], self.meanT_daily[i,j,:], 
    #                                                         self.shortRad_daily[i,j,:], self.wind2m_daily[i,j,:], self.totalPrec_daily[i,j,:], 
    #                                                         self.rel_humidity_daily[i,j,:], self.pet_daily[i,j,:])
                
    #             cycle_len_check_data = getCycleLengthCheckingData(self.LGPT5[i,j], self.LGPT10[i,j], self.LGP[i,j],
    #                                                             self.min_cycle_len, 'R', self.perennial, self.min_temp, 
    #                                                             self.max_cycle_len, self.cycle_len, self.hibernate)
                
    #             LAI_HI_data = getLAIandHIdata(self.LAi, self.HI, self.aLAI, self.bLAI, self.aHI, self.bHI)

    #             if len(np.array(self.Sa).shape) == 2:
    #                 Sa_temp = self.Sa[i, j]
    #             else:
    #                 Sa_temp = self.Sa

    #             values = simulateCropCycleOneLocation(start_doy, end_doy, step_doy, leap_year, cycle_len_check_data, LAI_HI_data, climate_data,
    #                                 self.latitude[i,j], self.elevation[i,j], self.plant_height, self.set_Tsum_screening, self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO,
    #                             self.setCropSpecificRule, self.data, self.legume, self.adaptability,
    #                             self.kc, self.d_per, Sa_temp, self.D1, self.D2, self.crop_group, self.yloss_f_all, self.yloss_f, 'R', self.crop_name)
                
    #             self.final_yield_rain[i,j] = values[0]
    #             self.wde_rain[i,j] = values[1]
    #             self.eta_rain[i,j]= values[2]
    #             self.fc1_rain[i,j] = values[3]
    #             self.fc2_rain[i,j] = values[4]
    #             self.crop_calender_rain[i,j] = values[5]

    #             count_pixel_completed = count_pixel_completed + 1
    #             print(f'\rDone:{round(count_pixel_completed / total*100, 2)} %', end='\r')

        
    #     print('\nRainfed Crop Simulation Completed')

    
    def simulateRainfedSugarcaneCropCycleOneTest(self, i:int, j:int):
        """Running the Rainfed Perennial crop cycle calculation/simulation for a single location.

        Args:
            i,j (int): row and column index.
        
        Return:
            yld1_r, fc1_r, fc2_r, eta_r, wde_r, cbd_r: maximum attainable yield, thermal reduction factor, water deficit factor, total actual evapotranspiration,
                                                       crop beginning date.
        """
        # set up the 2-year climate data
        climate_data = DuplicateOneYearClimateData(self.minT_daily[i,j,:], self.maxT_daily[i,j,:], self.meanT_daily[i,j,:], 
                                                self.shortRad_daily[i,j,:], self.wind2m_daily[i,j,:], self.totalPrec_daily[i,j,:], 
                                                self.rel_humidity_daily[i,j,:], self.pet_daily[i,j,:])
        
        min_T = climate_data[0]
        max_T = climate_data[1]
        mean_T = climate_data[2]
        shrt_rd = climate_data[3] # W/m2
        wind_sp = climate_data[4]
        pr = climate_data[5]
        rel_hum = climate_data[6]
        eto = climate_data[7]

        # initial variable setting
        yld0_r = 0.
        fc1_r = 0.
        fc2_r = 0.
        eta_r = 0.
        wde_r = 0.
        cbd_r = 0

        # Initial straightforward yes/no checking
        if InitialSuitabilityCheck(self.set_mask, self.im_mask[i,j], self.nodata_val, 
                                self.set_Permafrost_screening, self.permafrost_class[i,j], self.set_tclimate_screening,
                                self.t_climate[i,j], self.no_t_climate):
            # print('Initial Suitability not met')
            return yld0_r, fc1_r, fc2_r, eta_r, wde_r, cbd_r
        

        # beginning date of period with LGP longest and the longest LGP itself
        # -----------------------------------------------------------------------
        islgp = islgpt(mean_T[:365])
        xx = val10day(self.Eta365[i,j,:])
        yy = val10day(self.Etm365[i,j,:])
        lgp_whole = np.divide(xx, yy, where= yy>0, out = np.ones(xx.shape))

        count = []

        for k in range(len(lgp_whole)):
            if islgp[k] == 1 and lgp_whole[k] >=0.4:
                count.append(1)
            else:
                count.append(0)
        
        # if there are no growing days year-round, skip cycle searching
        if sum(count) ==0:
            # print('No LGP days')
            return yld0_r, fc1_r, fc2_r, eta_r, wde_r, cbd_r
        
        # find the length of the cycles
        lgp_components = search_cycles(count)

        # if there are no growing periods year-round, skip calculation.
        if len(lgp_components[0])==0:
            length = 0
        else:
            # find all days of each cycle
            sum_list = []
            
            for k in lgp_components[0]:
                if len(k) ==0:
                    sum_list.append(0)
                else:
                    sum_list.append(sum(k))
            
            # change the list into numpy array for possible error occurrence
            sum_list = np.array(sum_list)
            lgp_bd = lgp_components[1]
            idx = np.argwhere(sum_list == np.nanmax(sum_list))[0][0]
            blgp = lgp_bd[idx] +1

            length = int(np.nanmax(sum_list))

        if length ==0 or blgp == 0:
            # print('No beginning date of LGP')
            return yld0_r, fc1_r, fc2_r, eta_r, wde_r, cbd_r

        # ------------------------------------------------------------------------
        Tm = mean_T[:365]
        smTmean =  getInterpolatedTempData(Tm)

        
        # Effective cycle length determination

        lgpeff = min(length, self.max_cycle_len)

        # Minimum cycle length checking
        if lgpeff < self.min_cycle_len:
            # print('Effient cycle length does not meet minimum cycle length requirement')
            return yld0_r, fc1_r, fc2_r, eta_r, wde_r, cbd_r
        else:
            lenw = min(lgpeff, self.max_cycle_len)
            bday = blgp
            # eday = elgp # End date is not necessary

        bgro = getSugarcaneCCD(bday, smTmean, rel_hum)

        # From that bgro, the biomass calculation will be applied
        if bgro==0:
            # print('No beginning of growing period')
            return yld0_r, fc1_r, fc2_r, eta_r, wde_r, cbd_r

        tmp_profile = getTemperatureProfile(mean_T[bgro:bgro+365])
        crop_specific_profile = calculateTemperatureProfileClasses(self.data, tmp_profile, 365)


        # Thermal suitability starts here
        fc1_r = 1.
        tsum0 = getTemperatureSum(mean_T[bday:bday+365], 0)

        fc1_r = getReductionFactorNumba(self.set_Tsum_screening, self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO, tsum0,
                            self.setCropSpecificRule, crop_specific_profile, False, 0.)
        
        if fc1_r <0.001:
            return yld0_r, fc1_r, fc2_r, eta_r, wde_r, cbd_r
        
        # if the effective cycle length is less than reference cycle length, LAI and HI will be adjusted
        if lenw < self.cycle_len: 
            LAI_irr, HI_irr = LAI_HI_adjustment(self.LAi, self.HI, self.aLAI, self.bLAI, self.aHI, self.bHI, lenw)
        else:
            LAI_irr, HI_irr = self.LAi, self.HI
        

        bnr = calculateBiomassNumba(bgro-1, bgro-1+lenw, lenw, self.latitude[i,j],
                         shrt_rd[bgro-1:bgro-1+lenw], mean_T[bgro-1:bgro-1+lenw], min_T[bgro-1:bgro-1+lenw], max_T[bgro-1:bgro-1+lenw],
                        LAI_irr, self.legume, self.adaptability, self.leap_year)
        
        yld0_r = bnr * HI_irr * fc1_r

        if len(np.array(self.Sa).shape) == 2:
            Sa_temp = self.Sa[i, j]
        else:
            Sa_temp = self.Sa
        
        #Crop Water Requirement
        wde_r, fc2_r, eta_r,  yld1_r = calculateMoistureLimitedYieldNumba('R', self.kc, self.d_per, lenw, pr[bgro-1:bgro-1+lenw], eto[bgro-1:bgro-1+lenw],
                                                                        min_T[bgro-1:bgro-1+lenw], max_T[bgro-1:bgro-1+lenw], self.plant_height, wind_sp[bgro-1:bgro-1+lenw],
                                                                        Sa_temp, self.D1, self.D2, mean_T[bgro-1:bgro-1+lenw], self.crop_group, self.yloss_f_all, self.yloss_f, self.perennial, yld0_r)

        cbd_r = bgro

        return yld1_r, fc1_r, fc2_r, eta_r, wde_r, cbd_r
    


    def simulateIrrigatedSugarcaneCropCycleOneTest(self, i:int, j:int):
        """Running the Irrigated Sugarcane crop cycle calculation/simulation for a single location.

        Args:
            i,j (int): row and column index.
        
        Return:
            yld1_i, fc1_i, fc2_i, eta_i, wde_i, cbd_i: maximum attainable yield, thermal reduction factor, water deficit factor, total actual evapotranspiration,
                                                       crop beginning date.
        """
        # set up the 2-year climate data
        climate_data = DuplicateOneYearClimateData(self.minT_daily[i,j,:], self.maxT_daily[i,j,:], self.meanT_daily[i,j,:], 
                                                self.shortRad_daily[i,j,:], self.wind2m_daily[i,j,:], self.totalPrec_daily[i,j,:], 
                                                self.rel_humidity_daily[i,j,:], self.pet_daily[i,j,:])
        
        min_T = climate_data[0]
        max_T = climate_data[1]
        mean_T = climate_data[2]
        shrt_rd = climate_data[3] # W/m2
        wind_sp = climate_data[4]
        pr = climate_data[5]
        rel_hum = climate_data[6]
        eto = climate_data[7]

        # initial variable setting
        yld0_i = 0.
        fc1_i = 0.
        fc2_i = 0.
        eta_i = 0.
        wde_i = 0.
        cbd_i = 0

        # Initial straightforward yes/no checking
        if InitialSuitabilityCheck(self.set_mask, self.im_mask[i,j], self.nodata_val, 
                                self.set_Permafrost_screening, self.permafrost_class[i,j], self.set_tclimate_screening,
                                self.t_climate[i,j], self.no_t_climate):
            # print('Initial Suitability not met.')
            return yld0_i, fc1_i, fc2_i, eta_i, wde_i, cbd_i
        
        Tm = mean_T[:365]
        smTmean =  getInterpolatedTempData(Tm)

        # beginning of period with T >0, >5, >10 (All are array indices)
        lgbt = [0,0,0]
        lgbt[0] = findbeginningLGPT(smTmean, 0)
        lgbt[1] = findbeginningLGPT(smTmean, 5)
        lgbt[2] = findbeginningLGPT(smTmean, 10)
        
        # thermal growing periods
        lgpt = [0,0,0]
        lgpt[0] = getTemperatureGrowingPeriod(smTmean, 0)
        lgpt[1] = getTemperatureGrowingPeriod(smTmean, 5)
        lgpt[2] = getTemperatureGrowingPeriod(smTmean, 10)

        # print('Beginning Dates when Temp >0, >5, >10')

        if self.min_temp <= 8:
            length = lgpt[1]
            blgp = lgbt[1]
            elgp = blgp + length -1 # End date is not necessary
        else:
            length = lgpt[2]
            blgp = lgbt[2]
            elgp = blgp + length -1
        
        lgpeff = min(length, self.max_cycle_len)

        # Minimum cycle length checking. If satisfied, the beginning date of crop growth is determined.
        if lgpeff < self.min_cycle_len:
            # print('Effective cycle length not meet minimum length requirement.')
            return yld0_i, fc1_i, fc2_i, eta_i, wde_i, cbd_i
        else:
            lenw = min(lgpeff, self.max_cycle_len)
            bday = blgp
            eday = elgp
        
        
        # Determination of Sugarcane Crop Calendar Searching starting here
        bgro = getSugarcaneCCD(bday, smTmean, rel_hum)
        # From that bgro, the biomass calculation will be applied
        if bgro==0:
            # print('No beginning of the growing period')
            return yld0_i, fc1_i, fc2_i, eta_i, wde_i, cbd_i
        

        tmp_profile = getTemperatureProfile(mean_T[bgro:bgro+365])
        # Thermal suitability starts here
        fc1_i = 1.
        tsum0 = getTemperatureSum(mean_T[(bgro-1):(bgro-1+365)], 0)

        crop_specific_profile = calculateTemperatureProfileClasses(self.data, tmp_profile, 365)

        fc1_i = getReductionFactorNumba(self.set_Tsum_screening, self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO, tsum0,
                            self.setCropSpecificRule, crop_specific_profile, False, 0.)
        
        if fc1_i <0.001:
            return yld0_i, fc1_i, fc2_i, eta_i, wde_i, cbd_i
        
        # if the effective cycle length is less than 
        if lenw < self.cycle_len:
            LAI_irr, HI_irr = LAI_HI_adjustment(self.LAi, self.HI, self.aLAI, self.bLAI, self.aHI, self.bHI, lenw)
        else:
            LAI_irr, HI_irr = self.LAi, self.HI
        
        

        bni = calculateBiomassNumba(bgro-1, bgro-1+lenw, lenw, self.latitude[i,j],
                        shrt_rd[bgro-1:bgro-1+lenw], mean_T[bgro-1:bgro-1+lenw], min_T[bgro-1:bgro-1+lenw], max_T[bgro-1:bgro-1+lenw],
                        LAI_irr, self.legume, self.adaptability, self.leap_year)
        
        yld0_i = bni * HI_irr * fc1_i

        if len(np.array(self.Sa).shape) == 2:
            Sa_temp = self.Sa[i, j]
        else:
            Sa_temp = self.Sa
        
        #Crop Water Requirement
        wde_i, fc2_i, eta_i,  yld1_i = calculateMoistureLimitedYieldNumba('I', self.kc, self.d_per, lenw, pr[bgro-1:bgro-1+lenw], eto[bgro-1:bgro-1+lenw],
                                                                                        min_T[bgro-1:bgro-1+lenw], max_T[bgro-1:bgro-1+lenw], self.plant_height, wind_sp[bgro-1:bgro-1+lenw],
                                                                                        Sa_temp, self.D1, self.D2, mean_T[bgro-1:bgro-1+lenw], self.crop_group, self.yloss_f_all, self.yloss_f, self.perennial, yld0_i)

        cbd_i = bgro

        return yld1_i, fc1_i, fc2_i, eta_i, wde_i, cbd_i

        
#"""------------------       MAJOR CROP SIMULATION ROUTINE ENDS HERE    ----------------------------------"""
#"""------------------    IMPORTANT FUNCTIONALITIES TO CROP SIMULATIONS ------------------------------------"""
# These important functionalities are not embedded within Module 2 object class for future modification purposes.

def simulateCropCycleOneLocation(start_doy:int, end_doy:int, step_doy:int, leap_year:bool, cycle_len_check_data, LAI_HI_data, climate_data,
                                    lat:float, elev:float, plant_height:float, set_TSUM_screening:bool, LnS:int, LsO:int, LO:int, HnS:int, HsO:int, HO:int,
                                set_CropSpecificRule:bool, data, legume:int, adaptability:int,
                                kc, d_per, Sa, D1:float, D2:float, crop_group:int, yloss_f_all:float, yloss_f, irr_or_rain:str,crop_name:str):
    
    """NESTED FUNCTION: All simulation procedures are done for a single pixel location"""
    final_yld:float = 0.
    ccd: int = 0
    wde: float = 0.
    eta:float = 0.
    fc1: float = 0.
    fc2: float = 0.
    cycle_len:float = 0

    lgpt5, lgpt10, lgp, min_cycle_len, irr_or_rain, perennial_flg, min_temp_threshold, max_cycle_len, ref_cycle_len, hibernating_flag = cycle_len_check_data
    # set_mask, im_mask, nodata_val, set_Permafrost_screening, permafrost_class, set_tclimate_screening, t_climate, no_t_climate = init_suit_data

    lai, hi, alai, blai, ahi, bhi = LAI_HI_data

    # Effective cycle length determination for perennial crops
    if perennial_flg:
        cycle_len = DefineEffectiveCycleLength(min_temp_threshold, max_cycle_len,lgpt5, lgpt10, lgp, irr_or_rain, ref_cycle_len)
    # For     
    else:
        # For annual crops, no cycle length adjustment is needed.
        cycle_len = ref_cycle_len
    
    # For perennials, the effective cycle length will be used to adjust the LAI and HI
    LAi = 0.
    HI = 0

    if perennial_flg:
        LAi, HI = LAI_HI_adjustment(lai, hi, alai, blai, ahi, bhi, cycle_len)
    else:
        LAi, HI = lai, hi
    
    if LAi <= 0.001 or HI <= 0.001:
        return final_yld, wde, eta, fc1, fc2, ccd
    else:
        val = CropCycleLooping(start_doy, end_doy, step_doy, climate_data, min_temp_threshold, perennial_flg,
                        cycle_len, set_TSUM_screening, LnS, LsO, LO, HnS, HsO, HO, set_CropSpecificRule, data, 
                        lat, LAi, HI, legume, adaptability, plant_height,
                        kc, d_per, Sa, D1, D2, crop_group, yloss_f_all, yloss_f, irr_or_rain, leap_year, crop_name, hibernating_flag)
        
        final_yld, wde, eta, fc1, fc2, ccd = val[0], val[1], val[2], val[3], val[4], val[5] 

        return final_yld, wde, eta, fc1, fc2, ccd


        

def CropCycleLooping(start_doy:int, end_doy:int, step_doy:int, climate_data, min_T_threshold, perennial_flag:bool,
                     cycle_len:int, set_TSUM_screening:bool, LnS, LsO, LO, HnS, HsO, HO, set_CropSpecificRule:bool, data, 
                     lat:float, lai:float, hi:float, legume:int, adaptability:int, plant_height:float,
                     kc, d_per, Sa, D1:float, D2:float, crop_group:int, yloss_f_all:float, yloss_f, irr_or_rain:str, leap_year:bool,
                     crop_name:str,hibernation_flag:bool):

    """NESTED FUNCTION: evaluates the loop-based crop cycle simulation cycle."""
    # Only call the climate data once all initial flag checks are False
    min_T = climate_data[0]
    max_T = climate_data[1]
    mean_T = climate_data[2]
    shrt_rd = climate_data[3]
    wind_sp = climate_data[4]
    pr = climate_data[5]
    # rel_hum = climate_data[6]
    eto = climate_data[7]

    yld:float = 0.
    wde:float = 0.
    eta:float = 0.
    fc1:float = 0.
    fc2:float = 0.
    ccd:int = 0

    # important variable returning
    yd_arr = np.empty(0, dtype= float)
    wde_arr = np.empty(0, dtype= float)
    eta_arr= np.empty(0, dtype= float)
    fc1_arr= np.empty(0, dtype= float)
    fc2_arr= np.empty(0, dtype= float)

    for i_cycle in range(start_doy-1, end_doy, step_doy):

        cycle_yld:float = 0.
        cycle_wde: float = 0.
        cycle_eta:float = 0.
        cycle_fc1: float = 0.
        cycle_fc2: float = 0.

        """Check if the first day of a cycle meets minimum temperature requirement. If not, all outputs will be zero.
            And iterates to next cycle."""
        if mean_T[i_cycle]< min_T_threshold:
            yd_arr = np.append(yd_arr, 0.)
            wde_arr = np.append(wde_arr, 0.)
            eta_arr = np.append(eta_arr, 0.)
            fc1_arr = np.append(fc1_arr, 0.)
            fc2_arr = np.append(fc2_arr, 0.)
            continue
        
        cycle_fc1 = 1.
        # Thermal Screening 
        if perennial_flag:
            tsum0 = getTemperatureSum(mean_T[i_cycle:i_cycle+365], 0)
        else:
            tsum0 = getTemperatureSum(mean_T[i_cycle:i_cycle+cycle_len], 0)
        
        if hibernation_flag:
            #Critical breaking temperature (cbtr)
            jd1, jd2, vern_factor = check_vernalization(mean_T[i_cycle:i_cycle+365],crop_name) 
        else:
            vern_factor = 1.
        
        tmp_profile = calculateTemperatureProfileClasses(data, mean_T[i_cycle:i_cycle+cycle_len], cycle_len)
        cycle_fc1 = getReductionFactorNumba(set_TSUM_screening, LnS, LsO, LO, HnS, HsO, HO, tsum0,
                            set_CropSpecificRule, tmp_profile, hibernation_flag, vern_factor)
        
        if cycle_fc1 <=0.01 or not check_hibernating_crop_suitability(mean_T[i_cycle:i_cycle+365],crop_name):
            cycle_fc1, cycle_fc2, cycle_yld = 0., 0., 0.
            yd_arr = np.append(yd_arr, 0.)
            wde_arr = np.append(wde_arr, 0.)
            eta_arr = np.append(eta_arr, 0.)
            fc1_arr = np.append(fc1_arr, 0.)
            fc2_arr = np.append(fc2_arr, 0.)
            continue
        else:
            start = int(i_cycle+1)
            end= int(i_cycle+cycle_len+1)
            endi = int(i_cycle+cycle_len)

            # Biomass Calculation
            bn = calculateBiomassNumba(start, end, cycle_len, lat, shrt_rd[i_cycle:endi],
                                         mean_T[i_cycle:endi], min_T[i_cycle:endi],max_T[i_cycle:endi],
                                         lai, legume, adaptability, leap_year)
            cycle_yld = bn * hi * cycle_fc1

            #Crop Water Requirement
            cycle_wde, cycle_fc2, cycle_eta,  cycle_yld = calculateMoistureLimitedYieldNumba(irr_or_rain, kc, d_per, cycle_len, pr[i_cycle:endi], eto[i_cycle:endi],
                                                                                            min_T[i_cycle:endi], max_T[i_cycle:endi], plant_height, wind_sp[i_cycle:endi],
                                                                                            Sa, D1, D2, mean_T[i_cycle:endi], crop_group, yloss_f_all, yloss_f, perennial_flag, cycle_yld)
            
            # Appending to the list
            yd_arr = np.append(yd_arr, cycle_yld)
            wde_arr = np.append(wde_arr, cycle_wde)
            eta_arr = np.append(eta_arr, cycle_eta)
            fc1_arr = np.append(fc1_arr, cycle_fc1)
            fc2_arr = np.append(fc2_arr, cycle_fc2)
    
    # find DOY index of the maximum attainable yield
    idx = FindOptimalCropCalendarDOY(yd_arr)

    yld = yd_arr[idx]
    wde= wde_arr[idx]
    eta= eta_arr[idx]
    fc1= fc1_arr[idx]
    fc2= fc2_arr[idx]
    ccd = idx+1

    return yld, wde, eta, fc1, fc2, ccd

 #--------------------------------------------- Functions for Getting the Intermediate Values of Module II for Validation  ---------------------------------------------------------------#

def FindOptimalCropCalendarDOY(yld_cycles):
    """
    Standardized AEZ crop calendar determination routine.
    Returns an array index of maximum attainable yield for crop calendar.
    
    Args:
        yld_cycles (list): a python list of all cycles' yield
    Return:
        ccd_idx (int): crop calendar day index
    """

    i:int = 0

    if np.sum(np.ceil(yld_cycles)) in [365, 366]:
        return i
    else:
        i = np.argwhere(yld_cycles == np.nanmax(yld_cycles))[0][0]
        return i

@nb.jit(nopython = True)
def DefineEffectiveCycleLength(min_temp_threshold:float, max_cycle_len:int, 
                               lgpt5:int, lgpt10:int, lgp:int, irr_or_rain:str, ref_cycle_len:int):
    
    """Only this effective cycle length will be done to PERENNIAL CROPS.
    Different considerations for irrigated and rainfed conditions are implemented.

    Args:
        min_temp_threshold (float): crop-specific minimum temperature threshold [Unit: Deg Celsius]
        max_cycle_len (int): crop-specific maximum cycle length [Unit: Days]
        lgpt5 (int)
    """
    eff_cycle_len:int = 0

    # Irrigated perennials
    if irr_or_rain == 'I':
        if min_temp_threshold <= 8:
            eff_cycle_len = min(lgpt5, max_cycle_len)
        else:
            eff_cycle_len = min(lgpt10, max_cycle_len)
    # Rainfed perennials
    else:
        eff_cycle_len = min(lgp, max_cycle_len)
    
    if eff_cycle_len > ref_cycle_len:
        eff_cycle_len = ref_cycle_len
    
    return eff_cycle_len

def InitialSuitabilityCheck(set_mask:bool, mask, nodata_val:int, set_Permafrost_screening:bool, permafrost_class, 
                            set_tclimate_screening:bool, t_climate, no_t_climate:list[int]):
    """Initial suitability checking step."""

    flg:bool = False

    # check current location (pixel) is outside of study area or not. if it's outside of study area goes to next location (pixel)
    if set_mask:
        if mask == nodata_val:
            flg = True
            return flg

    # 2. Permafrost screening
    if set_Permafrost_screening:
        if np.logical_or(permafrost_class == 1, permafrost_class == 2):
            flg = True
            return flg
    
    if set_tclimate_screening:
        if t_climate in no_t_climate:
            flg = True
            return flg
    
    return flg

def CycleLengthChecking(lgpt5, lgpt10, lgp, min_cycle_len, irr_or_rain, perennial_flg, min_threshold):
    """Check whether the cycle length is within minimum cycle length requirements"""
    flg:bool = False

    # For annual crop
    if not perennial_flg:
        if irr_or_rain == 'I': # for irrigated condition
            if lgpt5 < min_cycle_len:
                flg = True
                return flg
        else:
            if lgp < min_cycle_len: # for rainfed condition
                flg = True
                return flg
    else:
        # For perennial crop
        if irr_or_rain == 'I': # for irrigated condition
            if min_threshold <= 8 and lgpt5 < min_cycle_len:
                flg = True
                return flg
            elif min_threshold >8 and lgpt10 < min_cycle_len:
                flg = True
                return flg
        else:
            if lgp < min_cycle_len:
                flg = True
                return flg
                
    return flg

def getInitialSuitabilityCheckData(set_mask, im_mask, nodata_val, set_Permafrost_screening, permafrost_class, set_tclimate_screening, t_climate, no_t_climate):
    """Grouping all flags and necessary variables required for Initial Suitability Check step."""
    return set_mask, im_mask, nodata_val, set_Permafrost_screening, permafrost_class, set_tclimate_screening, t_climate, no_t_climate

def getCycleLengthCheckingData(lgpt5, lgpt10, lgp, min_cycle_len, irr_or_rain, perennial_flg, min_temp_threshold, max_cycle_len, ref_cycle_len, hibernation_flg):
    """Grouping all the flags and necessary variables required for Cycle Length Checking step."""
    return lgpt5, lgpt10, lgp, min_cycle_len, irr_or_rain, perennial_flg, min_temp_threshold, max_cycle_len, ref_cycle_len, hibernation_flg
def getLAIandHIdata(LAI, HI, aLAI, bLAI, aHI, bHI):
    """Grouping all LAI and HI parameterizations for LAI and HI adjustment."""
    return LAI, HI, aLAI, bLAI, aHI, bHI

@nb.jit(nopython = True)
def DuplicateOneYearClimateData(min_temp, max_temp, mean_temp, short_rad, wind_sp, precip, rel_humid, eto):
    """Duplicate another year for computation purpose."""
    min_temp2 = np.concatenate((min_temp, min_temp))
    max_temp2 = np.concatenate((max_temp, max_temp))
    mean_temp2 = np.concatenate((mean_temp, mean_temp))
    short_rad2 = np.concatenate((short_rad, short_rad))
    wind_sp2 = np.concatenate((wind_sp, wind_sp))
    precip2 = np.concatenate((precip, precip))
    rel_humid2 = np.concatenate((rel_humid, rel_humid))
    eto2 = np.concatenate((eto, eto))

    return min_temp2, max_temp2, mean_temp2, short_rad2, wind_sp2, precip2, rel_humid2, eto2

@nb.jit(nopython = True)
def LAI_HI_adjustment(LAI, HI, aLAI, bLAI, aHI, bHI, eff_cycle_len):
    """Leaf Area Index and Harvest Index adjustment based on effective cycle length"""

    adj_LAI:float = 0.
    adj_HI:float = 0.

    """LAI adjustment"""
    if eff_cycle_len - aLAI > bLAI:
        adj_LAI =LAI
    elif eff_cycle_len - aLAI < 0:
        adj_LAI = 0.
    else:
        adj_LAI = LAI * ((eff_cycle_len-aLAI)/bLAI)
    
    """HI adjustment"""
    if eff_cycle_len - aHI > bHI:
        adj_HI =HI
    elif eff_cycle_len - aHI < 0:
        adj_HI = 0.
    else:
        adj_HI =  HI * ((eff_cycle_len-aHI)/bHI)
    
    return adj_LAI, adj_HI

def getSugarcaneCCD(begin_day, mean_T, rh):
    """ 
    Get the Sugarcane specific starting day of the growing period.
    
    Args:
        begin_day (int): beginning day determined by agro-climatic effective growing period
        mean_T (1D NumPy Array): One-year mean temperature [Unit: Deg C]
        rh (1D NumPy Array): One year relative humidity [Unit: 0-1, Unitless]
        TP_data (pd.DataFrame): crop-specifc temperature profile constraints
        perennial_flag (Boolean): perennial flag [True: perennial, False:non-perennial]
    Return:
        jstrt(int): starting date of the crop growing period
    """

    # Calculate the temperature profile
    # For perennial crops, 365 days length average temperature is used.
    A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, B0, B1, B2, B3, B4, B5, B6, B7, B8, B9 = getTemperatureProfile(mean_T[begin_day:begin_day+365])


    # Skip when tempeature profile doesn't meet criteria
    if (A6 + B6 >0) or (A5 + B5 > 0.167*360) or (A5 + B5 + A4 + B4 > 0.667 * 360) or (A3 +B3 + A4 + B4 < 0.084 * 360):
        # print('Temperature Profile Not met')
        return 0

    # set the starting date when the meeting is considered
    jstrt = np.argmax(mean_T[begin_day:begin_day+365]) +1 - (A0+A1+A2+A3+A4+A5)
    if jstrt <1: jstrt = jstrt + 365
    jripe = jstrt+270

    # average humidity at the maturation stage must be less than 60 %
    rh2 = np.concatenate([rh, rh])

    rh = rh2[jripe-1:jripe-1+60]
    rh = rh[rh>0.]
    rh = np.sum(rh)/60
    if rh > 0.6:
        # print('Relative Humidity requirement not met')
        return 0
    
    return jstrt

""" NEW - KB Jan 2025 """
def check_vernalization(Ta365: np.ndarray, crop_name: str) -> tuple:
    """
    Check if the vernalization requirement is met for a given crop.
    
    Parameters:
    - crop_name: str, crop name (e.g., 'winter_wheat', 'barley', 'rye')
    - Ta365: np.ndarray, 1D array of daily temperatures for a full year (365 values)
    
    Returns:
    - tuple: (jd1, jd2, vern_factor) representing the start and end of the vernalization period
    """

    # Define crop-specific vernalization parameters
    vernalization_params = {
        "winter_wheat": {"Topt": 5, "Tmx": 15, "Tmn": -1, "V100": 45, "V0": 10},
        "winter_barley": {"Topt": 4, "Tmx": 12, "Tmn": 0, "V100": 35, "V0": 8},
        "winter_rye": {"Topt": 5, "Tmx": 15, "Tmn": -2, "V100": 45, "V0": 10},
        "winter_rape": {"Topt": 3, "Tmx": 10, "Tmn": 0, "V100": 30, "V0": 8}
    }

    # Get parameters for the selected crop    
    params = vernalization_params[crop_name]
    Topt, Tmx, Tmn, V100, V0 = params["Topt"], params["Tmx"], params["Tmn"], params["V100"], params["V0"]

    # Determine the start (jd1) and end (jd2) of vernalization period
    # Identify the coldest period of the year
    jd1 = np.argmin(Ta365)  # Start at the coldest day
    jd2 = jd1 + 60  # Vernalization typically lasts 60 days
    

    # Handle year wrap-around
    if jd1 < 1:
        jd1 += 365
        jd2 += 365

    # Ensure jd2 does not exceed 365 (handling wrap-around)
    jd2 = min(jd2, 365)

    # Compute vernalization factor
    vern_factor = vrnfct(jd1, jd2, Tmn, Topt, Tmx, V100, V0, Ta365)
    # print(f'Vern start day: {jd1}, end: {jd2}, vern_fct={vern_factor}')

    return jd1, jd2, vern_factor

def vrnfct(jd1: int, jd2: int, Tmn: float, Topt: float, Tmx: float, V100: float, V0: float, Ta365: np.ndarray) -> int:
    """
    Calculate the vernalization factor.

    Args:
        jd1 (int): Start day of the period.
        jd2 (int): End day of the period.
        Tmn (float): Minimum temperature for vernalization.
        Topt (float): Optimal temperature for vernalization.
        Tmx (float): Maximum temperature for vernalization.
        V100 (float): Vernalization threshold for full effectiveness.
        V0 (float): Minimum vernalization threshold.
        Ta365 (list): Daily temperatures for the year (365 values).

    Returns:
        float: The calculated vernalization factor.
    """
    if jd1 <= 0:
        return 0.0

    # Calculate the sum of effective vernalization days 
    alf = np.log(2.0) / np.log((Tmx - Tmn) / (Topt - Tmn))
    beta = (Topt - Tmn) ** alf
    bet2 = beta ** 2

    # Sum of effective vernalization days
    vvd = 0.0
    for j in range(jd1, jd2 + 1):
        # Circular indexing for days of the year
        Ta = Ta365[(j - 1) % 365]  # Use (j-1) to match zero-based indexing in Python
        if Tmn < Ta < Tmx:
            xx = (Ta - Tmn) ** alf
            x2 = xx ** 2
            fvn = (2.0 * xx * beta - x2) / bet2
            vvd += fvn

    # Calvulate vernalization factor (a function of VVD)
    ## Calculate intermediate vernalization thresholds
    V50 = 0.5 * V100
    V90 = (0.9 / 0.1) ** 0.2 * V100
    V10 = (0.1 / 0.9) ** 0.2 * V100

    # Compute vrnfct based on vvd
    if vvd >= V100:
        vrnfct = 1.0 # Vernalization requirement met
    elif vvd >= V90:
        vrnfct = 0.9 + 0.1 * (vvd - V90) / (V100 - V90)
    elif vvd <= V0: 
        vrnfct = 0.0 # Vernalization not met
    elif vvd <= V10:
        vrnfct = 0.1 * (vvd - V0) / (V10 - V0)
    else:
        v5 = vvd ** 5
        c5 = V50 ** 5
        vrnfct = v5 / (v5 + c5)

    return vrnfct


def check_hibernating_crop_suitability(Ta, crop):
    """
    Check if a hibernating crop is suitable based on temperature conditions.
    
    Parameters:
    Ta (np.array): 1D array of daily temperatures (365 elements, °C)
    crop (str): Crop type (must be one of "winter_wheat", "winter_barley", "winter_rye","winter_rape")
    
    Returns:
    str: Suitability status ("Suitable", "Sub-optimum", or "Not Suitable")
    """
    # Define critical temperature thresholds
    crop_thresholds = {
        "winter_wheat": (-8, -11),
        "winter_barley": (-5, -7),
        "winter_rye": (-11, -16),
        "winter_rape": (-3,-5)
    }
    
    if crop not in crop_thresholds:
        return "Not Suitable (not a hibernating crop)"
    
    cbtr1, cbtr2 = crop_thresholds[crop]
    
    # Compute temperature amplitude (difference between warmest and coldest month)
    monthly_avg = [np.mean(Ta[i*30:(i+1)*30]) for i in range(12)]  # Approximate monthly means
    tadif0 = max(monthly_avg) - min(monthly_avg)
    
    # Determine the critical breaking temperature (cbtr)
    if tadif0 > 35:
        cbtr = cbtr1
    elif tadif0 > 20:
        cbtr = cbtr1 + (cbtr2 - cbtr1) * (35 - tadif0) / 15
    else:
        cbtr = cbtr2
    
    # Check dormancy period and temperature constraints
    dormancy_days = np.sum((Ta < 5) & (Ta >= cbtr))
    
    dorm_suited = False
    
    if dormancy_days > 200:
        dorm_suited = False
        # return f"Not Suitable (dormancy period too long).\nDormancy: {dormancy_days} days."
        pass
    if np.min(Ta) < cbtr:
        dorm_suited = False
        # return f"Not Suitable (below critical temperature of {cbtr}°C)"
        pass
    if np.mean(Ta) >= 5:
        dorm_suited = True
        # print(f'Suitable,cbtr={cbtr}, dormant days={dormancy_days}')

    return dorm_suited
    
""" END -  NEW - KB Jan 2025 """

    #----------------------------------------------DEVELLOPER'S CODES --------------------------------------------------#

 #--------------------------------------------- Functions for Getting the Intermediate Values of Module II for Validation  ---------------------------------------------------------------#
    # def simulationcropcycleintermediates(self, i:int, j:int, ccdi:int, ccdr:int, start_doy:int =1, end_doy:int= 365, step_doy:int = 1):

    #     ccdi2 = ccdi -1
    #     ccdr2 = ccdr -1


    #     if InitialSuitabilityCheck(self.set_mask, self.im_mask[i,j], self.nodata_val, 
    #                             self.set_Permafrost_screening, self.permafrost_class[i,j], self.set_tclimate_screening,
    #                             self.t_climate[i,j], self.no_t_climate):

    #         raise Exception('Initial Suitability not passed')
        
    #     climate_data = DuplicateOneYearClimateData(self.minT_daily[i,j,:], self.maxT_daily[i,j,:], self.meanT_daily[i,j,:], 
    #                                                         self.shortRad_daily[i,j,:], self.wind2m_daily[i,j,:], self.totalPrec_daily[i,j,:], 
    #                                                         self.rel_humidity_daily[i,j,:], self.pet_daily[i,j,:])


    #     cycle_len_check_data_irr = getCycleLengthCheckingData(self.LGPT5[i,j], self.LGPT10[i,j], self.LGP[i,j],
    #                                                     self.min_cycle_len, 'I', self.perennial, self.min_temp, 
    #                                                     self.max_cycle_len, self.cycle_len)
        
    #     cycle_len_check_data_rain = getCycleLengthCheckingData(self.LGPT5[i,j], self.LGPT10[i,j], self.LGP[i,j],
    #                                                     self.min_cycle_len, 'R', self.perennial, self.min_temp, 
    #                                                     self.max_cycle_len, self.cycle_len)
        
    #     LAI_HI_data = getLAIandHIdata(self.LAi, self.HI, self.aLAI, self.bLAI, self.aHI, self.bHI)

    #     if len(np.array(self.Sa).shape) == 2:
    #         Sa_temp = self.Sa[i, j]
    #     else:
    #         Sa_temp = self.Sa

    #     rain = simulateCropCycleOneLocationIntermediates(start_doy, end_doy, step_doy, self.leap_year, cycle_len_check_data_rain, LAI_HI_data, climate_data,
    #                         self.latitude[i,j], self.elevation[i,j], self.plant_height, self.set_Tsum_screening, self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO,
    #                     self.setCropSpecificRule, self.data, self.legume, self.adaptability,
    #                     self.kc, self.d_per, Sa_temp, self.D1, self.D2, self.crop_group, self.yloss_f_all, self.yloss_f, 'R')
        
    #     irrigated = simulateCropCycleOneLocationIntermediates(start_doy, end_doy, step_doy, self.leap_year, cycle_len_check_data_irr, LAI_HI_data, climate_data,
    #                         self.latitude[i,j], self.elevation[i,j], self.plant_height, self.set_Tsum_screening, self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO,
    #                     self.setCropSpecificRule, self.data, self.legume, self.adaptability,
    #                     self.kc, self.d_per, Sa_temp, self.D1, self.D2, self.crop_group, self.yloss_f_all, self.yloss_f, 'I')

    #     LAI_rain, HI_rain, LAI_irr, HI_rain = 0., 0., 0., 0.
    #     # Effective cycle length determination for perennial crops
    #     if CycleLengthChecking(self.LGPT5[i,j], self.LGPT10[i,j], self.LGP[i,j], self.min_cycle_len, 'R', self.perennial, self.min_temp):
    #         cycle_len_rain = 0
    #     else:
    #         if self.perennial:
    #             cycle_len_rain = DefineEffectiveCycleLength(self.min_temp, self.max_cycle_len, self.LGPT5[i,j], self.LGPT10[i,j], self.LGP[i,j], 'R', self.cycle_len)
    #             LAI_rain, HI_rain = LAI_HI_adjustment(self.LAi, self.HI, self.aLAI, self.bLAI, self.aHI, self.bHI, cycle_len_rain)
    #         else:
    #             # For annual crops, no cycle length adjustment is needed.
    #             cycle_len_rain = self.cycle_len
    #             LAI_rain, HI_rain = self.LAi, self.HI

    #     # Effective cycle length determination for perennial crops
    #     if CycleLengthChecking(self.LGPT5[i,j], self.LGPT10[i,j], self.LGP[i,j], self.min_cycle_len, 'I', self.perennial, self.min_temp):
    #         cycle_len_irr = 0
    #     else:
    #         if self.perennial:
    #             cycle_len_irr = DefineEffectiveCycleLength(self.min_temp, self.max_cycle_len, self.LGPT5[i,j], self.LGPT10[i,j], self.LGP[i,j], 'I', self.cycle_len)
    #             LAI_irr, HI_irr = LAI_HI_adjustment(self.LAi, self.HI, self.aLAI, self.bLAI, self.aHI, self.bHI, cycle_len_irr)
    #         else:
    #             # For annual crops, no cycle length adjustment is needed.
    #             cycle_len_irr = self.cycle_len
    #             LAI_irr, HI_irr = self.LAi, self.HI
        
            
    #     idxr = FindOptimalCropCalendarDOY(rain[0])
    #     idxi = FindOptimalCropCalendarDOY(irrigated[0])

    #     final_yield_rainfed = rain[0][idxr]
    #     crop_calender_rain = idxr + 1
    #     fc1_rain = rain[4][idxr]
    #     fc2_rain = rain[5][idxr]
    #     eta_rain = rain[2][idxr]
    #     wde_rain = rain[1][idxr]


    #     final_yield_irrig = irrigated[0][idxi]
    #     crop_calender_irr = idxi + 1
    #     fc1_irr = irrigated[4][idxi]
    #     fc2_irr = irrigated[5][idxi]
    #     eta_irr = irrigated[2][idxi]
    #     wde_irr = irrigated[1][idxi]


    #     final = {
    #         'Maximum Rainfed Yield': [final_yield_rainfed],
    #         'crop_calendar_rain':[float(crop_calender_rain)],
    #         'fc1_rain': [fc1_rain],
    #         'fc2_rain':[fc2_rain],
    #         'eta_rain':[eta_rain],
    #         'wde_rain':[wde_rain],

    #         'Maximum Irrigated Yield': [final_yield_irrig],
    #         'crop_calendar_irr':[float(crop_calender_irr)],
    #         'fc1_irr': [fc1_irr],
    #         'fc2_irr':[fc2_irr],
    #         'eta_irr':[eta_irr],
    #         'wde_irr':[wde_irr]
    #     }

    #     # Checking the intermediates from Thermal Screening
    #     # Irrigated Conditions
    #     fc1i_irr = 1.
    #     # Thermal Screening for irrigated conditions
    #     if self.perennial:
    #         tsum0i = getTemperatureSum(climate_data[2][ccdi2:ccdi2+365], 0)
    #         tprofilei = getTemperatureProfile(climate_data[2][ccdi2:ccdi2+365])
    #     else:
    #         tsum0i = getTemperatureSum(climate_data[2][ccdi2:ccdi2+cycle_len_irr], 0)
    #         tprofilei = getTemperatureProfile(climate_data[2][ccdi2:ccdi2+cycle_len_irr])
        
    #     tmp_profilei = calculateTemperatureProfileClasses(self.data, tprofilei, self.perennial)
    #     fc1i_irr  = getReductionFactorNumba(self.set_Tsum_screening, self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO, tsum0i,
    #                         self.setCropSpecificRule, tmp_profilei, self.perennial)
        
    #     tsum_fc1i = getReductionFactorNumba(self.set_Tsum_screening, self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO, tsum0i,
    #                 False, tmp_profilei, self.perennial)
    #     crop_specific_fc1i = getReductionFactorNumba(False, self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO, tsum0i,
    #                 self.setCropSpecificRule, tmp_profilei, self.perennial)
        
    #     ts_i = {
    #     'cycle_begin':[ccdi2+1],
    #     'cycle_end':[ccdi2+1+365 if self.perennial else ccdi2+1+int(cycle_len_irr)],
    #     'cycle_len_TSUM':[climate_data[2][ccdi2:ccdi2+365].shape[0] if self.perennial else climate_data[2][ccdi2:ccdi2+cycle_len_irr].shape[0]],
    #     'cycle_len_Tprofile':[climate_data[2][ccdi2:ccdi2+365].shape[0] if self.perennial else climate_data[2][ccdi2:ccdi2+cycle_len_irr].shape[0].shape[0]],
    #     'TSUM0':[tsum0i],
    #     'TProfile':[tprofilei],
    #     'LnS':[self.LnS],
    #     'LsO':[self.LsO],
    #     'LO':[self.LO],
    #     'HO':[self.HO],
    #     'HsO':[self.HsO],
    #     'HnS':[self.HnS],
    #     'fc1_TSUM0':[tsum_fc1i],
    #     'fc1_Tprofile':[crop_specific_fc1i],
    #     'final_fc1_irr':[np.nanmin([tsum_fc1i, crop_specific_fc1i])]
    #     }

    #     # Rainfed Conditions
    #     fc1i_rain = 1.
    #     # Thermal Screening for rainfed conditions
    #     if self.perennial:
    #         tsum0r = getTemperatureSum0(self.meanT_daily[i,j,ccdr2:ccdr2+365])
    #         tprofiler = getTemperatureProfile(self.meanT_daily[i,j,ccdr2:ccdr2+365])
    #     else:
    #         tsum0r= getTemperatureSum0(self.meanT_daily[i,j,ccdr2:ccdr2+cycle_len_rain])
    #         tprofiler = getTemperatureProfile(self.meanT_daily[i,j,ccdr2:ccdr2+cycle_len_rain])
        
    #     tmp_profiler = calculateTemperatureProfileClasses(self.data, tprofiler, self.perennial)
    #     fc1i_rain  = getReductionFactorNumba(self.set_Tsum_screening, self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO, tsum0r,
    #                         self.setCropSpecificRule, tmp_profiler, self.perennial)
        
    #     tsum_fc1r = getReductionFactorNumba(self.set_Tsum_screening, self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO, tsum0r,
    #                 False, tmp_profiler, self.perennial)
    #     crop_specific_fc1r = getReductionFactorNumba(False, self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO, tsum0r,
    #                 self.setCropSpecificRule, tmp_profiler, self.perennial)
        
    #     ts_r = {
    #     'cycle_begin':[ccdr2+1],
    #     'cycle_end':[ccdr2+1+365 if self.perennial else ccdr2+1+int(cycle_len_rain)],
    #     'cycle_len_TSUM':[self.meanT_daily[i,j,ccdr2:ccdr2+365].shape[0] if self.perennial else self.meanT_daily[i,j,ccdr2:ccdr2+cycle_len_rain]],
    #     'cycle_len_Tprofile':[self.meanT_daily[i,j,ccdr2:ccdr2+365].shape[0] if self.perennial else self.meanT_daily[i,j,ccdr2:ccdr2+cycle_len_rain]],
    #     'TSUM0':[tsum0r],
    #     'TProfile':[tprofiler],
    #     'LnS':[self.LnS],
    #     'LsO':[self.LsO],
    #     'LO':[self.LO],
    #     'HO':[self.HO],
    #     'HsO':[self.HsO],
    #     'HnS':[self.HnS],
    #     'fc1_TSUM0':[tsum_fc1r],
    #     'fc1_Tprofile':[crop_specific_fc1r],
    #     'final_fc1_irr':[np.nanmin([tsum_fc1r, crop_specific_fc1r])]
    #     }
        
    #     if LAI_irr <= 0.001 or HI_irr <=0.001:
    #         biomassi = {'Note': 'LAI_irr or HI_irr is less than 0.01. Simulation is not done.'}
    #         wati = {'Note': 'LAI_irr or HI_irr is less than 0.01. Simulation is not done.'}
    #         watr2 = {'Note': 'LAI_irr or HI_irr is less than 0.01. Simulation is not done.'}
    #     elif cycle_len_irr == 0:
    #         biomassi = {'Note': 'Cycle Length not enough. Simulation is not done.'}
    #         wati = {'Note': 'Cycle Length not enough. Simulation is not done.'}
    #         watr2 = {'Note': 'Cycle Length not enough. Simulation is not done.'}
    #     else:
    #         bni = calculateBiomassNumbaIntermediates(ccdi2+1, ccdi2+1+cycle_len_irr, cycle_len_irr, self.latitude[i,j],
    #                      climate_data[3][ccdi2:ccdi2+cycle_len_irr], climate_data[2][ccdi2:ccdi2+cycle_len_irr], climate_data[0][ccdi2:ccdi2+cycle_len_irr], climate_data[1][ccdi2:ccdi2+cycle_len_irr],
    #                     LAI_irr, self.legume, self.adaptability, self.leap_year)
            
    #         cycle_yldi = bni[0] * HI_irr * fc1i_irr

    #         biomassi = {
    #         'adaptability': [self.adaptability],
    #         'legume': [self.legume],
    #         'cycle_start': [ccdi2+1],
    #         'cycle_end': [ccdi2+1+cycle_len_irr],
    #         'LAI': [LAI_irr],
    #         'HI': [HI_irr],
    #         'Ac_mean': [bni[1]],
    #         'Bc_mean': [bni[2]],
    #         'Bo_mean': [bni[3]],
    #         'meanT_mean': [bni[4]],
    #         'dT_mean': [bni[5]],
    #         'Rg':[bni[6]],
    #         'f_day_clouded':[bni[7]],
    #         'pm': [bni[8]],
    #         'ct': [bni[9]],
    #         'growth ratio(l)': [bni[10]],
    #         'bgm': [bni[11]],
    #         'Bn': [bni[12]],
    #         'final irrigated yield': [np.round(cycle_yldi, 0).astype(int)]
    #         }

    #         # Crop Water Requirement for irrigated conditions
    #         cropwati = calculateMoistureLimitedYieldNumbaIntermediates('I', self.kc, self.d_per, cycle_len_irr,climate_data[5][ccdi2:ccdi2+cycle_len_irr], climate_data[7][ccdi2:ccdi2+cycle_len_irr], 
    #                                                                 climate_data[0][ccdi2:ccdi2+cycle_len_irr], climate_data[1][ccdi2:ccdi2+cycle_len_irr], self.plant_height, climate_data[4][ccdi2:ccdi2+cycle_len_irr],
    #                                         self.Sa, self.D1, self.D2, climate_data[2][ccdi2:ccdi2+cycle_len_irr], self.crop_group, self.yloss_f_all, self.yloss_f, self.perennial, np.round(cycle_yldi, 0).astype(int))    
    #         wati = {
    #             'cycle_start': [ccdi2+1],
    #             'cycle_end': [ccdi2+1+cycle_len_irr],
    #             'Original kc_initial': [self.kc[0]],
    #             'Original kc_reprodu':[self.kc[1]],
    #             'Original kc_maturity':[self.kc[2]],
    #             'Adjustedd kc_initial': [cropwati[6][0]],
    #             'Adjusted kc_reprodu':[cropwati[6][1]],
    #             'Adjusted kc_maturity':[cropwati[6][2]],
    #             'Soil Water Holding Capacity (Sa)':[self.Sa],
    #             'Soil Water Depletion Factor Group':[self.crop_group],
    #             'plant height': [self.plant_height],
    #             'kc_all': [self.kc_all],
    #             'y_loss_init':[self.yloss_f[0]],
    #             'y_loss_vege':[self.yloss_f[1]],
    #             'y_loss_repro':[self.yloss_f[2]],
    #             'y_loss_maturity': [self.yloss_f[3]],
    #             'y_loss_all': [self.yloss_f_all],
    #             'potential yield': [np.round(cycle_yldi, 0).astype(int) * fc1i_irr],
    #             'root_depth_start': [self.D1],
    #             'root_depth_end':[self.D2],
    #             'fc2 from water deficit of each growth cycle': [cropwati[4]],
    #             'fc2 for the entire cycle period':[cropwati[5]],
    #             'water_lim_yield': [cropwati[3]],
    #             'water deficit(wde)':[cropwati[0]],
    #             'total irrigation requirement (eta)':[cropwati[2]]
    #         }
    #         wati2 = {
    #             'DOY':np.arange(ccdi2+1, ccdi2+1+cycle_len_irr),
    #             'Sb':cropwati[8],
    #             'Wx':cropwati[9],
    #             'Wb':cropwati[10],
    #             'ETa':cropwati[11],
    #             'ETm':cropwati[12],
    #             'kc_daily':cropwati[13],
    #             'pc_daily':cropwati[14]
    #         }


    #     # Intermediate results for rainfed conditions
    #     if LAI_rain <= 0.001 or HI_rain <=0.001:
    #         biomassr = {'Note': 'LAI_rain or HI_rain is less than 0.001. Simulation is not done.'}
    #         watr = {'Note': 'LAI_rain or HI_rain is less than 0.001. Simulation is not done.'}
    #         watr2 = {'Note': 'LAI_rain or HI_rain is less than 0.001. Simulation is not done.'}
    #     elif cycle_len_rain == 0:
    #         biomassr = {'Note': 'Cycle Length not enough. Simulation is not done.'}
    #         watr = {'Note': 'Cycle Length not enough. Simulation is not done.'}
    #         watr2 = {'Note': 'Cycle Length not enough. Simulation is not done.'}
    #     else:
    #         bnr = calculateBiomassNumbaIntermediates(ccdr2+1, ccdr2+1+cycle_len_rain, cycle_len_rain, self.latitude[i,j],
    #                      climate_data[3][ccdr2:ccdr2+cycle_len_rain], climate_data[2][ccdr2:ccdr2+cycle_len_rain], climate_data[0][ccdr2:ccdr2+cycle_len_rain], climate_data[1][ccdr2:ccdr2+cycle_len_rain],
    #                     LAI_rain, self.legume, self.adaptability, self.leap_year)
            
    #         cycle_yldr = bnr[0] * HI_rain * fc1i_rain

    #         biomassr = {
    #         'adaptability': [self.adaptability],
    #         'legume': [self.legume],
    #         'cycle_start': [ccdr2+1],
    #         'cycle_end': [ccdr2+1+cycle_len_rain],
    #         'LAI': [LAI_rain],
    #         'HI': [HI_rain],
    #         'Ac_mean': [bnr[1]],
    #         'Bc_mean': [bnr[2]],
    #         'Bo_mean': [bnr[3]],
    #         'meanT_mean': [bnr[4]],
    #         'dT_mean': [bnr[5]],
    #         'Rg':[bnr[6]],
    #         'f_day_clouded':[bnr[7]],
    #         'pm': [bnr[8]],
    #         'ct': [bnr[9]],
    #         'growth ratio(l)': [bnr[10]],
    #         'bgm': [bnr[11]],
    #         'Bn': [bnr[12]],
    #         'Rainfed yield': [np.round(cycle_yldr, 0).astype(int)]
    #         }

    #         # Crop Water Requirement for rainfed conditions
    #         cropwatr = calculateMoistureLimitedYieldNumbaIntermediates('R', self.kc, self.d_per, cycle_len_rain,climate_data[5][ccdr2:ccdr2+cycle_len_rain], climate_data[7][ccdr2:ccdr2+cycle_len_rain], 
    #                                                                 climate_data[0][ccdr2:ccdr2+cycle_len_rain], climate_data[1][ccdr2:ccdr2+cycle_len_rain], self.plant_height, climate_data[4][ccdr2:ccdr2+cycle_len_rain],
    #                                         self.Sa, self.D1, self.D2, climate_data[2][ccdr2:ccdr2+cycle_len_rain], self.crop_group, self.yloss_f_all, self.yloss_f, self.perennial, np.round(cycle_yldr, 0).astype(int))    
    #         watr = {
    #             'cycle_start': [ccdr2+1],
    #             'cycle_end': [ccdr2+1+cycle_len_rain],
    #             'Original kc_initial': [self.kc[0]],
    #             'Original kc_reprodu':[self.kc[1]],
    #             'Original kc_maturity':[self.kc[2]],
    #             'Adjustedd kc_initial': [cropwatr[6][0]],
    #             'Adjusted kc_reprodu':[cropwatr[6][1]],
    #             'Adjusted kc_maturity':[cropwatr[6][2]],
    #             'Soil Water Holding Capacity (Sa)':[self.Sa],
    #             'Soil Water Depletion Factor Group':[self.crop_group],
    #             'plant height': [self.plant_height],
    #             'kc_all': [self.kc_all],
    #             'y_loss_init':[self.yloss_f[0]],
    #             'y_loss_vege':[self.yloss_f[1]],
    #             'y_loss_repro':[self.yloss_f[2]],
    #             'y_loss_maturity': [self.yloss_f[3]],
    #             'y_loss_all': [self.yloss_f_all],
    #             'potential yield': [np.round(cycle_yldr, 0).astype(int) * fc1i_rain],
    #             'root_depth_start': [self.D1],
    #             'root_depth_end':[self.D2],
    #             'fc2 from water deficit of each growth cycle': [cropwatr[4]],
    #             'fc2 for the entire cycle period':[cropwatr[5]],
    #             'water_lim_yield': [cropwatr[3]],
    #             'water deficit(wde)':[cropwatr[0]],
    #             'total irrigation requirement (eta)':[cropwatr[2]]
    #         }
    #         watr2 = {
    #             'DOY':np.arange(ccdr2+1, ccdr2+1+cycle_len_rain),
    #             'Sb':cropwatr[8],
    #             'Wx':cropwatr[9],
    #             'Wb':cropwatr[10],
    #             'ETa':cropwatr[11],
    #             'ETm':cropwatr[12],
    #             'kc_daily':cropwatr[13],
    #             'pc_daily': cropwatr[14]
    #         }
    #     cycle = {
    #         'Cycles': np.arange(1,367) if self.leap_year else np.arange(1,366),
    #         'Rainfed Yield ': rain[0],
    #         'fc1_rain': rain[4],
    #         'fc2_rain': rain[5],
    #         'eta_rain': rain[2],
    #         'wde_rain': rain[1],

    #         'Irrigated Yield ': irrigated[0],
    #         'fc1_irr': irrigated[4],
    #         'fc2_irr': irrigated[5],
    #         'eta_irr': irrigated[2] ,
    #         'wde_irr': irrigated[1],
    #     }

    #     general = {
    #     'row': [i],
    #     'col': [j],
    #     'mask': [self.im_mask[i, j]],
    #     'permafrost': [self.permafrost_class[i, j]],
    #     'TClimate': [self.t_climate[i, j]],
    #     'perennial_flag': [self.perennial],
    #     'LGPT5':[self.LGPT5[i, j]],
    #     'LGPT10':[self.LGPT10[i, j]],
    #     'LGP':[self.LGP[i, j]],
    #     'elevation':[self.elevation[i, j]],
    #     'Latitude': [self.latitude[i, j]],
    #     'Minimum cycle length':[self.min_cycle_len],
    #     'Maximum cycle length': [self.max_cycle_len],
    #     'aLAI': [self.aLAI],
    #     'bLAI':[self.bLAI],
    #     'aHI':[self.aHI],
    #     'bHI':[self.bHI],
    #     'Reference_cycle_len':[self.cycle_len],
    #     'Effective cycle length rainfed':[cycle_len_rain],
    #     'Effective cycle length irrigated':[cycle_len_irr],
    #     'Original LAI rain':[self.LAi],
    #     'Original HI rain':[self.HI],
    #     'Original LAI irr':[self.LAi],
    #     'Original HI irr':[self.HI],
    #     'Adjusted LAI rain':[LAI_rain],
    #     'Adjusted HI rain':[HI_rain],
    #     'Adjusted LAI irr':[LAI_irr],
    #     'Adjusted HI irr':[HI_irr],
    #     }

    #     climate = {
    #         'min_temp(DegC)':self.minT_daily[i,j,:],
    #         'max_temp(DegC)':self.maxT_daily[i,j,:],
    #         'mean_temp(DegC)':self.meanT_daily[i,j,:],
    #         'shortrad(Wm-2)':self.shortRad_daily[i,j,:],
    #         'shortrad(MJ/m2/day)': (self.shortRad_daily[i,j,:] * 3600 * 24)/1000000,
    #         'shortrad(calcm-2day-1)':self.shortRad_daily[i,j,:] * 2.06362854686156,
    #         'windspeed(ms-1)':self.wind2m_daily[i,j,:],
    #         'precipitation(mmday-1)':self.totalPrec_daily[i,j,:],
    #         'rel_humid(decimal)':self.rel_humidity_daily[i,j,:],
    #         'ETo (mmday-1)':self.pet_daily[i,j,:]
    #         }
    #     print('\nSimulations Completed !')
    #     return [general ,climate, cycle, final ,biomassi, wati, wati2, biomassr, watr, watr2, ts_i, ts_r]

# def CropCycleLoopingIntermediates(start_doy:int, end_doy:int, step_doy:int, climate_data, min_T_threshold, perennial_flag:bool,
#                      cycle_len:int, set_TSUM_screening:bool, LnS, LsO, LO, HnS, HsO, HO, set_CropSpecificRule:bool, data, 
#                      lat:float, lai:float, hi:float, legume:int, adaptability:int, plant_height:float,
#                      kc, d_per, Sa, D1:float, D2:float, crop_group:int, yloss_f_all:float, yloss_f, irr_or_rain:str, leap_year:bool):

#     """Simulating the cycles to obtain list of each cycle's yield, fc1, fc2, eta, wde."""
#     # Only call the climate data once all initial flag checks are False
#     min_T = climate_data[0]
#     max_T = climate_data[1]
#     mean_T = climate_data[2]
#     shrt_rd = climate_data[3]
#     wind_sp = climate_data[4]
#     pr = climate_data[5]
#     rel_hum = climate_data[6]
#     eto = climate_data[7]

#     # important variable returning
#     yd_arr = np.empty(0, dtype= float)
#     wde_arr = np.empty(0, dtype= float)
#     eta_arr= np.empty(0, dtype= float)
#     fc1_arr= np.empty(0, dtype= float)
#     fc2_arr= np.empty(0, dtype= float)

#     for i_cycle in range(start_doy-1, end_doy, step_doy):

#         cycle_yld:float = 0.
#         cycle_wde: float = 0.
#         cycle_eta:float = 0.
#         cycle_fc1: float = 0.
#         cycle_fc2: float = 0.

#         """Check if the first day of a cycle meets minimum temperature requirement. If not, all outputs will be zero.
#             And iterates to next cycle."""
#         if mean_T[i_cycle]< min_T_threshold:
#             yd_arr = np.append(yd_arr, 0.)
#             wde_arr = np.append(wde_arr, 0.)
#             eta_arr = np.append(eta_arr, 0.)
#             fc1_arr = np.append(fc1_arr, 0.)
#             fc2_arr = np.append(fc2_arr, 0.)
#             continue
        
#         cycle_fc1 = 1.
#         # Thermal Screening 
#         if perennial_flag:
#             tsum0 = getTemperatureSum(mean_T[i_cycle:i_cycle+365], 0)
#         else:
#             tsum0 = getTemperatureSum(mean_T[i_cycle:i_cycle+cycle_len], 0)
        
#         tmp_profile = calculateTemperatureProfileClasses(data, mean_T[i_cycle:i_cycle+cycle_len], cycle_len)
#         cycle_fc1 = getReductionFactorNumba(set_TSUM_screening, LnS, LsO, LO, HnS, HsO, HO, tsum0,
#                             set_CropSpecificRule, tmp_profile, hibernation_flag, vern_factor)
        
#         if cycle_fc1 <=0.001:
#             cycle_fc1, cycle_fc2, cycle_yld = 0., 0., 0.
#             yd_arr = np.append(yd_arr, 0.)
#             wde_arr = np.append(wde_arr, 0.)
#             eta_arr = np.append(eta_arr, 0.)
#             fc1_arr = np.append(fc1_arr, 0.)
#             fc2_arr = np.append(fc2_arr, 0.)
#             continue
#         else:
#             # Biomass Calculation
#             bn = calculateBiomassNumba(i_cycle+1, i_cycle+1+cycle_len, cycle_len, lat, shrt_rd[i_cycle:i_cycle+cycle_len],
#                                          mean_T[i_cycle:i_cycle+cycle_len], min_T[i_cycle:i_cycle+cycle_len],max_T[i_cycle:i_cycle+cycle_len],
#                                          lai, legume, adaptability, leap_year)
#             cycle_yld = bn * hi * cycle_fc1

#             #Crop Water Requirement
#             cycle_wde, cycle_fc2, cycle_eta, cycle_yld = calculateMoistureLimitedYieldNumba(irr_or_rain, kc, d_per, cycle_len, pr[i_cycle:i_cycle+cycle_len], eto[i_cycle:i_cycle+cycle_len],
#                                                                                             min_T[i_cycle:i_cycle+cycle_len], max_T[i_cycle:i_cycle+cycle_len], plant_height, wind_sp[i_cycle:i_cycle+cycle_len],
#                                                                                             Sa, D1, D2, mean_T[i_cycle:i_cycle+cycle_len], crop_group, yloss_f_all, yloss_f, perennial_flag, cycle_yld)

#             # Appending to the list
#             yd_arr = np.append(yd_arr, cycle_yld)
#             wde_arr = np.append(wde_arr, cycle_wde)
#             eta_arr = np.append(eta_arr, cycle_eta)
#             fc1_arr = np.append(fc1_arr, cycle_fc1)
#             fc2_arr = np.append(fc2_arr, cycle_fc2)
        
#     return yd_arr, wde_arr, eta_arr, eta_arr, fc1_arr, fc2_arr

# def simulateCropCycleOneLocationIntermediates(start_doy:int, end_doy:int, step_doy:int, leap_year:bool, cycle_len_check_data, LAI_HI_data, climate_data,
#                                     lat:float, elev:float, plant_height:float, set_TSUM_screening:bool, LnS:int, LsO:int, LO:int, HnS:int, HsO:int, HO:int,
#                                 set_CropSpecificRule:bool, data, legume:int, adaptability:int,
#                                 kc, d_per, Sa, D1:float, D2:float, crop_group:int, yloss_f_all:float, yloss_f, irr_or_rain:str):
    
#     """NESTED FUNCTION: All simulation procedures are done for a single pixel location"""
#     final_yld:float = 0.
#     ccd: int = 0
#     wde: float = 0.
#     eta:float = 0.
#     fc1: float = 0.
#     fc2: float = 0.
#     cycle_len:float = 0

#     lgpt5, lgpt10, lgp, min_cycle_len, irr_or_rain, perennial_flg, min_temp_threshold, max_cycle_len, ref_cycle_len = cycle_len_check_data

#     lai, hi, alai, blai, ahi, bhi = LAI_HI_data

#     LAi, HI, cycle_len = 0., 0, 0

#     # Effective cycle length determination for perennial crops
#     if perennial_flg:
#         cycle_len = DefineEffectiveCycleLength(min_temp_threshold, max_cycle_len,lgpt5, lgpt10, lgp, irr_or_rain, ref_cycle_len)
#         LAi, HI = LAI_HI_adjustment(lai, hi, alai, blai, ahi, bhi, cycle_len)
#     # For     
#     else:
#         # For annual crops, no cycle length adjustment is needed.
#         cycle_len = ref_cycle_len
#         LAi, HI = lai, hi
    

#     if LAi <= 0.001 or HI <= 0.001 or cycle_len<0:
#         return final_yld, wde, eta, fc1, fc2, ccd
#     else:
#         val = CropCycleLoopingIntermediates(start_doy, end_doy, step_doy, climate_data, min_temp_threshold, perennial_flg,
#                         cycle_len, set_TSUM_screening, LnS, LsO, LO, HnS, HsO, HO, set_CropSpecificRule, data, 
#                         lat, LAi, HI, legume, adaptability, plant_height,
#                         kc, d_per, Sa, D1, D2, crop_group, yloss_f_all, yloss_f, irr_or_rain, leap_year)
    
#         final_yld, wde, eta, fc1, fc2, ccd = val[0], val[1], val[2], val[3], val[4], val[5] 

#         return final_yld, wde, eta, fc1, fc2, ccd