"""
PyAEZ version 2.1.0 (June 2023)
This CropSimulation Class simulates all the possible crop cycles to find 
the best crop cycle that produces maximum yield for a particular grid
2020: N. Lakmal Deshapriya
2022/2023: Swun Wunna Htet
"""

import numpy as np
import pandas as pd
try:
    import gdal
except:
    from osgeo import gdal

from pyaez import UtilitiesCalc,BioMassCalc,ETOCalc,CropWatCalc,ThermalScreening

class CropSimulation(object):

    def __init__(self):
        """Initiate a Class instance
        """        
        self.set_mask = False
        self.set_tclimate_screening = False
        self.set_lgpt_screening = False
        self.set_Tsum_screening = False
        self.set_Permafrost_screening = False  
        self.set_adjustment = False 
        self.setTypeBConstraint = True

    def setMonthlyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity):
        """Read and load the MONTHLY climate data into the Class
        Args:
            min_temp (3D NumPy): Monthly minimum temperature [Celcius]
            max_temp (3D NumPy): Monthly maximum temperature [Celcius]
            precipitation (3D NumPy): Monthly total precipitation [mm/day]
            short_rad (3D NumPy): Monthly solar radiation [W/m2]
            wind_speed (3D NumPy): Monthly windspeed at 2m altitude [m/s]
            rel_humidity (3D NumPy): Monthly relative humidity [percentage decimal, 0-1]
        """
        rel_humidity[rel_humidity > 0.99] = 0.99
        rel_humidity[rel_humidity < 0.05] = 0.05
        short_rad[short_rad < 0] = 0
        self.minT_monthly = min_temp
        self.maxT_monthly = max_temp
        self.totalPrec_monthly = precipitation
        self.shortRad_monthly = short_rad
        self.wind2m_monthly = wind_speed
        self.rel_humidity_monthly = rel_humidity
        self.set_monthly = True

    def setDailyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity):
        """Load DAILY climate data into the Class

        Args:
            min_temp (3D NumPy): Daily minimum temperature [Celcius]
            max_temp (3D NumPy): Daily maximum temperature [Celcius]
            precipitation (3D NumPy): Daily total precipitation [mm/day]
            short_rad (3D NumPy): Daily solar radiation [W/m2]
            wind_speed (3D NumPy): Daily windspeed at 2m altitude [m/s]
            rel_humidity (3D NumPy): Daily relative humidity [percentage decimal, 0-1]
        """
        rel_humidity[rel_humidity > 0.99] = 0.99
        rel_humidity[rel_humidity < 0.05] = 0.05
        short_rad[short_rad < 0] = 0
        self.minT_daily = min_temp
        self.maxT_daily = max_temp
        self.totalPrec_daily = precipitation
        self.shortRad_daily = short_rad
        self.wind2m_daily = wind_speed
        self.rel_humidity_daily = rel_humidity
        self.set_monthly = False

    def setLocationTerrainData(self, lat_min, lat_max, elevation):
        """Load geographical extents and elevation data in to the Class, 
           and create a latitude map

        Args:
            lat_min (float): the minimum latitude of the AOI in decimal degrees
            lat_max (float): the maximum latitude of the AOI in decimal degrees
            elevation (2D NumPy): elevation map in metres
        """
        self.elevation = elevation
        self.im_height = elevation.shape[0]
        self.im_width = elevation.shape[1]
        self.latitude_map = UtilitiesCalc.UtilitiesCalc().generateLatitudeMap(lat_min, lat_max, self.im_height, self.im_width)
    
    # For this function, we need to explain how to set up excel sheet in the User Guide (Important)
    def readCropandCropCycleParameters(self, file_path, crop_name):
        """
        Mandatory function to import the excel sheet of crop-specific parameters,
        crop water requirements, management info, perennial adjustment parameters,
        and TSUM screening thresholds.

        Parameters
        ----------
        file_path : String.
            The file path of the external excel sheet in xlsx.
        crop_name : String.
            Unique name of crop for crop simulation.

        Returns
        -------
        None.

        """

        self.crop_name = crop_name
        df = pd.read_excel(file_path)

        crop_df_index = df.index[df['Crop_name'] == crop_name].tolist()[0]
        crop_df = df.loc[df['Crop_name'] == crop_name]

        self.setCropParameters(LAI=crop_df['LAI'][crop_df_index], HI=crop_df['HI'][crop_df_index], legume=crop_df['legume'][crop_df_index], adaptability=int(crop_df['adaptability'][crop_df_index]), cycle_len=int(crop_df['cycle_len'][crop_df_index]), D1=crop_df['D1']
                               [crop_df_index], D2=crop_df['D2'][crop_df_index], min_temp=crop_df['min_temp'][crop_df_index], aLAI=crop_df['aLAI'][crop_df_index], bLAI=crop_df['bLAI'][crop_df_index], aHI=crop_df['aHI'][crop_df_index], bHI=crop_df['bHI'][crop_df_index])
        self.setCropCycleParameters(stage_per=[crop_df['stage_per_1'][crop_df_index], crop_df['stage_per_2'][crop_df_index], crop_df['stage_per_3'][crop_df_index], crop_df['stage_per_4'][crop_df_index]], kc=[crop_df['kc_0'][crop_df_index], crop_df['kc_1'][crop_df_index], crop_df['kc_2']
                                    [crop_df_index]], kc_all=crop_df['kc_all'][crop_df_index], yloss_f=[crop_df['yloss_f0'][crop_df_index], crop_df['yloss_f1'][crop_df_index], crop_df['yloss_f2'][crop_df_index], crop_df['yloss_f3'][crop_df_index]], yloss_f_all=crop_df['yloss_f_all'][crop_df_index])

        # perennial = 1, annual = 0
        if crop_df['annual/perennial flag'][crop_df_index] == 1:
            self.perennial = True
        else:
            self.perennial = False

        # If users provide all TSUM thresholds, TSUM screening
        if np.all([crop_df['LnS'][crop_df_index] != np.nan, crop_df['LsO'][crop_df_index] != np.nan, crop_df['LO'][crop_df_index] != np.nan, crop_df['HnS'][crop_df_index] != np.nan, crop_df['HsO'][crop_df_index] != np.nan, crop_df['HO'][crop_df_index] != np.nan]):
            self.setTSumScreening(LnS=crop_df['LnS'][crop_df_index], LsO=crop_df['LsO'][crop_df_index], LO=crop_df['LO'][crop_df_index],
                                  HnS=crop_df['HnS'][crop_df_index], HsO=crop_df['HsO'][crop_df_index], HO=crop_df['HO'][crop_df_index])

        # releasing memory
        del (crop_df_index, crop_df)


    def setSoilWaterParameters(self, Sa, pc):
        """This function allow user to set up the parameters related to the soil water storage.

        Args:
            Sa (float or 2D numpy): Available  soil moisture holding capacity
            pc (float): Soil water depletion fraction below which ETa<ETo
        """        
        self.Sa = Sa  # available soil moisture holding capacity (mm/m) , assumption
        self.pc = pc  # soil water depletion fraction below which ETa < ETo (from literature)

    
    
    """Supporting functions nested within the mandatory functions"""

    def setCropParameters(self, LAI, HI, legume, adaptability, cycle_len, D1, D2, min_temp, aLAI, bLAI, aHI, bHI):
        """This function allows users to set up the main crop parameters necessary for PyAEZ.

        Args:
            LAI (float): Leaf Area Index
            HI (float): Harvest Index
            legume (binary, yes=1, no=0): Is the crop legume?
            adaptability (int): Crop adaptability clases (1-4)
            cycle_len (int): Length of crop cycle
            D1 (float): Rooting depth at the beginning of the crop cycle [m]
            D2 (float): Rooting depth after crop maturity [m]
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

    def setCropCycleParameters(self, stage_per, kc, kc_all, yloss_f, yloss_f_all):
        self.d_per = stage_per  # Percentage for D1, D2, D3, D4 stages
        self.kc = kc  # 3 crop water requirements for initial, reproductive, the end of the maturation stages
        self.kc_all = kc_all  # crop water requirements for entire growth cycle
        self.yloss_f = yloss_f  # yield loss for D1, D2, D3, D4
        self.yloss_f_all = yloss_f_all  # yield loss for entire growth cycle

    
    
    '''Additional optional functions'''

    # set mask of study area, this is optional
    def setStudyAreaMask(self, admin_mask, no_data_value):
        """Set clipping mask of the area of interest (optional)

        Args:
            admin_mask (2D NumPy/Binary): mask to extract only region of interest
            no_data_value (int): pixels with this value will be omitted during PyAEZ calculations
        """
        self.im_mask = admin_mask
        self.nodata_val = no_data_value
        self.set_mask = True

    def ImportLGPandLGPTforPerennial(self, lgp, lgpt5, lgpt10):
        """
        Mandatory step of input data required for perennial crop simulation.
        This function is run before the actual crop simulation.

        Parameters
        ----------
        lgp : 2-D numpy array
            Length of Growing Period.
        lgpt5 : 2-D numpy array
            Temperature Growing Period at 5℃ threshold.
        lgpt10 : 2-D numpy array
            Temperature Growing Period at 10℃ threshold.

        Returns
        -------
        None.

        """
        self.LGP = lgp
        self.LGPT5 = lgpt5
        self.LGPT10 = lgpt10

    def adjustForPerennialCrop(self,  cycle_len, aLAI, bLAI, aHI, bHI, rain_or_irr):
        """If a perennial crop is introduced, PyAEZ will perform adjustment 
        on the Leaf Area Index (LAI) and the Harvest Index (HI) based 
        on the effective growing period.

        Args:
            aLAI (int): alpha coefficient for LAI
            bLAI (int): beta coefficient for LAI
            aHI (int): alpha coefficient for HI
            bHI (int): beta coefficient for HI
        """        
        if rain_or_irr == 'rain':
            # leaf area index adjustment for perennial crops
            self.LAi_rain = self.LAi * ((cycle_len-aLAI)/bLAI)
            # harvest index adjustment for perennial crops
            self.HI_rain = self.HI * ((cycle_len-aHI)/bHI)

        if rain_or_irr == 'irr':
            # leaf area index adjustment for perennial crops
            self.LAi_irr = self.LAi * ((cycle_len-aLAI)/bLAI)
            # harvest index adjustment for perennial crops
            self.HI_irr = self.HI * ((cycle_len-aHI)/bHI)

    """ Thermal Screening functions (Optional)"""

    def setThermalClimateScreening(self, t_climate, no_t_climate):
        """
        The thermal screening function omit out user-specified thermal climate classes
        not suitable for a particular crop for crop simulation. Using this optional 
        function will activate application of thermal climate screening in crop cycle simulation.
    

        Parameters
        ----------
        t_climate : 2-D numpy array
            Thermal Climate.
        no_t_climate : list
            A list of thermal climate classes not suitable for crop simulation.

        Returns
        -------
        None.

        """
        self.t_climate = t_climate
        self.no_t_climate = no_t_climate  # list of unsuitable thermal climate

        self.set_tclimate_screening = True

    # set suitability screening, this is also optional
    def setLGPTScreening(self, no_lgpt, optm_lgpt):
        """Set screening parameters for thermal growing period (LGPt)

        Args:
            no_lgpt (3-item list): 3 'not suitable' LGPt conditions
            optm_lgpt (3-item list): 3 'optimum' LGPt conditions
        """        
        self.no_lgpt = no_lgpt
        self.optm_lgpt = optm_lgpt

        self.set_lgpt_screening = True

    def setTSumScreening(self, LnS, LsO, LO, HnS, HsO, HO):
        """
        This thermal screening corresponds to Type A constraint (TSUM Screeing) of GAEZ which
        uses six TSUM thresholds for optimal, sub-optimal and not suitable conditions. Using 
        this optional function will activate application of TSUM screening in crop cycle simulation.
        

        Parameters
        ----------
        LnS : Integer
            Lower boundary of not-suitable accumulated heat unit range.
        LsO : Integer
            Lower boundary of sub-optimal accumulated heat unit range.
        LO : Integer
            Lower boundary of optimal accumulated heat unit range.
        HnS : Integer
            Upper boundary of not-suitable accumulated heat range.
        HsO : Integer
            Upper boundary of sub-optimal accumulated heat range.
        HO : Integer
            Upper boundary of not-suitable accumulated heat range.

        Returns
        -------
        None.

        """
        self.LnS = LnS  # Lower boundary/ not suitable
        self.LsO = LsO  # Lower boundary/ sub optimal
        self.LO = LO  # Lower boundary / optimal
        self.HnS = HnS  # Upper boundary/ not suitable
        self.HsO = HsO  # Upper boundary / sub-optimal
        self.HO = HO  # Upper boundary / optimal
        self.set_Tsum_screening = True

    def setPermafrostScreening(self, permafrost_class):

        self.permafrost_class = permafrost_class  # permafrost class 2D numpy array
        self.set_Permafrost_screening = True

    def setupTypeBConstraint(self, file_path, crop_name):
        """
        Optional function initiates the type B constraint (Temperature Profile 
        Constraint) on the existing crop based on user-specified constraint rules.

        Parameters
        ----------
        file_path : xlsx
            The file path of excel sheet where the Type B constraint rules are provided.
        crop_name : String
            Unique name of crop to consider. The name must be the same provided in excel sheet.


        Returns
        -------
        None.

        """

        data = pd.read_excel(file_path)
        self.crop_name = crop_name

        self.data = data.loc[data['Crop'] == self.crop_name]

        self.setTypeBConstraint = True

        # releasing data
        del (data)

    """ The main functions of MODULE II: Crop Simulation"""

    def simulateCropCycle(self, start_doy=1, end_doy=365, step_doy=1, leap_year=False):
        """Running the crop cycle calculation/simulation

        Args:
            start_doy (int, optional): Starting Julian day for simulating period. Defaults to 1.
            end_doy (int, optional): Ending Julian day for simulating period. Defaults to 365.
            step_doy (int, optional): Spacing (in days) between 2 adjacent crop simulations. Defaults to 1.
            leap_year (bool, optional): whether or not the simulating year is a leap year. Defaults to False.

        """        

        # just a counter to keep track of progress
        count_pixel_completed = 0
        total = self.im_height * self.im_width

        # this stores final result
        self.final_yield_rainfed = np.zeros((self.im_height, self.im_width))
        self.final_yield_irrig = np.zeros((self.im_height, self.im_width))
        self.crop_calender_irr = np.zeros(
            (self.im_height, self.im_width), dtype=int)
        self.crop_calender_rain = np.zeros(
            (self.im_height, self.im_width), dtype=int)
        # 6. New Modification
        self.fc2 = np.zeros((self.im_height, self.im_width))

        if self.perennial:
            self.fc1_rain = np.zeros((self.im_height, self.im_width))
            self.fc1_irr = np.zeros((self.im_height, self.im_width))
        else:
            self.fc1 = np.zeros((self.im_height, self.im_width))

        for i_row in range(self.im_height):

            for i_col in range(self.im_width):

                # print('\r{} {} '.format(i_row, i_col), end = '')

                # check current location (pixel) is outside of study area or not. if it's outside of study area goes to next location (pixel)
                # Those unsuitable
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        count_pixel_completed = count_pixel_completed +1
                        print('\rDone %: ' + str(round(count_pixel_completed /
                        total*100, 2)), end='\r')
                        continue

                # 2. Permafrost screening
                if self.set_Permafrost_screening:
                    if np.logical_or(self.permafrost_class[i_row, i_col] == 1, self.permafrost_class[i_row, i_col] == 2):
                        count_pixel_completed = count_pixel_completed +1
                        print('\rDone %: ' + str(round(count_pixel_completed /
                        total*100, 2)), end='\r')
                        continue

                # Thermal Climate Screening
                if self.set_tclimate_screening:
                    if self.t_climate[i_row, i_col] in self.no_t_climate:
                        count_pixel_completed = count_pixel_completed +1
                        print('\rDone %: ' + str(round(count_pixel_completed /
                        total*100, 2)), end='\r')
                        continue
                
                count_pixel_completed = count_pixel_completed + 1
                # this allows handing leap and non-leap year differently. This is only relevant for monthly data because this value will be used in interpolations.
                # In case of daily data, length of vector will be taken as number of days in  a year.
                if leap_year:
                    days_in_year = 366
                else:
                    days_in_year = 365

                # extract climate data for particular location. And if climate data are monthly data, they are interpolated as daily data
                if self.set_monthly:
                    obj_utilities = UtilitiesCalc.UtilitiesCalc()

                    minT_daily_point = obj_utilities.interpMonthlyToDaily(
                        self.minT_monthly[i_row, i_col, :], 1, days_in_year)
                    maxT_daily_point = obj_utilities.interpMonthlyToDaily(
                        self.maxT_monthly[i_row, i_col, :], 1, days_in_year)
                    shortRad_daily_point = obj_utilities.interpMonthlyToDaily(
                        self.shortRad_monthly[i_row, i_col, :],  1, days_in_year, no_minus_values=True)
                    wind2m_daily_point = obj_utilities.interpMonthlyToDaily(
                        self.wind2m_monthly[i_row, i_col, :],  1, days_in_year, no_minus_values=True)
                    totalPrec_daily_point = obj_utilities.interpMonthlyToDaily(
                        self.totalPrec_monthly[i_row, i_col, :],  1, days_in_year, no_minus_values=True)
                    rel_humidity_daily_point = obj_utilities.interpMonthlyToDaily(
                        self.rel_humidity_monthly[i_row, i_col, :],  1, days_in_year, no_minus_values=True)
                else:
                    minT_daily_point = self.minT_daily[i_row, i_col, :]
                    maxT_daily_point = self.maxT_daily[i_row, i_col, :]
                    shortRad_daily_point = self.shortRad_daily[i_row, i_col, :]
                    wind2m_daily_point = self.wind2m_daily[i_row, i_col, :]
                    totalPrec_daily_point = self.totalPrec_daily[i_row, i_col, :]
                    rel_humidity_daily_point = self.rel_humidity_daily[i_row, i_col, :]

                # calculate ETO for full year for particular location (pixel) 7#
                obj_eto = ETOCalc.ETOCalc(
                    1, minT_daily_point.shape[0], self.latitude_map[i_row, i_col], self.elevation[i_row, i_col])

                # 7. New Modification
                # shortRad_dailyy_point_MJm2day = (shortRad_daily_point*3600*24)/1000000 # convert w/m2 to MJ/m2/day

                # 7. Minor change for validation purposes: shortRad_daily_point is replaced in shortRad_dailyy_point_MJm2day. (Sunshine hour data for KB Etocalc)
                obj_eto.setClimateData(minT_daily_point, maxT_daily_point,
                                       wind2m_daily_point, shortRad_daily_point, rel_humidity_daily_point)
                pet_daily_point = obj_eto.calculateETO()

                # list that stores yield estimations and thermal screening factors of all cycles per particular location (pixel)
                yield_of_all_crop_cycles_rainfed = []
                yield_of_all_crop_cycles_irrig = []

                if self.perennial:
                    fc1_rain_lst = []
                    fc1_irr_lst = []
                else:
                    fc_lst = []
                    fc2_lst = []

                """Adjustment of cycle length, LAI and HI for Perennials"""
                if self.perennial:

                    self.set_adjustment = True

                    """ Adjustment for rainfed conditions"""

                    if self.LGP[i_row, i_col] < self.cycle_len:

                        # LGP duration will be efficient cycle length for rainfed conditions
                        # Later, we use LGP length to adjust for LAI and HI for rainfed conditions
                        self.cycle_len_rain = int(self.LGP[i_row, i_col])
                        self.adjustForPerennialCrop(
                            self.cycle_len_rain, aLAI=self.aLAI, bLAI=self.bLAI, aHI=self.aHI, bHI=self.bHI, rain_or_irr='rain')
                    else:
                        self.cycle_len_rain = self.cycle_len
                        self.LAi_rain = self.LAi
                        self.HI_rain = self.HI

                # print('Perennial adjustment rainfed DONE      ', end = '')

                """ Adjustment for irrigated conditions"""

                if self.perennial:

                    """Use LGPT5 for minimum temperature less than or equal to five deg Celsius"""
                    if self.min_temp <= 5:

                        if self.LGPT5[i_row, i_col] < self.cycle_len:

                            self.cycle_len_irr = int(self.LGPT5[i_row, i_col].copy())
                            self.adjustForPerennialCrop(
                                self.cycle_len_irr, aLAI=self.aLAI, bLAI=self.bLAI, aHI=self.aHI, bHI=self.bHI, rain_or_irr='irr')

                        else:
                            self.cycle_len_irr = self.cycle_len
                            self.LAi_irr = self.LAi
                            self.HI_irr = self.HI

                    """Use LGPT10 for minimum temperature greater than five deg Celsius"""

                    if self.min_temp > 5:

                        if self.LGPT10[i_row, i_col] < self.cycle_len:

                            self.cycle_len_irr = int((self.LGPT10[i_row, i_col]).copy())
                            self.adjustForPerennialCrop(
                                self.cycle_len_irr, aLAI=self.aLAI, bLAI=self.bLAI, aHI=self.aHI, bHI=self.bHI, rain_or_irr='irr')

                        else:
                            self.cycle_len_irr = self.cycle_len
                            self.LAi_irr = self.LAi
                            self.HI_irr = self.HI

                # print('Perennial adjustment irrigated DONE      ', end = '')

                """ Calculation of each individual day's yield for rainfed and irrigated conditions"""

                for i_cycle in range(start_doy, end_doy+1, step_doy):

                    """Repeat the climate data two times and concatenate for computational convenience. If perennial, the cycle length
                    will be different for separate conditions"""

                    minT_daily_2year = np.tile(minT_daily_point, 2)
                    maxT_daily_2year = np.tile(maxT_daily_point, 2)
                    shortRad_daily_2year = np.tile(shortRad_daily_point, 2)
                    # wind2m_daily_2year = np.tile(wind2m_daily_point, 2)
                    totalPrec_daily_2year = np.tile(totalPrec_daily_point, 2)
                    pet_daily_2year = np.tile(pet_daily_point, 2)

                    # print('cycle starting')

                    """ Time slicing tiled climate data with corresponding cycle lengths for rainfed and irrigated conditions"""

                    if self.perennial:

                        """For rainfed"""

                        # extract climate data within the season to pass in to calculation classes
                        minT_daily_season_rain = minT_daily_2year[i_cycle: i_cycle +
                                                                  self.cycle_len_rain]
                        maxT_daily_season_rain = maxT_daily_2year[i_cycle: i_cycle +
                                                                  self.cycle_len_rain]
                        shortRad_daily_season_rain = shortRad_daily_2year[i_cycle: i_cycle+
                                                                          self.cycle_len_rain]
                        # wind2m_daily_season_rain = wind2m_daily_2year[i_cycle : i_cycle+self.cycle_len_rain]
                        # totalPrec_daily_season_rain = totalPrec_daily_2year[i_cycle : i_cycle+self.cycle_len_rain]
                        # pet_daily_season_rain = pet_daily_2year[i_cycle : i_cycle+self.cycle_len_rain]

                        """For irrigated"""
                        # extract climate data within the season to pass in to calculation classes
                        minT_daily_season_irr = minT_daily_2year[i_cycle: i_cycle +
                                                                 self.cycle_len_irr]
                        maxT_daily_season_irr = maxT_daily_2year[i_cycle: i_cycle +
                                                                 self.cycle_len_irr]
                        shortRad_daily_season_irr = shortRad_daily_2year[i_cycle: i_cycle+
                                                                         self.cycle_len_irr]
                        # wind2m_daily_season_irr = wind2m_daily_2year[i_cycle : i_cycle+self.cycle_len_irr]
                        # totalPrec_daily_season_irr = totalPrec_daily_2year[i_cycle : i_cycle+self.cycle_len_irr]
                        # pet_daily_season_irr = pet_daily_2year[i_cycle : i_cycle+self.cycle_len_irr]

                    else:

                        # extract climate data within the season to pass in to calculation classes
                        minT_daily_season = minT_daily_2year[i_cycle : i_cycle+self.cycle_len]
                        maxT_daily_season = maxT_daily_2year[i_cycle : i_cycle+self.cycle_len ]
                        shortRad_daily_season = shortRad_daily_2year[i_cycle : i_cycle+self.cycle_len]
                        # wind2m_daily_season = wind2m_daily_2year[i_cycle : i_cycle+self.cycle_len]
                        totalPrec_daily_season = totalPrec_daily_2year[i_cycle : i_cycle+self.cycle_len ]
                        pet_daily_season = pet_daily_2year[i_cycle : i_cycle+self.cycle_len ]

                        """Thermal Screening using each cycle length for rainfed and irrigated conditions"""

                        """ For the perennial, the adjusted cycle length for rainfed and irrigated conditions will be used. For the rest,
                        the user-specified cycle length will be applied"""

                    if self.set_adjustment:

                        """Creating Thermal Screening object classes for perennial rainfed and irrigated conditions"""
                        # print('adjustment start')
                        obj_screening_rain = ThermalScreening.ThermalScreening()
                        obj_screening_irr = ThermalScreening.ThermalScreening()

                        obj_screening_rain.setparameteradjusted(
                            cycle_len_rain=self.cycle_len_rain, cycle_len_irri=self.cycle_len_irr, Start_day=start_doy)
                        obj_screening_irr.setparameteradjusted(
                            cycle_len_rain=self.cycle_len_rain, cycle_len_irri=self.cycle_len_irr, Start_day=start_doy)

                        obj_screening_rain.setClimateData(
                            minT_daily_season_rain, maxT_daily_season_rain)
                        obj_screening_irr.setClimateData(
                            minT_daily_season_irr, maxT_daily_season_irr)

                        # if self.set_tclimate_screening:
                        #     obj_screening_rain.setThermalClimateScreening(self.t_climate[i_row, i_col], self.no_t_climate)
                        #     obj_screening_irr.setThermalClimateScreening(self.t_climate[i_row, i_col], self.no_t_climate)

                        if self.set_lgpt_screening:
                            obj_screening_rain.setLGPTScreening(
                                no_lgpt=self.no_lgpt, optm_lgpt=self.optm_lgpt)
                            obj_screening_irr.setLGPTScreening(
                                no_lgpt=self.no_lgpt, optm_lgpt=self.optm_lgpt)

                        # 5 Modification (SWH)
                        if self.set_Tsum_screening:
                            obj_screening_rain.setTSumScreening(
                                LnS=self.LnS, LsO=self.LsO, LO=self.LO, HnS=self.HnS, HsO=self.HsO, HO=self.HO)
                            obj_screening_irr.setTSumScreening(
                                LnS=self.LnS, LsO=self.LsO, LO=self.LO, HnS=self.HnS, HsO=self.HsO, HO=self.HO)

                        # 8 Modification
                        if self.setTypeBConstraint:
                            obj_screening_rain.applyTypeBConstraint(
                                data=self.data, input_temp_profile=obj_screening_rain.tprofile_rain, perennial_flag=True)
                            obj_screening_irr.applyTypeBConstraint(
                                data=self.data, input_temp_profile=obj_screening_irr.tprofile_irr, perennial_flag=True)

                        fc1_rain = 1
                        fc1_irr = 1
                        # print("\r 1. fc1 rain = {} and fc1 irr = {}".format(fc1_rain, fc1_irr)  , end = '\n')

                        # print("\rfc1 irr = ", str(fc1_irr), end = '\n')
                        # print("\nfc1 rain = ", str(fc1_rain))

                        if not obj_screening_rain.getSuitability():
                            continue
                        else:
                            fc1_rain = obj_screening_rain.getReductionFactor2(
                                rain_or_irr='rain')  # fc1 for rainfed condition

                        if not obj_screening_irr.getSuitability():
                            continue
                        else:
                            fc1_irr = obj_screening_irr.getReductionFactor2(
                                rain_or_irr='irr')  # fc1 for irrigated condition

                        # print("fc1 rain = ", str(fc1_rain) + r'    \n')
                        # print("fc1 irr = ", str(fc1_irr) + r'    \n')
                        # print("\r 2. fc1 rain = {} and fc1 irr = {}".format(fc1_rain, fc1_irr)  , end = '\n')

                        """Biomass Calculation and maximum attainable yield for perennial crops (rainfed and irrigated conditions). For perennial, cropwatcalc is not required."""

                        """Rainfed"""
                        obj_maxyield_rain = BioMassCalc.BioMassCalc(
                            i_cycle, i_cycle+self.cycle_len_rain-1, self.latitude_map[i_row, i_col])
                        obj_maxyield_rain.setClimateData(
                            minT_daily_season_rain, maxT_daily_season_rain, shortRad_daily_season_rain)
                        obj_maxyield_rain.setCropParameters(
                            self.LAi_rain, self.HI_rain, self.legume, self.adaptability)
                        obj_maxyield_rain.calculateBioMass()
                        est_yield_rainfed = obj_maxyield_rain.calculateYield()

                        # reduce thermal screening factor
                        est_yield_rainfed = est_yield_rainfed * fc1_rain

                        """append current cycle yield to a list rainfed"""
                        yield_of_all_crop_cycles_rainfed.append(
                            est_yield_rainfed)
                        fc1_rain_lst.append(fc1_rain)

                        """Irrigated"""
                        obj_maxyield_irr = BioMassCalc.BioMassCalc(
                            i_cycle, i_cycle+self.cycle_len_irr-1, self.latitude_map[i_row, i_col])
                        obj_maxyield_irr.setClimateData(
                            minT_daily_season_irr, maxT_daily_season_irr, shortRad_daily_season_irr)
                        obj_maxyield_irr.setCropParameters(
                            self.LAi_irr, self.HI_irr, self.legume, self.adaptability)
                        obj_maxyield_irr.calculateBioMass()
                        est_yield_irrigated = obj_maxyield_irr.calculateYield()

                        # reduce thermal screening factor
                        est_yield_irrigated = est_yield_irrigated * fc1_irr

                        """append current cycle yield to a list rainfed"""
                        yield_of_all_crop_cycles_irrig.append(
                            est_yield_irrigated)
                        fc1_irr_lst.append(fc1_irr)

                        # print("\rrainfed yield = {} and irrigated yield = {}".format(est_yield_rainfed, est_yield_irrigated)  , end = '\n')

                    else:

                        """Biomass Calculation and maximum attainable yield for non-perennial crops (rainfed and irrigated conditions). In here, Moisture limited yield calculation is done."""

                        """For rainfed and irrigated conditions, the biomass calculation is the same start."""
                        obj_screening = ThermalScreening.ThermalScreening()
                        obj_screening.setparameter(
                            cycle_len=self.cycle_len, start_day=start_doy)
                        obj_screening.setClimateData(
                            minT_daily_season, maxT_daily_season)

                        # if self.set_tclimate_screening:
                        #     obj_screening.setThermalClimateScreening(self.t_climate[i_row, i_col], self.no_t_climate)
                        if self.set_lgpt_screening:
                            obj_screening.setLGPTScreening(
                                self.no_lgpt, self.optm_lgpt)
                        # 5 Modification (SWH)
                        if self.set_Tsum_screening:
                            obj_screening.setTSumScreening(
                                self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO)

                        # Can remove this function (Updated new function goes to setupTypeBConstraint)
                        # if self.set_Tprofile_screening:
                        #     obj_screening.setTProfileScreening(self.no_Tprofile, self.optm_Tprofile)
                        if self.setTypeBConstraint:
                            obj_screening.applyTypeBConstraint(
                                data=self.data, input_temp_profile=obj_screening.tprofile, perennial_flag=False)

                        fc = 1

                        if not obj_screening.getSuitability():
                            continue
                        else:
                            # the setting of rainfed or irrigated conditions will be neglected
                            fc = obj_screening.getReductionFactor2(
                                rain_or_irr='other')

                        # print('\r fc list length =      ', len(fc_lst), end = '\n')

                        """For both irrigated and rainfed conditions"""

                        """Rainfed Conditions"""
                        obj_maxyield = BioMassCalc.BioMassCalc(
                            i_cycle, i_cycle+self.cycle_len-1, self.latitude_map[i_row, i_col])
                        obj_maxyield.setClimateData(
                            minT_daily_season, maxT_daily_season, shortRad_daily_season)
                        obj_maxyield.setCropParameters(
                            self.LAi, self.HI, self.legume, self.adaptability)
                        obj_maxyield.calculateBioMass()
                        est_yield_irrigated = obj_maxyield.calculateYield()

                        est_yield_irrigated = est_yield_irrigated * fc

                        yield_of_all_crop_cycles_irrig.append(
                            est_yield_irrigated)
                        fc_lst.append(fc)

                        """CropWatCalc is applied for rainfed conditions"""

                        obj_cropwat = CropWatCalc.CropWatCalc(
                            i_cycle, i_cycle+self.cycle_len-1)
                        obj_cropwat.setClimateData(
                            pet_daily_season, totalPrec_daily_season)
                        # check Sa is a raster or single value and extract Sa value accordingly
                        if len(np.array(self.Sa).shape) == 2:
                            Sa_temp = self.Sa[i_row, i_col]
                        else:
                            Sa_temp = self.Sa
                        obj_cropwat.setCropParameters(self.d_per, self.kc, self.kc_all, self.yloss_f,
                                                      self.yloss_f_all, est_yield_irrigated, self.D1, self.D2, Sa_temp, self.pc)
                        est_yield_moisture_limited = obj_cropwat.calculateMoistureLimitedYield()

                        fc2_value = obj_cropwat.getfc2factormap()
                        fc2_lst.append(fc2_value)

                        """append current cycle yield to a list"""
                        yield_of_all_crop_cycles_rainfed.append(
                            est_yield_moisture_limited)

                """Getting Maximum Attainable Yield from the list for irrigated and rainfed conditions and the Crop Calendar"""

                # get maximum yield from all simulation for a particular location (pixel) and assign to final map
                if len(yield_of_all_crop_cycles_irrig) > 0:
                    self.final_yield_irrig[i_row, i_col] = np.max(
                        yield_of_all_crop_cycles_irrig)
                    self.crop_calender_irr[i_row, i_col] = (yield_of_all_crop_cycles_irrig.index(np.max(
                        yield_of_all_crop_cycles_irrig)) + 1) * step_doy  # Crop calendar for irrigated condition

                if len(yield_of_all_crop_cycles_rainfed) > 0:
                    self.final_yield_rainfed[i_row, i_col] = np.max(
                        yield_of_all_crop_cycles_rainfed)
                    self.crop_calender_rain[i_row, i_col] = (yield_of_all_crop_cycles_rainfed.index(np.max(
                        yield_of_all_crop_cycles_rainfed)) + 1) * step_doy  # Crop calendar for rainfed condition

                if self.perennial:

                    if len(fc1_rain_lst) > 0 or len(fc1_irr_lst) > 0:
                        self.fc1_rain[i_row, i_col] = fc1_rain_lst[yield_of_all_crop_cycles_rainfed.index(
                            np.max(yield_of_all_crop_cycles_rainfed))]
                        self.fc1_irr[i_row, i_col] = fc1_irr_lst[yield_of_all_crop_cycles_irrig.index(
                            np.max(yield_of_all_crop_cycles_irrig))]
                else:

                    if len(fc_lst) > 0:
                        self.fc1[i_row, i_col] = fc_lst[yield_of_all_crop_cycles_rainfed.index(
                            np.max(yield_of_all_crop_cycles_rainfed))]

                    if len(fc2_lst) > 0:
                        self.fc2[i_row, i_col] = fc2_lst[yield_of_all_crop_cycles_rainfed.index(
                            np.max(yield_of_all_crop_cycles_rainfed))]

                print('\rDone %: ' + str(round(count_pixel_completed /
                        total*100, 2)), end='\r')

        print('\nSimulations Completed !')

    def getEstimatedYieldRainfed(self):
        """Estimation of Maximum Yield for Rainfed scenario

        Returns:
            2D NumPy: the maximum attainable yield under the provided climate conditions, 
                      under rain-fed conditions [kg/ha]
        """        
        return self.final_yield_rainfed

    def getEstimatedYieldIrrigated(self):
        """Estimation of Maximum Yield for Irrigated scenario

        Returns:
            2D NumPy: the maximum attainable yield under the provided climate conditions, 
                      under irrigated conditions [kg/ha]
        """
        return self.final_yield_irrig

    def getOptimumCycleStartDateIrrigated(self):
        """
        Function for optimum starting date for irrigated condition.

        Returns
        -------
        TYPE: 2-D numpy array.
            Optimum starting date for irrigated condition.

        """
        return self.crop_calender_irr

    def getOptimumCycleStartDateRainfed(self):
        """
        Function for optimum starting date for rainfed condition.

        Returns
        -------
        TYPE: 2-D numpy array.
            Optimum starting date for rainfed condition.

        """
        return self.crop_calender_rain

    def getThermalReductionFactor(self):
        """
        Function for thermal reduction factor (fc1) map. For perennial crop,
        the function produces a list of fc1 maps for both conditions. Only one 
        fc1 map is produced for non-perennial crops, representing both rainfed 
        and irrigated conditions

        Returns
        -------
        TYPE: A python list of 2-D numpy arrays: [fc1 rainfed, fc1 irrigated] 
        or a 2-D numpy array.
            Thermal reduction factor map (fc1) for corresponding conditions.

        """
        if self.perennial:
            return [self.fc1_rain, self.fc1_irr]
        else:
            return self.fc1

    def getMoistureReductionFactor(self):
        """
        Function for reduction factor map due to moisture deficit (fc2) for 
        rainfed condition. Only fc2 map is produced for non-perennial crops.
        
        Returns
        -------
        TYPE: 2-D numpy array
            Reduction factor due to moisture deficit (fc2).

        """

        if not self.perennial:
            return self.fc2
        else:
            print('Map is not produced because moisture deficit does not apply limitation to Perennials')


#----------------- End of file -------------------------#
