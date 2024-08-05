"""
PyAEZ version 3.0 (Dec 2024)
This CropSimulation Class simulates all the possible crop cycles to find 
the best crop cycle that produces maximum yield for a particular grid
2020: N. Lakmal Deshapriya
2022/2023: Swun Wunna Htet, Kittiphon Boonma
2023 (Dec): Swun Wunna Htet
2024 (Dec): Swun Wunna Htet

Modifications
1.  Minimum cycle length checking logic added to crop simulation.
2.  New crop parameters: minimum cycle length, maximum cycle length, plant height is added logic added.
3.  Removing unnecessary variables in the algorithm for slight code enhancement.
"""

import numpy as np
import pandas as pd
try:
    import gdal
except:
    from osgeo import gdal

from pyaez import UtilitiesCalc,BioMassCalc,ETOCalc,CropWatCalc,ThermalScreening, LGPCalc

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
        self.setTypeBConstraint = False
        self.set_monthly = False
        self.set_daily = False
    
    """--------------------- MANDATORY FUNCTIONS START HERE --------------------------"""

    def setMonthlyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity):
        """
        (MANDATORY FUNCTION)
        Load MONTHLY climate data into the Class and calculate the Reference Evapotranspiration (ETo).
        All climatic variables used in the simulations will be interpolated into daily values with 365 days.
        Users need to run setLocationTerrainData() before running this function.

        Parameters
        ----------
        min_temp (3D NumPy): Monthly minimum temperature [Celcius]
        max_temp (3D NumPy): Monthly maximum temperature [Celcius]
        precipitation (3D NumPy): Monthly total precipitation [mm/day]
        short_rad (3D NumPy): Monthly solar radiation [W/m2]
        wind_speed (3D NumPy): Monthly windspeed at 2m altitude [m/s]
        rel_humidity (3D NumPy): Monthly relative humidity [percentage decimal, 0-1]
        
        Returns
        -------
        None.
        """
        rel_humidity[rel_humidity > 0.99] = 0.99
        rel_humidity[rel_humidity < 0.05] = 0.05
        short_rad[short_rad < 0] = 0
        wind_speed[wind_speed < 0] = 0

        meanT_monthly = 0.5*(min_temp+max_temp)

        
        # Empty array creation
        self.meanT_daily = np.zeros((self.im_height, self.im_width, 365))
        self.totalPrec_daily = np.zeros((self.im_height, self.im_width, 365))
        self.pet_daily = np.zeros((self.im_height, self.im_width, 365))
        self.minT_daily = np.zeros((self.im_height, self.im_width, 365))
        self.maxT_daily = np.zeros((self.im_height, self.im_width, 365))
        self.shortRad_daily = np.zeros((self.im_height, self.im_width, 365))
        self.wind2m_daily = np.zeros((self.im_height, self.im_width, 365))
        self.rel_humidity_daily = np.zeros((self.im_height, self.im_width, 365))

        # Interpolate monthly to daily data
        obj_utilities = UtilitiesCalc.UtilitiesCalc()

        for i_row in range(self.im_height):
            for i_col in range(self.im_width):

                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue

                self.meanT_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(meanT_monthly[i_row, i_col, :], 1, 365)
                self.minT_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(min_temp[i_row, i_col, :], 1, 365)
                self.maxT_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(max_temp[i_row, i_col, :], 1, 365)
                self.totalPrec_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(precipitation[i_row, i_col, :], 1, 365, no_minus_values=True)
                self.shortRad_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(short_rad[i_row, i_col, :], 1, 365, no_minus_values=True)
                self.wind2m_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(wind_speed[i_row, i_col, :], 1, 365, no_minus_values=True)
                self.rel_humidity_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(rel_humidity[i_row, i_col, :], 1, 365, no_minus_values=True)

                # calculation of reference evapotranspiration (ETo)
                obj_eto = ETOCalc.ETOCalc(1, 365, self.latitude[i_row, i_col], self.elevation[i_row, i_col])
                
                # convert w/m2 to MJ/m2/day
                shortrad_daily_MJm2day = (self.shortRad_daily * 3600 * 24)/1000000
                obj_eto.setClimateData(self.minT_daily[i_row, i_col, :], self.maxT_daily[i_row, i_col, :],
                                        self.wind2m_daily[i_row, i_col, :], shortrad_daily_MJm2day, self.rel_humidity_daily[i_row, i_col, :])
                self.pet_daily[i_row, i_col, :] = obj_eto.calculateETO()

        # Sea-level adjusted mean temperature
        self.meanT_daily_sealevel = self.meanT_daily + \
            np.tile(np.reshape(self.elevation/100*0.55,
                    (self.im_height, self.im_width, 1)), (1, 1, 365))
        # P over PET ratio(to eliminate nan in the result, nan is replaced with zero)
        self.P_by_PET_daily = np.divide(
            self.totalPrec_daily, self.pet_daily, out=np.zeros_like(self.totalPrec_daily), where=(self.pet_daily != 0))

        self.set_monthly=True
    
    def setDailyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity, leap_year = False):
        """
        (MANDATORY FUNCTION)
        Load DAILY climate data into the Class and calculate the Reference Evapotranspiration (ETo).
        Users need to run setLocationTerrainData() before running this function.

        Parameter
        ---------
        min_temp (3D NumPy): Daily minimum temperature [Celcius]
        max_temp (3D NumPy): Daily maximum temperature [Celcius]
        precipitation (3D NumPy): Daily total precipitation [mm/day]
        short_rad (3D NumPy): Daily solar radiation [W/m2]
        wind_speed (3D NumPy): Daily windspeed at 2m altitude [m/s]
        rel_humidity (3D NumPy): Daily relative humidity [percentage decimal, 0-1]
        leap_year (Boolean): True for time dimension of 366, False for time dimension of 365 (Default)
        
        Returns
        -------
        None.
        """
        doy = 366 if leap_year else 365

        # Interpolate monthly to daily data
        obj_utilities = UtilitiesCalc.UtilitiesCalc()

        self.minT_daily = min_temp.copy()
        self.maxT_daily = max_temp.copy()
        self.meanT_daily = (self.minT_daily + self.maxT_daily)/2
        self.totalPrec_daily = precipitation.copy()
        self.shortRad_daily = short_rad.copy()
        self.wind2m_daily = wind_speed.copy()
        self.rel_humidity_daily = rel_humidity.copy()

        self.rel_humidity_daily[self.rel_humidity_daily > 0.99] = 0.99
        self.rel_humidity_daily[self.rel_humidity_daily < 0.05] = 0.05
        self.shortRad_daily[self.shortRad_daily < 0] = 0

        self.pet_daily = np.zeros((self.im_height, self.im_width, doy))

        for i_row in range(self.im_height):
            for i_col in range(self.im_width):

                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                # calculation of reference evapotranspiration (ETo)
                obj_eto = ETOCalc.ETOCalc(1, doy, self.latitude[i_row, i_col], self.elevation[i_row, i_col])
                
                # convert w/m2 to MJ/m2/day
                shortrad_daily_MJm2day = (self.shortRad_daily[i_row, i_col,:] * 3600 * 24)/1000000
                obj_eto.setClimateData(self.minT_daily[i_row, i_col, :], self.maxT_daily[i_row, i_col, :],
                                        self.wind2m_daily[i_row, i_col, :], shortrad_daily_MJm2day, self.rel_humidity_daily[i_row, i_col, :])
                self.pet_daily[i_row, i_col, :] = obj_eto.calculateETO()
        
        self.set_daily = True
    
    def setLocationTerrainData(self, lat_min, lat_max, elevation):
        """
        (MANDAOTRY FUNCTION)
        Load geographical extents and elevation data in to the Class, 
        and create a latitude map.

        Parameters
        ----------
        lat_min (float): the minimum latitude of the AOI in decimal degrees
        lat_max (float): the maximum latitude of the AOI in decimal degrees
        elevation (2D NumPy): elevation map in metres
        
        Returns
        -------
        None.
        """
        self.elevation = elevation
        self.im_height = elevation.shape[0]
        self.im_width = elevation.shape[1]
        self.latitude = UtilitiesCalc.UtilitiesCalc().generateLatitudeMap(lat_min, lat_max, self.im_height, self.im_width)
    
    def readCropandCropCycleParameters(self, file_path, crop_name):
        """
        (MANDATORY FUNCTION)
        Importing the excel sheet of crop-specific parameters,
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
                               [crop_df_index], D2=crop_df['D2'][crop_df_index], min_temp=crop_df['min_temp'][crop_df_index], aLAI=crop_df['aLAI'][crop_df_index], bLAI=crop_df['bLAI'][crop_df_index], aHI=crop_df['aHI'][crop_df_index], bHI=crop_df['bHI'][crop_df_index],
                               min_cycle_len=crop_df['min_cycle_len'][crop_df_index], max_cycle_len=crop_df['max_cycle_len'][crop_df_index], plant_height = crop_df['height'][crop_df_index])
        self.setCropCycleParameters(stage_per=[crop_df['stage_per_1'][crop_df_index], crop_df['stage_per_2'][crop_df_index], crop_df['stage_per_3'][crop_df_index], crop_df['stage_per_4'][crop_df_index]], kc=[crop_df['kc_0'][crop_df_index], crop_df['kc_1'][crop_df_index], crop_df['kc_2']
                                    [crop_df_index]], kc_all=crop_df['kc_all'][crop_df_index], yloss_f=[crop_df['yloss_f0'][crop_df_index], crop_df['yloss_f1'][crop_df_index], crop_df['yloss_f2'][crop_df_index], crop_df['yloss_f3'][crop_df_index]], yloss_f_all=crop_df['yloss_f_all'][crop_df_index])
        # perennial = 1, annual = 0
        if crop_df['annual/perennial flag'][crop_df_index] == 1:
            self.perennial = True
        else:
            self.perennial = False

        # If users provide all TSUM thresholds, TSUM screening will be done. Otherwise, TSUM screening will not be activated.
        if np.all([crop_df['LnS'][crop_df_index] != np.nan, crop_df['LsO'][crop_df_index] != np.nan, crop_df['LO'][crop_df_index] != np.nan, crop_df['HnS'][crop_df_index] != np.nan, crop_df['HsO'][crop_df_index] != np.nan, crop_df['HO'][crop_df_index] != np.nan]):
            self.setTSumScreening(LnS=crop_df['LnS'][crop_df_index], LsO=crop_df['LsO'][crop_df_index], LO=crop_df['LO'][crop_df_index],
                                  HnS=crop_df['HnS'][crop_df_index], HsO=crop_df['HsO'][crop_df_index], HO=crop_df['HO'][crop_df_index])

        # releasing memory
        del (crop_df_index, crop_df)

    def setSoilWaterParameters(self, Sa, pc):
        """
        (MANDATORY FUNCTION)
        Setting up the parameters related to the soil water storage.

        Parameters
        ----------
        Sa (float or 2D numpy): Available soil moisture holding capacity (mm/m)
        pc (float): Soil water depletion fraction below which ETa<ETo
    
        Returns
        -------
        None.
        """        
        self.Sa = Sa  # available soil moisture holding capacity (mm/m) , assumption
        self.pc = pc  # soil water depletion fraction below which ETa < ETo (from literature)


    """Nested functions within the mandatory functions"""

    def setCropParameters(self, LAI, HI, legume, adaptability, cycle_len, D1, D2, min_temp, aLAI, bLAI, aHI, bHI, min_cycle_len, max_cycle_len, plant_height):
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
        self.d_per = stage_per  # Percentage for D1, D2, D3, D4 stages
        self.kc = kc  # 3 crop water requirements for initial, reproductive, the end of the maturation stages
        self.kc_all = kc_all  # crop water requirements for entire growth cycle
        self.yloss_f = yloss_f  # yield loss for D1, D2, D3, D4
        self.yloss_f_all = yloss_f_all  # yield loss for entire growth cycle
    
    def LeafAreaIndexAdjustment(self, cycle_len, oLAI, aLAI, bLAI):
        """
        (NESTED FUNCTION)
        Leaf Area Index adjustment function for perennial crops within effective cycle length
        Exclusively for PERENNIAL simulation.
        
        Parameters
        ----------
        cycle_len (int): effective cycle length [days]
        oLAI (float): original LAI
        aLAI (int): alpha LAI threshold
        bLAI (int): beta LAI threshold

        Return
        ------
        Adjusted LAI (float)
        """
        return  oLAI * ((cycle_len-aLAI)/bLAI)
    
    def HarvestIndexAdjustment(self, cycle_len, oHI, aHI, bHI):
        """
        (NESTED FUNCTION)
        Harvest Index adjustment function for perennial crops within effective cycle length
        Exclusively for PERENNIAL simulation.
        
        Parameters
        ----------
        cycle_len (int): effective cycle length [days]
        oHI (float): original HI
        aHI (int): alpha HI threshold
        bHI (int): beta HI threshold

        Return
        ------
        Adjusted HI (float)
        """
        return  oHI * ((cycle_len-aHI)/bHI)
    
    # def adjustForPerennialCrop(self,  cycle_len, oLAI, oHI,aLAI, bLAI, aHI, bHI):
    #     """
    #     (NESTED FUNCTION)
    #     Leaf Area Index (LAI) and the Harvest Index (HI) based on the effective growing period.
    #     Exclusively for PERENNAIL simulations.

    #     Parameters
    #     ----------
    #     oLAI (float): original LAI
    #     oHI (float): original HI
    #     aLAI (int): alpha coefficient for LAI
    #     bLAI (int): beta coefficient for LAI
    #     aHI (int): alpha coefficient for HI
    #     bHI (int): beta coefficient for HI
        
    #     Returns:
    #     --------
    #     tuple (adjusted LAI, adjusted HI)
    #     """        
    #     # leaf area index adjustment for perennial crops
    #     adj_LAI = oLAI * ((cycle_len-aLAI)/bLAI)
    #     # harvest index adjustment for perennial crops
    #     adj_HI = oHI * ((cycle_len-aHI)/bHI)

    #     return adj_LAI,adj_HI
    
    def ImportLGPandLGPT(self, lgp, lgpt5, lgpt10):
        """
        (MANDATORY FUNCTION)
        Importing LGP and temperature growing period data.

        Parameters
        ----------
        lgp (2-D NumPy Array): Length of Growing Period [days].
        lgpt5 (2-D NumPy Array): Temperature Growing Period at 5℃ threshold.
        lgpt10 (2-D NumPy Array): Temperature Growing Period at 10℃ threshold.

        Returns
        -------
        None.

        """
        self.LGP = lgp
        self.LGPT5 = lgpt5
        self.LGPT10 = lgpt10
    
    """----------------------------  MANDATORY FUNCTIONS END HERE   --------------------------"""
    """---------------------THERMAL SCREENING FUNCTIONS STARTS HERE (OPTIONAL)--------------------------"""

    def setThermalClimateScreening(self, t_climate, no_t_climate):
        """
        The thermal screening function omit out user-specified thermal climate classes
        not suitable for a particular crop for crop simulation. Using this optional 
        function will activate application of thermal climate screening in crop cycle simulation.
    

        Parameters
        ----------
        t_climate (2-D NumPy Array): Thermal Climate.
        no_t_climate (list): A list of thermal climate classes not suitable for crop simulation.

        Returns
        -------
        None.

        """
        self.t_climate = t_climate
        self.no_t_climate = no_t_climate  # list of unsuitable thermal climate

        self.set_tclimate_screening = True

    
    def setLGPTScreening(self, no_lgpt, optm_lgpt):
        """
        Set screening parameters for thermal growing period (LGPt).

        Developer's Note: This function will not be available in the new function.

        Parameters
        ----------
        no_lgpt (3-item list): 3 'not suitable' LGPt conditions
        optm_lgpt (3-item list): 3 'optimum' LGPt conditions
        
        Returns
        -------
        None.
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
        LnS (int): Lower boundary of not-suitable accumulated heat unit range.
        LsO (int): Lower boundary of sub-optimal accumulated heat unit range.
        LO (int): Lower boundary of optimal accumulated heat unit range.
        HnS (int):Upper boundary of not-suitable accumulated heat range.
        HsO (int): Upper boundary of sub-optimal accumulated heat range.
        HO (int): Upper boundary of not-suitable accumulated heat range.

        Returns
        -------
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
        This thermal screening corresponds to permafrost characteristics screening.  
        Using this optional function will activate  permafrost screening in crop cycle simulation.
        
        Parameters
        ----------
        permaforst_class (2-D NumPy Array): Permafrost class (Obtained from Module I: Climate Regime).

        Returns
        -------
        None.

        """
        self.permafrost_class = permafrost_class  # permafrost class 2D numpy array
        self.set_Permafrost_screening = True

    def setCropSpecificRule(self, file_path, crop_name):
        """
        Optional function initiates the Crop Specific Rule (Temperature Profile 
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

        self.setTypeBConstraint = True

        # releasing data
        del (data)
    
    """---------------------THERMAL SCREENING FUNCTIONS END HERE (OPTIONAL)--------------------------"""
    """---------------------------- OPTIONAL FUNCTIONS STARTS HERE  ---------------------------------"""  

    def setStudyAreaMask(self, admin_mask, no_data_value):
        """Set clipping mask of the area of interest (optional)

        Args:
            admin_mask (2D NumPy/Binary): mask to extract only region of interest
            no_data_value (int): pixels with this value will be omitted during PyAEZ calculations
        Return:
            None.
        """
        self.im_mask = admin_mask
        self.nodata_val = no_data_value
        self.set_mask = True
    
    """---------------------------- OPTIONAL FUNCTIONS END HERE  ---------------------------------"""
    """------------------   MAIN FUNCTION OF CROP SIMULATION STARTS HERE  ------------------------"""

    def simulateCropCycle(self, start_doy=1, end_doy=365, step_doy=1, leap_year=False):
        """Running the crop cycle calculation/simulation.

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
        self.crop_calender_irr = np.zeros((self.im_height, self.im_width), dtype=int)
        self.crop_calender_rain = np.zeros((self.im_height, self.im_width), dtype=int)
        
        self.fc2 = np.zeros((self.im_height, self.im_width))
        self.fc1_rain = np.zeros((self.im_height, self.im_width))
        self.fc1_irr = np.zeros((self.im_height, self.im_width))


        for i_row in range(self.im_height):

            for i_col in range(self.im_width):

                # check current location (pixel) is outside of study area or not. if it's outside of study area goes to next location (pixel)
                # Those unsuitable
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        count_pixel_completed = count_pixel_completed + 1
                        print('\rDone %: ' + str(round(count_pixel_completed /
                        total*100, 2)), end='\r')
                        continue

                # 2. Permafrost screening
                if self.set_Permafrost_screening:
                    if np.logical_or(self.permafrost_class[i_row, i_col] == 1, self.permafrost_class[i_row, i_col] == 2):
                        count_pixel_completed = count_pixel_completed + 1
                        print('\rDone %: ' + str(round(count_pixel_completed /
                        total*100, 2)), end='\r')
                        continue

                # Thermal Climate Screening
                if self.set_tclimate_screening:
                    if self.t_climate[i_row, i_col] in self.no_t_climate:
                        count_pixel_completed = count_pixel_completed + 1
                        
                        print('\rDone %: ' + str(round(count_pixel_completed /
                        total*100, 2)), end='\r')
                        continue
                
                """Cycle length checking for rainfed and irrigated annuals.
                    Concerns with LGPt5 (irrigated) and LGP (rainfed)"""
                if not self.perennial:

                    "Harvest Index and Leaf Area Index are not adjusted."
                    LAi_rain = self.LAi
                    HI_rain = self.HI

                    LAi_irr = self.LAi
                    HI_irr = self.HI
                    
                    # for irrigated conditions
                    # In real simulation, this pixel would be omitted out
                    if int(self.LGPT5[i_row, i_col]) < self.min_cycle_len:
                        cycle_len_irr = 0
                    else:
                        cycle_len_irr = self.cycle_len
                    
                    # for rainfed condition
                    # In real simulation, this pixel would be omitted out for 
                    if int(self.LGP[i_row, i_col]) < self.min_cycle_len:
                        cycle_len_rain = 0
                    else:
                        cycle_len_rain = self.cycle_len
                    
                else:
                    """Cycle length checking for rainfed and irrigated perennials.
                    Concerns with LGPt5 (irrigated) and LGP (rainfed)"""

                    """Adjustment of cycle length, LAI and HI for Perennials"""
                    self.set_adjustment = True

                    """ Adjustment for RAINFED conditions"""
                    if int(self.LGP[i_row, i_col]) < self.min_cycle_len:
                        cycle_len_rain = 0
                        
                    else:
                        # effective cycle length will be our cycle length
                        cycle_len_rain = min(int(self.LGP[i_row, i_col]), self.max_cycle_len)

                        # Leaf Area Index adjustment for rainfed condition
                        if cycle_len_rain - self.aLAI > self.bLAI:
                            LAI_rain = self.LAi
                        elif cycle_len_rain - self.aLAI < 0:
                            LAI_rain = 0
                        else:
                            LAI_rain = self.LeafAreaIndexAdjustment(cycle_len= cycle_len_rain, oLAI = self.LAI, aLAI = self.aLAI, bLAI = self.bLAI)

                        # Harvest Index adjustment for rainfed conditions
                        if cycle_len_rain -self.aHI > self.bHI:
                            HI_rain = self.HI
                        elif cycle_len_rain - self.aHI <0:
                            HI_rain = 0
                        else:
                            HI_rain = self.HarvestIndexAdjustment(cycle_len = cycle_len_rain, oHI = self.HI, aHI = self.aHI, bHI = self.bHI)
                        
                    """ Adjustment for IRRIGATED conditions"""
                    """Use LGPT5 for minimum temperatures less than 8. Use LGPT10 for temperature greater than 8."""
                    # effective cycle length will be our cycle length
                    if self.min_temp <= 8:
                        if int(self.LGPT5[i_row, i_col]) < self.min_cycle_len:
                            self.cycle_len_irr = 0
                        else:
                            self.cycle_len_irr = min(int(self.LGPT5[i_row, i_col]), self.max_cycle_len)
                    elif self.min_temp >8:
                        if int(self.LGPT10[i_row, i_col]) < self.min_cycle_len:
                            self.cycle_len_irr = 0
                        else:
                            self.cycle_len_irr = min(int(self.LGPT10[i_row, i_col]), self.max_cycle_len)
                    
                    # Leaf Area Index adjustment for irrigated condition
                        if cycle_len_irr - self.aLAI > self.bLAI:
                            LAI_irr = self.LAi
                        elif cycle_len_irr - self.aLAI < 0:
                            LAI_irr = 0
                        else:
                            LAI_irr = self.LeafAreaIndexAdjustment(cycle_len= cycle_len_irr, oLAI = self.LAI, aLAI = self.aLAI, bLAI = self.bLAI)

                        # Harvest Index adjustment for irrigated condition
                        if cycle_len_irr -self.aHI > self.bHI:
                            HI_irr = self.HI
                        elif cycle_len_irr - self.aHI <0:
                            HI_irr = 0
                        else:
                            HI_irr = self.HarvestIndexAdjustment(cycle_len = cycle_len_irr, oHI = self.HI, aHI = self.aHI, bHI = self.bHI)
                
                count_pixel_completed = count_pixel_completed + 1       
                # this allows handing leap and non-leap year differently. This is only relevant for monthly data because this value will be used in interpolations.
                # In case of daily data, length of vector will be taken as number of days in  a year.
                if leap_year:
                    days_in_year = 366
                else:
                    days_in_year = 365

                # extract daily climate data for particular location.

                minT_daily_point = self.minT_daily[i_row, i_col, :]
                maxT_daily_point = self.maxT_daily[i_row, i_col, :]
                meanT_daily_point = self.meanT_daily[i_row, i_col,:]
                shortRad_daily_point = self.shortRad_daily[i_row, i_col, :]
                wind2m_daily_point = self.wind2m_daily[i_row, i_col, :]
                totalPrec_daily_point = self.totalPrec_daily[i_row, i_col, :]
                rel_humidity_daily_point = self.rel_humidity_daily[i_row, i_col, :]
                pet_daily_point = self.pet_daily[i_row, i_col, :]   


                # Empty arrays that stores yield estimations and fc1 and fc2 of all cycles per particular location (pixel)
                yield_of_all_crop_cycles_rainfed = np.empty(0, dtype= np.float16)
                yield_of_all_crop_cycles_irrig = np.empty(0, dtype= np.float16)

                fc1_rain_lst = np.empty(0, dtype= np.float16)
                fc1_irr_lst = np.empty(0, dtype= np.float16)

                fc2_lst = np.empty(0, dtype= np.float16)


                """ Calculation of each individual day's yield for rainfed and irrigated conditions"""

                for i_cycle in range(start_doy-1, end_doy, step_doy):

                    """Check if the first day of a cycle meets minimum temperature requirement. If not, all outputs will be zero.
                        And iterates to next cycle."""
                    if (minT_daily_point[i_cycle]+maxT_daily_point[i_cycle])/2 < self.min_temp:
                        est_yield_moisture_limited = 0.
                        fc1_rain = 0.
                        fc1_irr =0.
                        fc2_value = 0.
                        est_yield_irrigated = 0.

                        yield_of_all_crop_cycles_rainfed = np.append(yield_of_all_crop_cycles_rainfed, 0.)
                        yield_of_all_crop_cycles_irrig = np.append(yield_of_all_crop_cycles_irrig, 0.)
                        fc1_rain_lst = np.append(fc1_rain_lst, 0.)
                        fc1_irr_lst = np.append(fc1_irr_lst, 0.)
                        fc2_lst = np.append(fc2_lst, 0.)
                        continue
                    
                    """Repeat the climate data two times and concatenate for computational convenience. If perennial, the cycle length
                            will be different for separate conditions"""
                    minT_daily_2year = np.tile(minT_daily_point, 2)
                    maxT_daily_2year = np.tile(maxT_daily_point, 2)
                    shortRad_daily_2year = np.tile(shortRad_daily_point, 2)
                    wind2m_daily_2year = np.tile(wind2m_daily_point,2)
                    totalPrec_daily_2year = np.tile(totalPrec_daily_point, 2)
                    pet_daily_2year = np.tile(pet_daily_point, 2)
                    meanT_daily_2year = np.tile(meanT_daily_point, 2)
                    
                    if cycle_len_rain in[-1,0]:
                        est_yield_moisture_limited = 0.
                        fc1_rain = 0.
                        fc2_value = 0.
                    else:
                        """ Time slicing tiled climate data with corresponding cycle lengths for rainfed and irrigated conditions"""
                        """For rainfed"""

                        # extract climate data within the season to pass in to calculation classes
                        minT_daily_season_rain = minT_daily_2year[i_cycle: i_cycle + int(cycle_len_rain)]
                        maxT_daily_season_rain = maxT_daily_2year[i_cycle: i_cycle + int(cycle_len_rain)]
                        shortRad_daily_season_rain = shortRad_daily_2year[i_cycle: i_cycle+int(cycle_len_rain)]
                        pet_daily_season_rain = pet_daily_2year[i_cycle: i_cycle+int(cycle_len_rain)]
                        totalPrec_daily_season_rain = totalPrec_daily_2year[i_cycle: i_cycle+int(cycle_len_rain)]
                        wind_sp_daily_season_rain = wind2m_daily_2year[i_cycle: i_cycle+int(cycle_len_rain)]
                        
                        """Creating Thermal Screening object classes for perennial RAINFED conditions"""
                        obj_screening_rain = ThermalScreening.ThermalScreening()
                        """ For Perennials, 365 days of climate data will be used for Thermal Screening.
                            For Annuals, climate data within crop-specific cycle length will be used for Thermal Screening."""
                        if self.perennial:
                            obj_screening_rain.setClimateData(meanT_daily_2year[i_cycle: i_cycle+365], meanT_daily_2year[i_cycle: i_cycle+365])
                        else:
                            obj_screening_rain.setClimateData(meanT_daily_2year[i_cycle: i_cycle+cycle_len_irr], meanT_daily_2year[i_cycle: i_cycle+cycle_len_irr])
                        

                        if self.set_lgpt_screening:
                            obj_screening_rain.setLGPTScreening(no_lgpt=self.no_lgpt, optm_lgpt=self.optm_lgpt)

                        # TSUM Screening
                        if self.set_Tsum_screening:
                            obj_screening_rain.setTSumScreening(LnS=self.LnS, LsO=self.LsO, LO=self.LO, HnS=self.HnS, HsO=self.HsO, HO=self.HO)

                        # Crop-Specific Rule Screening
                        if self.setTypeBConstraint:
                            obj_screening_rain.applyTypeBConstraint(data=self.data, input_temp_profile=obj_screening_rain.tprofile, perennial_flag= self.perennial)

                        # Initial set up value for fc1 RAINFED
                        fc1_rain = 1.
                        fc1_rain = obj_screening_rain.getReductionFactor2()  # fc1 for rainfed condition
                        
                        if fc1_rain == 0.:
                            est_yield_moisture_limited = 0.
                            fc1_rain = 0.
                            fc2_value = 0.
                        else:
                            
                            if LAI_rain <=0 or HI_rain <= 0:
                                est_yield_rainfed = 0
                            else:

                                """If fc1 RAINFED IS NOT ZERO >>> BIOMASS RAINFED STARTS"""
                                obj_maxyield_rain = BioMassCalc.BioMassCalc(i_cycle+1, i_cycle+1+cycle_len_rain-1, self.latitude[i_row, i_col])
                                obj_maxyield_rain.setClimateData(minT_daily_season_rain, maxT_daily_season_rain, shortRad_daily_season_rain)
                                obj_maxyield_rain.setCropParameters(LAi_rain, HI_rain, self.legume, self.adaptability)
                                obj_maxyield_rain.calculateBioMass()
                                est_yield_rainfed = obj_maxyield_rain.calculateYield()

                            # reduce thermal screening factor
                            est_yield_rainfed = est_yield_rainfed * fc1_rain

                            """ For Annual RAINFED, crop water requirements are calculated in full procedures.
                                For Perennial RAINFED, procedures related with yield loss factors are omitted out.
                                """
                            fc2_value = 1.
                            
                            obj_cropwat = CropWatCalc.CropWatCalc(
                                i_cycle+1, i_cycle+1+cycle_len_rain-1, perennial_flag = self.perennial)
                            obj_cropwat.setClimateData(pet_daily_season_rain, totalPrec_daily_season_rain, 
                                                       wind_sp_daily_season_rain, minT_daily_season_rain, 
                                                       maxT_daily_season_rain)
                            
                            # check Sa is a raster or single value and extract Sa value accordingly
                            if len(np.array(self.Sa).shape) == 2:
                                Sa_temp = self.Sa[i_row, i_col]
                            else:
                                Sa_temp = self.Sa
                            obj_cropwat.setCropParameters(self.d_per, self.kc, self.kc_all, self.yloss_f,
                                                            self.yloss_f_all, est_yield_rainfed, self.D1, self.D2, Sa_temp, self.pc, self.plant_height)
                            est_yield_moisture_limited = obj_cropwat.calculateMoistureLimitedYield()

                            fc2_value = obj_cropwat.getfc2factormap()

                    yield_of_all_crop_cycles_rainfed = np.append(yield_of_all_crop_cycles_rainfed, est_yield_moisture_limited)
                    fc2_lst = np.append(fc2_lst, fc2_value)
                    fc1_rain_lst = np.append(fc1_rain_lst, fc1_rain)

                    """Error checking code snippet"""
                    if est_yield_moisture_limited == None or est_yield_moisture_limited == np.nan:
                        raise Exception('Crop Water Yield not returned. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
                    if fc2_value == None or fc2_value == np.nan:
                        raise Exception('fc2 value not returned. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
                    if len(fc1_rain_lst) != i_cycle+1 or fc1_rain == None or fc1_rain == np.nan:
                        raise Exception('Fc1 rain not properly appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
                    if len(yield_of_all_crop_cycles_rainfed) != i_cycle+1:
                        raise Exception('Rainfed yield list not properly appended') 
                    if len(fc2_lst) != i_cycle+1:
                        raise Exception('Fc2 list not appended properly. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))


                    ##########################
                    if self.cycle_len_irr in [-1, 0]:
                        est_yield_irrigated = 0.
                        fc1_irr = 0.
                    else:
                            
                        # extract climate data within the season to pass in to calculation classes
                        minT_daily_season_irr = minT_daily_2year[i_cycle: i_cycle +int(self.cycle_len_irr)]
                        maxT_daily_season_irr = maxT_daily_2year[i_cycle: i_cycle +int(self.cycle_len_irr)]
                        meanT_daily_season_irr = meanT_daily_2year[i_cycle: i_cycle +int(self.cycle_len_irr)]
                        shortRad_daily_season_irr = shortRad_daily_2year[i_cycle: i_cycle+int(self.cycle_len_irr)]
                        pet_daily_season_irr = pet_daily_2year[i_cycle: i_cycle+int(self.cycle_len_irr)]
                        totalPrec_daily_season_irr = totalPrec_daily_2year[i_cycle: i_cycle+int(self.cycle_len_irr)]
                        
                    
                        """Creating Thermal Screening object classes for IRRIGATED conditions"""
                        obj_screening_irr = ThermalScreening.ThermalScreening()
                        
                        """ For Perennials, 365 days of climate data will be used for Thermal Screening.
                        For Annuals, climate data within crop-specific cycle length will be used for Thermal Screening."""
                        if self.perennial:
                            obj_screening_irr.setClimate(meanT_daily_2year[i_cycle: i_cycle+365], meanT_daily_2year[i_cycle: i_cycle+365])
                        else:
                            obj_screening_irr.setClimateData(minT_daily_season_irr, maxT_daily_season_irr)

                        if self.set_lgpt_screening:
                            obj_screening_irr.setLGPTScreening(no_lgpt=self.no_lgpt, optm_lgpt=self.optm_lgpt)

                        # TSUM Screening
                        if self.set_Tsum_screening:
                            obj_screening_irr.setTSumScreening(LnS=self.LnS, LsO=self.LsO, LO=self.LO, HnS=self.HnS, HsO=self.HsO, HO=self.HO)

                        # Crop-Specific Rule Screening
                        if self.setTypeBConstraint:
                            obj_screening_irr.applyTypeBConstraint(data=self.data, input_temp_profile=obj_screening_irr.tprofile, perennial_flag= self.perennial)

                        # Initial set up value for fc1 RAINFED
                        fc1_irr = 1.

                        fc1_irr = obj_screening_irr.getReductionFactor2()  # fc1 for rainfed condition

                
                        if fc1_irr == 0.:
                            est_yield_irrigated = 0.
                        else:
                            if LAI_irr <=0 or HI_irr <= 0:
                                est_yield_irrigated = 0
                            else:
                                """If fc1 IRRIGATED IS NOT ZERO >>> BIOMASS IRRIGATED STARTS"""
                                obj_maxyield_irr = BioMassCalc.BioMassCalc(
                                    i_cycle+1, i_cycle+1+self.cycle_len_irr-1, self.latitude[i_row, i_col])
                                obj_maxyield_irr.setClimateData(
                                    minT_daily_season_irr, maxT_daily_season_irr, shortRad_daily_season_irr)
                                obj_maxyield_irr.setCropParameters(
                                    self.LAi_irr, self.HI_irr, self.legume, self.adaptability)
                                obj_maxyield_irr.calculateBioMass()
                                est_yield_irrigated = obj_maxyield_irr.calculateYield()
                            # reduce thermal screening factor
                            est_yield_irrigated = est_yield_irrigated * fc1_irr

                    yield_of_all_crop_cycles_irrig = np.append(yield_of_all_crop_cycles_irrig, est_yield_irrigated)
                    fc1_irr_lst = np.append(fc1_irr_lst, fc1_irr)

                    # Error raising
                    if est_yield_irrigated == None or est_yield_irrigated== np.nan:
                        raise Exception('Biomass Yield for irrigated not returned. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))

                    if len(yield_of_all_crop_cycles_irrig) != i_cycle+1:
                        raise Exception('Irrigated yield list not properly appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
                                
                    if len(fc1_irr_lst) != i_cycle+1 or fc1_irr == None or fc1_irr == np.nan:
                        raise Exception('Fc1 irr not properly appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
                    
                    if len(fc1_irr_lst)!= i_cycle+1 or fc1_rain == None or fc1_rain == np.nan:
                        raise Exception('Fc1 irr not properly appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))

                """Getting Maximum Attainable Yield from the list for irrigated and rainfed conditions and the Crop Calendar"""

                # get agro-climatic yield and crop calendar for IRRIGATED condition
                if np.logical_and(len(yield_of_all_crop_cycles_irrig) == len(fc1_irr_lst), len(yield_of_all_crop_cycles_irrig) == len(fc1_irr_lst)):

                    self.final_yield_irrig[i_row, i_col] = np.max(yield_of_all_crop_cycles_irrig) # Maximum attainable yield

                    # Array index where maximum yield is obtained
                    i = np.where(yield_of_all_crop_cycles_irrig == np.max(yield_of_all_crop_cycles_irrig))[0][0] # index of maximum yield

                    self.crop_calender_irr[i_row, i_col] = int(i+1)*step_doy # Crop calendar for irrigated condition

                    self.fc1_irr[i_row, i_col] = fc1_irr_lst[i] # fc1 irrigated for the specific crop calendar DOY

                # get agro-climatic yield and crop calendar for RAINFED condition
                if np.logical_and(len(yield_of_all_crop_cycles_rainfed) == len(fc1_rain_lst), len(yield_of_all_crop_cycles_rainfed) == len(fc1_rain_lst)):
                    self.final_yield_rainfed[i_row, i_col] = np.max(yield_of_all_crop_cycles_rainfed) # Maximum attainable yield

                    i1 = np.where(yield_of_all_crop_cycles_rainfed == np.max(yield_of_all_crop_cycles_rainfed))[0][0] # index of maximum yield
                    
                    self.crop_calender_rain[i_row, i_col] = int(i1+1) * step_doy # Crop calendar for rainfed condition
                    
                    self.fc1_rain[i_row, i_col] = fc1_rain_lst[i1]
                    
                    # if not self.perennial:
                    self.fc2[i_row, i_col] = fc2_lst[i1]


                print('\rDone %: ' + str(round(count_pixel_completed / total*100, 2)), end='\r')
        
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
        return [self.fc1_rain, self.fc1_irr]

    def getMoistureReductionFactor(self):
        """
        Function for reduction factor map due to moisture deficit (fc2) for 
        rainfed condition. Only fc2 map is produced for non-perennial crops.
        
        Returns
        -------
        TYPE: 2-D numpy array
            Reduction factor due to moisture deficit (fc2).

        """

        return self.fc2
    
    """------------------   MAIN FUNCTION OF CROP SIMULATION ENDS HERE  ------------------------"""
    """------------------       MODULE II:CROP SIMULATION ENDS HERE     ------------------------"""

    #----------------------------------------------DEVELLOPER'S CODES --------------------------------------------------#
    def simulateCropCycle(self, start_doy=1, end_doy=365, step_doy=1, leap_year=False):
        """Running the crop cycle calculation/simulation.

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
        self.crop_calender_irr = np.zeros((self.im_height, self.im_width), dtype=int)
        self.crop_calender_rain = np.zeros((self.im_height, self.im_width), dtype=int)
        
        self.fc2 = np.zeros((self.im_height, self.im_width))
        self.fc1_rain = np.zeros((self.im_height, self.im_width))
        self.fc1_irr = np.zeros((self.im_height, self.im_width))


        for i_row in range(self.im_height):

            for i_col in range(self.im_width):

                # check current location (pixel) is outside of study area or not. if it's outside of study area goes to next location (pixel)
                # Those unsuitable
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        count_pixel_completed = count_pixel_completed + 1
                        print('\rDone %: ' + str(round(count_pixel_completed /
                        total*100, 2)), end='\r')
                        continue

                # 2. Permafrost screening
                if self.set_Permafrost_screening:
                    if np.logical_or(self.permafrost_class[i_row, i_col] == 1, self.permafrost_class[i_row, i_col] == 2):
                        count_pixel_completed = count_pixel_completed + 1
                        print('\rDone %: ' + str(round(count_pixel_completed /
                        total*100, 2)), end='\r')
                        continue

                # Thermal Climate Screening
                if self.set_tclimate_screening:
                    if self.t_climate[i_row, i_col] in self.no_t_climate:
                        count_pixel_completed = count_pixel_completed + 1
                        
                        print('\rDone %: ' + str(round(count_pixel_completed /
                        total*100, 2)), end='\r')
                        continue
                
                """Cycle length checking for rainfed and irrigated annuals.
                    Concerns with LGPt5 (irrigated) and LGP (rainfed)"""
                if not self.perennial:

                    "Harvest Index and Leaf Area Index are not adjusted."
                    LAi_rain = self.LAi
                    HI_rain = self.HI

                    LAi_irr = self.LAi
                    HI_irr = self.HI
                    
                    # for irrigated conditions
                    # In real simulation, this pixel would be omitted out
                    if int(self.LGPT5[i_row, i_col]) < self.min_cycle_len:
                        cycle_len_irr = 0
                    else:
                        cycle_len_irr = self.cycle_len
                    
                    # for rainfed condition
                    # In real simulation, this pixel would be omitted out for 
                    if int(self.LGP[i_row, i_col]) < self.min_cycle_len:
                        cycle_len_rain = 0
                    else:
                        cycle_len_rain = self.cycle_len
                    
                else:
                    """Cycle length checking for rainfed and irrigated perennials.
                    Concerns with LGPt5 (irrigated) and LGP (rainfed)"""

                    """Adjustment of cycle length, LAI and HI for Perennials"""
                    self.set_adjustment = True

                    """ Adjustment for RAINFED conditions"""
                    if int(self.LGP[i_row, i_col]) < self.min_cycle_len:
                        cycle_len_rain = 0
                        
                    else:
                        # effective cycle length is determined by comparing LGP with maximum cycle length. However, 
                        # the effective cycle length must not be greater than reference cycle length
                        cycle_len_rain = self.cycle_len if (min(int(self.LGP[i_row, i_col]), self.max_cycle_len) > self.cycle_len) else (min(int(self.LGP[i_row, i_col]), self.max_cycle_len))

                        # Leaf Area Index adjustment for rainfed condition
                        if (cycle_len_rain - self.aLAI > self.bLAI):
                            LAI_rain = self.LAi
                        elif cycle_len_rain - self.aLAI < 0:
                            LAI_rain = 0
                        else:
                            LAI_rain = self.LeafAreaIndexAdjustment(cycle_len= cycle_len_rain, oLAI = self.LAI, aLAI = self.aLAI, bLAI = self.bLAI)

                        # Harvest Index adjustment for rainfed conditions
                        if cycle_len_rain -self.aHI > self.bHI:
                            HI_rain = self.HI
                        elif cycle_len_rain - self.aHI <0:
                            HI_rain = 0
                        else:
                            HI_rain = self.HarvestIndexAdjustment(cycle_len = cycle_len_rain, oHI = self.HI, aHI = self.aHI, bHI = self.bHI)
                        
                    """ Adjustment for IRRIGATED conditions"""
                    """Use LGPT5 for minimum temperatures less than 8. Use LGPT10 for temperature greater than 8."""
                    # effective cycle length will be our cycle length
                    if self.min_temp <= 8:
                        if int(self.LGPT5[i_row, i_col]) < self.min_cycle_len:
                            cycle_len_irr = 0
                        else:
                            cycle_len_irr = min(int(self.LGPT5[i_row, i_col]), self.max_cycle_len)
                    elif self.min_temp >8:
                        if int(self.LGPT10[i_row, i_col]) < self.min_cycle_len:
                            cycle_len_irr = 0
                        else:
                            cycle_len_irr = min(int(self.LGPT10[i_row, i_col]), self.max_cycle_len)
                    
                    cycle_len_irr = self.cycle_len if (cycle_len_irr > self.cycle_len) else cycle_len_irr
                    
                    # Leaf Area Index adjustment for irrigated condition
                    if cycle_len_irr - self.aLAI > self.bLAI:
                        LAI_irr = self.LAi
                    elif cycle_len_irr - self.aLAI < 0:
                        LAI_irr = 0
                    else:
                        LAI_irr = self.LeafAreaIndexAdjustment(cycle_len= cycle_len_irr, oLAI = self.LAI, aLAI = self.aLAI, bLAI = self.bLAI)

                    # Harvest Index adjustment for irrigated condition
                    if cycle_len_irr -self.aHI > self.bHI:
                        HI_irr = self.HI
                    elif cycle_len_irr - self.aHI <0:
                        HI_irr = 0
                    else:
                        HI_irr = self.HarvestIndexAdjustment(cycle_len = cycle_len_irr, oHI = self.HI, aHI = self.aHI, bHI = self.bHI)
                
                count_pixel_completed = count_pixel_completed + 1       
                # this allows handing leap and non-leap year differently. This is only relevant for monthly data because this value will be used in interpolations.
                # In case of daily data, length of vector will be taken as number of days in  a year.
                if leap_year:
                    days_in_year = 366
                else:
                    days_in_year = 365

                # extract daily climate data for particular location.

                minT_daily_point = self.minT_daily[i_row, i_col, :]
                maxT_daily_point = self.maxT_daily[i_row, i_col, :]
                meanT_daily_point = self.meanT_daily[i_row, i_col,:]
                shortRad_daily_point = self.shortRad_daily[i_row, i_col, :]
                wind2m_daily_point = self.wind2m_daily[i_row, i_col, :]
                totalPrec_daily_point = self.totalPrec_daily[i_row, i_col, :]
                rel_humidity_daily_point = self.rel_humidity_daily[i_row, i_col, :]
                pet_daily_point = self.pet_daily[i_row, i_col, :]   


                # Empty arrays that stores yield estimations and fc1 and fc2 of all cycles per particular location (pixel)
                yield_of_all_crop_cycles_rainfed = np.empty(0, dtype= np.float16)
                yield_of_all_crop_cycles_irrig = np.empty(0, dtype= np.float16)

                fc1_rain_lst = np.empty(0, dtype= np.float16)
                fc1_irr_lst = np.empty(0, dtype= np.float16)

                fc2_lst = np.empty(0, dtype= np.float16)


                """ Calculation of each individual day's yield for rainfed and irrigated conditions"""

                for i_cycle in range(start_doy-1, end_doy, step_doy):

                    """Check if the first day of a cycle meets minimum temperature requirement. If not, all outputs will be zero.
                        And iterates to next cycle."""
                    if (minT_daily_point[i_cycle]+maxT_daily_point[i_cycle])/2 < self.min_temp:
                        est_yield_moisture_limited = 0.
                        fc1_rain = 0.
                        fc1_irr =0.
                        fc2_value = 0.
                        est_yield_irrigated = 0.

                        yield_of_all_crop_cycles_rainfed = np.append(yield_of_all_crop_cycles_rainfed, 0.)
                        yield_of_all_crop_cycles_irrig = np.append(yield_of_all_crop_cycles_irrig, 0.)
                        fc1_rain_lst = np.append(fc1_rain_lst, 0.)
                        fc1_irr_lst = np.append(fc1_irr_lst, 0.)
                        fc2_lst = np.append(fc2_lst, 0.)
                        continue
                    
                    """Repeat the climate data two times and concatenate for computational convenience. If perennial, the cycle length
                            will be different for separate conditions"""
                    minT_daily_2year = np.tile(minT_daily_point, 2)
                    maxT_daily_2year = np.tile(maxT_daily_point, 2)
                    meanT_daily_2year = np.tile(meanT_daily_point, 2)
                    shortRad_daily_2year = np.tile(shortRad_daily_point, 2)
                    wind2m_daily_2year = np.tile(wind2m_daily_point,2)
                    totalPrec_daily_2year = np.tile(totalPrec_daily_point, 2)
                    pet_daily_2year = np.tile(pet_daily_point, 2)
                    meanT_daily_2year = np.tile(meanT_daily_point, 2)
                    
                    if cycle_len_rain in[-1,0]:
                        est_yield_moisture_limited = 0.
                        fc1_rain = 0.
                        fc2_value = 0.
                    else:
                        """ Time slicing tiled climate data with corresponding cycle lengths for rainfed and irrigated conditions"""
                        """For rainfed"""

                        # extract climate data within the season to pass in to calculation classes
                        minT_daily_season_rain = minT_daily_2year[i_cycle: i_cycle + int(cycle_len_rain)]
                        maxT_daily_season_rain = maxT_daily_2year[i_cycle: i_cycle + int(cycle_len_rain)]
                        meanT_daily_season_rain = meanT_daily_2year[i_cycle: i_cycle + int(cycle_len_rain)]
                        shortRad_daily_season_rain = shortRad_daily_2year[i_cycle: i_cycle+int(cycle_len_rain)]
                        pet_daily_season_rain = pet_daily_2year[i_cycle: i_cycle+int(cycle_len_rain)]
                        totalPrec_daily_season_rain = totalPrec_daily_2year[i_cycle: i_cycle+int(cycle_len_rain)]
                        wind_sp_daily_season_rain = wind2m_daily_2year[i_cycle: i_cycle+int(cycle_len_rain)]
                        
                        """Creating Thermal Screening object classes for perennial RAINFED conditions"""
                        obj_screening_rain = ThermalScreening.ThermalScreening()

                        """ For Perennials, 365 days of climate data will be used for Thermal Screening.
                            For Annuals, climate data within crop-specific cycle length will be used for Thermal Screening."""
                        if self.perennial:
                            obj_screening_rain.setClimateData(meanT_daily_2year[i_cycle: i_cycle+365], meanT_daily_2year[i_cycle: i_cycle+365])
                        else:
                            obj_screening_rain.setClimateData(meanT_daily_season_rain, meanT_daily_season_rain)
                        

                        if self.set_lgpt_screening:
                            obj_screening_rain.setLGPTScreening(no_lgpt=self.no_lgpt, optm_lgpt=self.optm_lgpt)

                        # TSUM Screening
                        if self.set_Tsum_screening:
                            obj_screening_rain.setTSumScreening(LnS=self.LnS, LsO=self.LsO, LO=self.LO, HnS=self.HnS, HsO=self.HsO, HO=self.HO)

                        # Crop-Specific Rule Screening
                        if self.setTypeBConstraint:
                            obj_screening_rain.applyTypeBConstraint(data=self.data, input_temp_profile=obj_screening_rain.tprofile, perennial_flag= self.perennial)

                        # Initial set up value for fc1 RAINFED
                        fc1_rain = 1.
                        fc1_rain = obj_screening_rain.getReductionFactor2()  # fc1 for rainfed condition
                        
                        if fc1_rain == 0.:
                            est_yield_moisture_limited = 0.
                            fc1_rain = 0.
                            fc2_value = 0.
                        else:
                            
                            if LAI_rain <=0 or HI_rain <= 0:
                                est_yield_rainfed = 0
                            else:

                                """If fc1 RAINFED IS NOT ZERO >>> BIOMASS RAINFED STARTS"""
                                obj_maxyield_rain = BioMassCalc.BioMassCalc(i_cycle+1, i_cycle+1+cycle_len_rain-1, self.latitude[i_row, i_col])
                                obj_maxyield_rain.setClimateData(minT_daily_season_rain, maxT_daily_season_rain, shortRad_daily_season_rain)
                                obj_maxyield_rain.setCropParameters(LAi_rain, HI_rain, self.legume, self.adaptability)
                                obj_maxyield_rain.calculateBioMass()
                                est_yield_rainfed = obj_maxyield_rain.calculateYield()

                            # reduce thermal screening factor
                            est_yield_rainfed = est_yield_rainfed * fc1_rain

                            """ For Annual RAINFED, crop water requirements are calculated in full procedures.
                                For Perennial RAINFED, procedures related with yield loss factors are omitted out.
                                """
                            fc2_value = 1.
                            
                            obj_cropwat = CropWatCalc.CropWatCalc(
                                i_cycle+1, i_cycle+1+cycle_len_rain, perennial_flag = self.perennial)
                            obj_cropwat.setClimateData(pet_daily_season_rain, totalPrec_daily_season_rain, 
                                                       wind_sp_daily_season_rain, minT_daily_season_rain, 
                                                       maxT_daily_season_rain)
                            
                            # check Sa is a raster or single value and extract Sa value accordingly
                            if len(np.array(self.Sa).shape) == 2:
                                Sa_temp = self.Sa[i_row, i_col]
                            else:
                                Sa_temp = self.Sa
                            obj_cropwat.setCropParameters(self.d_per, self.kc, self.kc_all, self.yloss_f,
                                                            self.yloss_f_all, est_yield_rainfed, self.D1, self.D2, Sa_temp, self.pc, self.plant_height)
                            est_yield_moisture_limited = obj_cropwat.calculateMoistureLimitedYield()

                            fc2_value = obj_cropwat.getfc2factormap()

                    yield_of_all_crop_cycles_rainfed = np.append(yield_of_all_crop_cycles_rainfed, est_yield_moisture_limited)
                    fc2_lst = np.append(fc2_lst, fc2_value)
                    fc1_rain_lst = np.append(fc1_rain_lst, fc1_rain)

                    """Error checking code snippet"""
                    if est_yield_moisture_limited == None or est_yield_moisture_limited == np.nan:
                        raise Exception('Crop Water Yield not returned. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
                    if fc2_value == None or fc2_value == np.nan:
                        raise Exception('fc2 value not returned. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
                    if len(fc1_rain_lst) != i_cycle+1 or fc1_rain == None or fc1_rain == np.nan:
                        raise Exception('Fc1 rain not properly appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
                    if len(yield_of_all_crop_cycles_rainfed) != i_cycle+1:
                        raise Exception('Rainfed yield list not properly appended') 
                    if len(fc2_lst) != i_cycle+1:
                        raise Exception('Fc2 list not appended properly. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))


                    ##########################
                    if cycle_len_irr in [-1, 0]:
                        est_yield_irrigated = 0.
                        fc1_irr = 0.
                    else:
                            
                        # extract climate data within the season to pass in to calculation classes
                        minT_daily_season_irr = minT_daily_2year[i_cycle: i_cycle +int(cycle_len_irr)]
                        maxT_daily_season_irr = maxT_daily_2year[i_cycle: i_cycle +int(cycle_len_irr)]
                        meanT_daily_season_irr = meanT_daily_2year[i_cycle: i_cycle +int(cycle_len_irr)]
                        meanT_daily_season_irr = meanT_daily_2year[i_cycle: i_cycle +int(cycle_len_irr)]
                        shortRad_daily_season_irr = shortRad_daily_2year[i_cycle: i_cycle+int(cycle_len_irr)]
                        pet_daily_season_irr = pet_daily_2year[i_cycle: i_cycle+int(cycle_len_irr)]
                        totalPrec_daily_season_irr = totalPrec_daily_2year[i_cycle: i_cycle+int(cycle_len_irr)]
                        
                    
                        """Creating Thermal Screening object classes for IRRIGATED conditions"""
                        obj_screening_irr = ThermalScreening.ThermalScreening()
                        
                        """ For Perennials, 365 days of climate data will be used for Thermal Screening.
                        For Annuals, climate data within crop-specific cycle length will be used for Thermal Screening."""
                        if self.perennial:
                            obj_screening_irr.setClimateData(meanT_daily_2year[i_cycle: i_cycle+365], meanT_daily_2year[i_cycle: i_cycle+365])
                        else:
                            obj_screening_irr.setClimateData(minT_daily_season_irr, maxT_daily_season_irr)

                        if self.set_lgpt_screening:
                            obj_screening_irr.setLGPTScreening(no_lgpt=self.no_lgpt, optm_lgpt=self.optm_lgpt)

                        # TSUM Screening
                        if self.set_Tsum_screening:
                            obj_screening_irr.setTSumScreening(LnS=self.LnS, LsO=self.LsO, LO=self.LO, HnS=self.HnS, HsO=self.HsO, HO=self.HO)

                        # Crop-Specific Rule Screening
                        if self.setTypeBConstraint:
                            obj_screening_irr.applyTypeBConstraint(data=self.data, input_temp_profile=obj_screening_irr.tprofile, perennial_flag= self.perennial)

                        # Initial set up value for fc1 RAINFED
                        fc1_irr = 1.

                        fc1_irr = obj_screening_irr.getReductionFactor2()  # fc1 for rainfed condition

                
                        if fc1_irr == 0.:
                            est_yield_irrigated = 0.
                        else:
                            if LAI_irr <=0 or HI_irr <= 0:
                                est_yield_irrigated = 0
                            else:
                                """If fc1 IRRIGATED IS NOT ZERO >>> BIOMASS IRRIGATED STARTS"""
                                obj_maxyield_irr = BioMassCalc.BioMassCalc(
                                    i_cycle+1, i_cycle+1+self.cycle_len_irr-1, self.latitude[i_row, i_col])
                                obj_maxyield_irr.setClimateData(
                                    minT_daily_season_irr, maxT_daily_season_irr, shortRad_daily_season_irr)
                                obj_maxyield_irr.setCropParameters(
                                    self.LAi_irr, self.HI_irr, self.legume, self.adaptability)
                                obj_maxyield_irr.calculateBioMass()
                                est_yield_irrigated = obj_maxyield_irr.calculateYield()
                            # reduce thermal screening factor
                            est_yield_irrigated = est_yield_irrigated * fc1_irr

                    yield_of_all_crop_cycles_irrig = np.append(yield_of_all_crop_cycles_irrig, est_yield_irrigated)
                    fc1_irr_lst = np.append(fc1_irr_lst, fc1_irr)

                    # Error raising
                    if est_yield_irrigated == None or est_yield_irrigated== np.nan:
                        raise Exception('Biomass Yield for irrigated not returned. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))

                    if len(yield_of_all_crop_cycles_irrig) != i_cycle+1:
                        raise Exception('Irrigated yield list not properly appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
                                
                    if len(fc1_irr_lst) != i_cycle+1 or fc1_irr == None or fc1_irr == np.nan:
                        raise Exception('Fc1 irr not properly appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
                    
                    if len(fc1_irr_lst)!= i_cycle+1 or fc1_rain == None or fc1_rain == np.nan:
                        raise Exception('Fc1 irr not properly appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))

                """Getting Maximum Attainable Yield from the list for irrigated and rainfed conditions and the Crop Calendar"""

                # get agro-climatic yield and crop calendar for IRRIGATED condition
                if np.logical_and(len(yield_of_all_crop_cycles_irrig) == len(fc1_irr_lst), len(yield_of_all_crop_cycles_irrig) == len(fc1_irr_lst)):

                    self.final_yield_irrig[i_row, i_col] = np.nanmax(yield_of_all_crop_cycles_irrig) # Maximum attainable yield

                    # Array index where maximum yield is obtained
                    i = np.where(yield_of_all_crop_cycles_irrig == np.max(yield_of_all_crop_cycles_irrig))[0][0] # index of maximum yield

                    self.crop_calender_irr[i_row, i_col] = int(i+1)*step_doy if self.final_yield_irrig[i_row, i_col] != 0 else 0 # Crop calendar for irrigated condition

                    self.fc1_irr[i_row, i_col] = fc1_irr_lst[i] # fc1 irrigated for the specific crop calendar DOY

                # get agro-climatic yield and crop calendar for RAINFED condition
                if np.logical_and(len(yield_of_all_crop_cycles_rainfed) == len(fc1_rain_lst), len(yield_of_all_crop_cycles_rainfed) == len(fc1_rain_lst)):
                    self.final_yield_rainfed[i_row, i_col] = np.max(yield_of_all_crop_cycles_rainfed) # Maximum attainable yield

                    i1 = np.where(yield_of_all_crop_cycles_rainfed == np.max(yield_of_all_crop_cycles_rainfed))[0][0] # index of maximum yield
                    
                    self.crop_calender_rain[i_row, i_col] = int(i1+1) * step_doy if self.final_yield_rainfed[i_row, i_col] != 0 else 0# Crop calendar for rainfed condition
                    
                    self.fc1_rain[i_row, i_col] = fc1_rain_lst[i1]
                    
                    # if not self.perennial:
                    self.fc2[i_row, i_col] = fc2_lst[i1]


                print('\rDone %: ' + str(round(count_pixel_completed / total*100, 2)), end='\r')
        
        print('\nSimulations Completed !')
    
    def simulateCropCycleOne(self, row, col, ccdi, ccdr, step_doy=1, leap_year=False):
        """Running the crop cycle calculation/simulation for a particular pixel location.

        Args:
            row (int): row index of a location
            col (int): column index of a location
            ccd (int): starting date you want to simulate
            step_doy (int, optional): Spacing (in days) between 2 adjacent crop simulations. Defaults to 1.
            leap_year (bool, optional): whether or not the simulating year is a leap year. Defaults to False.

        """        

        # just a counter to keep track of progress
        i_row = row
        i_col = col

        # GAEZ date starting date you want to investigate the intermediate values
        ccdi_idx = ccdi-1
        ccdr_idx = ccdr-1

        start_doy, end_doy =1, 365

        # this stores final result
        final_yield_rainfed = 0.
        final_yield_irrig = 0.
        crop_calender_irr = 0.
        crop_calender_rain = 0.
        fc2 = 0.
        fc1_rain = 0.
        fc1_irr = 0.

        LAI_rain = 0.
        HI_rain = 0.
        LAI_irr = 0.
        HI_irr = 0.

        # check current location (pixel) is outside of study area or not. if it's outside of study area goes to next location (pixel)
        # Those unsuitable
        if self.set_mask:
            if self.im_mask[i_row, i_col] == self.nodata_val:
                raise Exception('Pixel in the Mask Area.')

        # 2. Permafrost screening
        if self.set_Permafrost_screening:
            if np.logical_or(self.permafrost_class[i_row, i_col] == 1, self.permafrost_class[i_row, i_col] == 2):
                raise Exception('Pixel in permafrost area.')

        # Thermal Climate Screening
        if self.set_tclimate_screening:
            if self.t_climate[i_row, i_col] in self.no_t_climate:
                raise Exception('Pixel in unsuitable thermal climate.')
        
        """Cycle length checking for rainfed and irrigated annuals.
            Concerns with LGPt5 (irrigated) and LGP (rainfed)"""
        if not self.perennial:

            "Harvest Index and Leaf Area Index are not adjusted."
            LAI_rain = self.LAi
            HI_rain = self.HI

            LAI_irr = self.LAi
            HI_irr = self.HI
            
            # for irrigated conditions
            # In real simulation, this pixel would be omitted out
            if int(self.LGPT5[i_row, i_col]) < self.min_cycle_len:
                cycle_len_irr = 0
            else:
                cycle_len_irr = self.cycle_len
            
            # for rainfed condition
            # In real simulation, this pixel would be omitted out for 
            if int(self.LGP[i_row, i_col]) < self.min_cycle_len:
                cycle_len_rain = 0
            else:
                cycle_len_rain = self.cycle_len
            
        else:
            """Cycle length checking for rainfed and irrigated perennials.
            Concerns with LGPt5 (irrigated) and LGP (rainfed)"""

            """Adjustment of cycle length, LAI and HI for Perennials"""
            self.set_adjustment = True

            """ Adjustment for RAINFED conditions"""
            if int(self.LGP[i_row, i_col]) < self.min_cycle_len:
                cycle_len_rain = 0
                
            else:
                # effective cycle length will be our cycle length. The CYLeff should not be greater than CYCref.
                cycle_len_rain = self.cycle_len if (min(int(self.LGP[i_row, i_col]), self.max_cycle_len) >= self.cycle_len) else (min(int(self.LGP[i_row, i_col]), self.max_cycle_len))

                # Leaf Area Index adjustment for rainfed condition
                if (cycle_len_rain - self.aLAI > self.bLAI):
                    LAI_rain = self.LAi
                elif cycle_len_rain <= self.aLAI:
                    LAI_rain = 0
                else:
                    LAI_rain = self.LeafAreaIndexAdjustment(cycle_len= cycle_len_rain, oLAI = self.LAi, aLAI = self.aLAI, bLAI = self.bLAI)
                
                # Harvest Index adjustment for rainfed conditions
                if cycle_len_rain -self.aHI > self.bHI:
                    HI_rain = self.HI
                elif cycle_len_rain<= self.aHI:
                    HI_rain = 0
                else:
                    HI_rain = self.HarvestIndexAdjustment(cycle_len = cycle_len_rain, oHI = self.HI, aHI = self.aHI, bHI = self.bHI)
                
            """ Adjustment for IRRIGATED conditions"""
            """Use LGPT5 for minimum temperatures less than 8. Use LGPT10 for temperature greater than 8."""
            # effective cycle length will be our cycle length
            if self.min_temp <= 8:
                if int(self.LGPT5[i_row, i_col]) < self.min_cycle_len:
                    cycle_len_irr = 0
                else:
                    cycle_len_irr = min(int(self.LGPT5[i_row, i_col]), self.max_cycle_len)
            elif self.min_temp >8:
                if int(self.LGPT10[i_row, i_col]) < self.min_cycle_len:
                    cycle_len_irr = 0
                else:
                    cycle_len_irr = min(int(self.LGPT10[i_row, i_col]), self.max_cycle_len)
            
                cycle_len_irr = self.cycle_len if (cycle_len_irr >= self.cycle_len) else cycle_len_irr

            # Leaf Area Index adjustment for irrigated condition
                if cycle_len_irr - self.aLAI > self.bLAI:
                    LAI_irr = self.LAi
                elif cycle_len_irr <= self.aLAI:
                    LAI_irr = 0
                else:
                    LAI_irr = self.LeafAreaIndexAdjustment(cycle_len= cycle_len_irr, oLAI = self.LAi, aLAI = self.aLAI, bLAI = self.bLAI)

                # Harvest Index adjustment for irrigated condition
                if cycle_len_irr -self.aHI > self.bHI:
                    HI_irr = self.HI
                elif cycle_len_irr<= self.aHI:
                    HI_irr = 0
                else:
                    HI_irr = self.HarvestIndexAdjustment(cycle_len = cycle_len_irr, oHI = self.HI, aHI = self.aHI, bHI = self.bHI)
           
        # this allows handing leap and non-leap year differently. This is only relevant for monthly data because this value will be used in interpolations.
        # In case of daily data, length of vector will be taken as number of days in  a year.
        if leap_year:
            days_in_year = 366
        else:
            days_in_year = 365

        # extract daily climate data for particular location.

        minT_daily_point = self.minT_daily[i_row, i_col, :]
        maxT_daily_point = self.maxT_daily[i_row, i_col, :]
        meanT_daily_point = self.meanT_daily[i_row, i_col,:]
        shortRad_daily_point = self.shortRad_daily[i_row, i_col, :]
        wind2m_daily_point = self.wind2m_daily[i_row, i_col, :]
        totalPrec_daily_point = self.totalPrec_daily[i_row, i_col, :]
        rel_humidity_daily_point = self.rel_humidity_daily[i_row, i_col, :]
        pet_daily_point = self.pet_daily[i_row, i_col, :]   


        # Empty arrays that stores yield estimations and fc1 and fc2 of all cycles per particular location (pixel)
        yield_of_all_crop_cycles_rainfed = np.empty(0, dtype= np.float16)
        yield_of_all_crop_cycles_irrig = np.empty(0, dtype= np.float16)

        fc1_rain_lst = np.empty(0, dtype= np.float16)
        fc1_irr_lst = np.empty(0, dtype= np.float16)

        fc2_lst = np.empty(0, dtype= np.float16)


        """ Calculation of each individual day's yield for rainfed and irrigated conditions"""

        for i_cycle in range(start_doy-1, end_doy, step_doy):

            """Check if the first day of a cycle meets minimum temperature requirement. If not, all outputs will be zero.
                And iterates to next cycle."""
            if (minT_daily_point[i_cycle]+maxT_daily_point[i_cycle])/2 < self.min_temp:
                est_yield_moisture_limited = 0.
                fc1_rain = 0.
                fc1_irr =0.
                fc2_value = 0.
                est_yield_irrigated = 0.

                yield_of_all_crop_cycles_rainfed = np.append(yield_of_all_crop_cycles_rainfed, 0.)
                yield_of_all_crop_cycles_irrig = np.append(yield_of_all_crop_cycles_irrig, 0.)
                fc1_rain_lst = np.append(fc1_rain_lst, 0.)
                fc1_irr_lst = np.append(fc1_irr_lst, 0.)
                fc2_lst = np.append(fc2_lst, 0.)
                continue
            
            """Repeat the climate data two times and concatenate for computational convenience. If perennial, the cycle length
                    will be different for separate conditions"""
            minT_daily_2year = np.tile(minT_daily_point, 2)
            maxT_daily_2year = np.tile(maxT_daily_point, 2)
            meanT_daily_2year = np.tile(meanT_daily_point, 2)
            shortRad_daily_2year = np.tile(shortRad_daily_point, 2)
            wind2m_daily_2year = np.tile(wind2m_daily_point,2)
            totalPrec_daily_2year = np.tile(totalPrec_daily_point, 2)
            pet_daily_2year = np.tile(pet_daily_point, 2)
            meanT_daily_2year = np.tile(meanT_daily_point, 2)
            
            if cycle_len_rain in[-1,0]:
                est_yield_moisture_limited = 0.
                fc1_rain = 0.
                fc2_value = 0.
            else:
                """ Time slicing tiled climate data with corresponding cycle lengths for rainfed and irrigated conditions"""
                """For rainfed"""

                # extract climate data within the season to pass in to calculation classes
                minT_daily_season_rain = minT_daily_2year[i_cycle: i_cycle + int(cycle_len_rain)]
                maxT_daily_season_rain = maxT_daily_2year[i_cycle: i_cycle + int(cycle_len_rain)]
                meanT_daily_season_rain = meanT_daily_2year[i_cycle: i_cycle + int(cycle_len_rain)]
                shortRad_daily_season_rain = shortRad_daily_2year[i_cycle: i_cycle+int(cycle_len_rain)]
                pet_daily_season_rain = pet_daily_2year[i_cycle: i_cycle+int(cycle_len_rain)]
                totalPrec_daily_season_rain = totalPrec_daily_2year[i_cycle: i_cycle+int(cycle_len_rain)]
                wind_sp_daily_season_rain = wind2m_daily_2year[i_cycle: i_cycle+int(cycle_len_rain)]
                
                """Creating Thermal Screening object classes for perennial RAINFED conditions"""
                obj_screening_rain = ThermalScreening.ThermalScreening()
                """ For Perennials, 365 days of climate data will be used for Thermal Screening.
                    For Annuals, climate data within crop-specific cycle length will be used for Thermal Screening."""
                if self.perennial:
                    obj_screening_rain.setClimateData(meanT_daily_2year[i_cycle: i_cycle+365], meanT_daily_2year[i_cycle: i_cycle+365])
                else:
                    obj_screening_rain.setClimateData(meanT_daily_season_rain, meanT_daily_season_rain)
                

                if self.set_lgpt_screening:
                    obj_screening_rain.setLGPTScreening(no_lgpt=self.no_lgpt, optm_lgpt=self.optm_lgpt)

                # TSUM Screening
                if self.set_Tsum_screening:
                    obj_screening_rain.setTSumScreening(LnS=self.LnS, LsO=self.LsO, LO=self.LO, HnS=self.HnS, HsO=self.HsO, HO=self.HO)

                # Crop-Specific Rule Screening
                if self.setTypeBConstraint:
                    obj_screening_rain.applyTypeBConstraint(data=self.data, input_temp_profile=obj_screening_rain.tprofile, perennial_flag= self.perennial)

                # Initial set up value for fc1 RAINFED
                fc1_rain = 1.
                fc1_rain = obj_screening_rain.getReductionFactor2()  # fc1 for rainfed condition
                
                if fc1_rain == 0.:
                    est_yield_moisture_limited = 0.
                    fc1_rain = 0.
                    fc2_value = 0.
                else:
                    
                    if LAI_rain <=0 or HI_rain <= 0:
                        est_yield_rainfed = 0
                    else:

                        """If fc1 RAINFED IS NOT ZERO >>> BIOMASS RAINFED STARTS"""
                        obj_maxyield_rain = BioMassCalc.BioMassCalc(i_cycle+1, i_cycle+1+cycle_len_rain, self.latitude[i_row, i_col], leap_year)
                        obj_maxyield_rain.setClimateData(minT_daily_season_rain, maxT_daily_season_rain, shortRad_daily_season_rain)
                        obj_maxyield_rain.setCropParameters(LAI_rain, HI_rain, self.legume, self.adaptability)
                        obj_maxyield_rain.calculateBioMass()
                        est_yield_rainfed = obj_maxyield_rain.calculateYield()

                    # reduce thermal screening factor
                    est_yield_rainfed = est_yield_rainfed * fc1_rain

                    """ For Annual RAINFED, crop water requirements are calculated in full procedures.
                        For Perennial RAINFED, procedures related with yield loss factors are omitted out.
                        """
                    if est_yield_rainfed <=0:
                        est_yield_moisture_limited = 0.
                        fc2_value = 0.
                    
                    else:
                        obj_cropwat = CropWatCalc.CropWatCalc(
                            i_cycle+1, i_cycle+1+cycle_len_rain, perennial_flag = self.perennial)
                        obj_cropwat.setClimateData(pet_daily_season_rain, totalPrec_daily_season_rain, 
                                                    wind_sp_daily_season_rain, minT_daily_season_rain, 
                                                    maxT_daily_season_rain)
                        
                        # check Sa is a raster or single value and extract Sa value accordingly
                        if len(np.array(self.Sa).shape) == 2:
                            Sa_temp = self.Sa[i_row, i_col]
                        else:
                            Sa_temp = self.Sa
                        obj_cropwat.setCropParameters(self.d_per, self.kc, self.kc_all, self.yloss_f,
                                                        self.yloss_f_all, est_yield_rainfed, self.D1, self.D2, Sa_temp, self.pc, self.plant_height)
                        est_yield_moisture_limited = obj_cropwat.calculateMoistureLimitedYield()

                        fc2_value = obj_cropwat.getfc2factormap()

            yield_of_all_crop_cycles_rainfed = np.append(yield_of_all_crop_cycles_rainfed, est_yield_moisture_limited)
            fc2_lst = np.append(fc2_lst, fc2_value)
            fc1_rain_lst = np.append(fc1_rain_lst, fc1_rain)

            """Error checking code snippet"""
            if est_yield_moisture_limited == None or est_yield_moisture_limited == np.nan:
                raise Exception('Crop Water Yield not returned. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
            if fc2_value == None or fc2_value == np.nan:
                raise Exception('fc2 value not returned. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
            if len(fc1_rain_lst) != i_cycle+1 or fc1_rain == None or fc1_rain == np.nan:
                raise Exception('Fc1 rain not properly appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
            if len(yield_of_all_crop_cycles_rainfed) != i_cycle+1:
                raise Exception('Rainfed yield list not properly appended') 
            if len(fc2_lst) != i_cycle+1:
                raise Exception('Fc2 list not appended properly. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))


            ##########################
            if cycle_len_irr in [-1, 0]:
                est_yield_irrigated = 0.
                fc1_irr = 0.
            else:
                    
                # extract climate data within the season to pass in to calculation classes
                minT_daily_season_irr = minT_daily_2year[i_cycle: i_cycle +int(cycle_len_irr)]
                maxT_daily_season_irr = maxT_daily_2year[i_cycle: i_cycle +int(cycle_len_irr)]
                meanT_daily_season_irr = meanT_daily_2year[i_cycle: i_cycle +int(cycle_len_irr)]
                shortRad_daily_season_irr = shortRad_daily_2year[i_cycle: i_cycle+int(cycle_len_irr)]
                pet_daily_season_irr = pet_daily_2year[i_cycle: i_cycle+int(cycle_len_irr)]
                totalPrec_daily_season_irr = totalPrec_daily_2year[i_cycle: i_cycle+int(cycle_len_irr)]
                
            
                """Creating Thermal Screening object classes for IRRIGATED conditions"""
                obj_screening_irr = ThermalScreening.ThermalScreening()
                
                """ For Perennials, 365 days of climate data will be used for Thermal Screening.
                For Annuals, climate data within crop-specific cycle length will be used for Thermal Screening."""
                if self.perennial:
                    obj_screening_irr.setClimateData(meanT_daily_2year[i_cycle: i_cycle+365], meanT_daily_2year[i_cycle: i_cycle+365])
                else:
                    obj_screening_irr.setClimateData(meanT_daily_season_irr, meanT_daily_season_irr)

                if self.set_lgpt_screening:
                    obj_screening_irr.setLGPTScreening(no_lgpt=self.no_lgpt, optm_lgpt=self.optm_lgpt)

                # TSUM Screening
                if self.set_Tsum_screening:
                    obj_screening_irr.setTSumScreening(LnS=self.LnS, LsO=self.LsO, LO=self.LO, HnS=self.HnS, HsO=self.HsO, HO=self.HO)

                # Crop-Specific Rule Screening
                if self.setTypeBConstraint:
                    obj_screening_irr.applyTypeBConstraint(data=self.data, input_temp_profile=obj_screening_irr.tprofile, perennial_flag= self.perennial)

                # Initial set up value for fc1 RAINFED
                fc1_irr = 1.

                fc1_irr = obj_screening_irr.getReductionFactor2()  # fc1 for rainfed condition

        
                if fc1_irr == 0.:
                    est_yield_irrigated = 0.
                else:
                    if LAI_irr <=0 or HI_irr <= 0:
                        est_yield_irrigated = 0
                    else:
                        """If fc1 IRRIGATED IS NOT ZERO >>> BIOMASS IRRIGATED STARTS"""
                        obj_maxyield_irr = BioMassCalc.BioMassCalc(i_cycle+1, i_cycle+1+cycle_len_irr, self.latitude[i_row, i_col], leap_year)
                        obj_maxyield_irr.setClimateData(minT_daily_season_irr, maxT_daily_season_irr, shortRad_daily_season_irr)
                        obj_maxyield_irr.setCropParameters(LAI_irr, HI_irr, self.legume, self.adaptability)
                        obj_maxyield_irr.calculateBioMass()
                        est_yield_irrigated = obj_maxyield_irr.calculateYield()
                    # reduce thermal screening factor
                    est_yield_irrigated = est_yield_irrigated * fc1_irr

            yield_of_all_crop_cycles_irrig = np.append(yield_of_all_crop_cycles_irrig, est_yield_irrigated)
            fc1_irr_lst = np.append(fc1_irr_lst, fc1_irr)

            # Error raising
            if est_yield_irrigated == None or est_yield_irrigated== np.nan:
                raise Exception('Biomass Yield for irrigated not returned. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))

            if len(yield_of_all_crop_cycles_irrig) != i_cycle+1:
                raise Exception('Irrigated yield list not properly appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
                        
            if len(fc1_irr_lst) != i_cycle+1 or fc1_irr == None or fc1_irr == np.nan:
                raise Exception('Fc1 irr not properly appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
            
            if len(fc1_irr_lst)!= i_cycle+1 or fc1_rain == None or fc1_rain == np.nan:
                raise Exception('Fc1 irr not properly appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))

        """Getting Maximum Attainable Yield from the list for irrigated and rainfed conditions and the Crop Calendar"""

        # get agro-climatic yield and crop calendar for IRRIGATED condition
        if np.logical_and(len(yield_of_all_crop_cycles_irrig) == len(fc1_irr_lst), len(yield_of_all_crop_cycles_irrig) == len(fc1_irr_lst)):

            final_yield_irrig = np.nanmax(yield_of_all_crop_cycles_irrig) # Maximum attainable yield

            # Array index where maximum yield is obtained
            i = np.where(yield_of_all_crop_cycles_irrig == np.nanmax(yield_of_all_crop_cycles_irrig))[0][0] # index of maximum yield

            crop_calender_irr = (int(i+1)*step_doy) if final_yield_irrig != 0 else 0 # Crop calendar for irrigated condition

            fc1_irr = fc1_irr_lst[i] # fc1 irrigated for the specific crop calendar DOY

        # get agro-climatic yield and crop calendar for RAINFED condition
        if np.logical_and(len(yield_of_all_crop_cycles_rainfed) == len(fc1_rain_lst), len(yield_of_all_crop_cycles_rainfed) == len(fc1_rain_lst)):
            final_yield_rainfed = np.nanmax(yield_of_all_crop_cycles_rainfed) # Maximum attainable yield

            i1 = np.where(yield_of_all_crop_cycles_rainfed == np.max(yield_of_all_crop_cycles_rainfed))[0][0] # index of maximum yield
            
            crop_calender_rain = (int(i1+1) * step_doy) if final_yield_rainfed != 0 else 0 # Crop calendar for rainfed condition
            
            fc1_rain = fc1_rain_lst[i1]
            
            # if not self.perennial:
            fc2 = fc2_lst[i1]

        general = {
            'row': [i_row],
            'col': [i_col],
            'mask': [self.im_mask[i_row, i_col]],
            'permafrost': [self.permafrost_class[i_row, i_col]],
            'TClimate': [self.t_climate[i_row, i_col]],
            'perennial_flag': [self.perennial],
            'LGPT5':[self.LGPT5[i_row, i_col]],
            'LGPT10':[self.LGPT10[i_row, i_col]],
            'LGP':[self.LGP[i_row, i_col]],
            'elevation':[self.elevation[i_row, i_col]],
            'Latitude': [self.latitude[i_row, i_col]],
            'Minimum cycle length':[self.min_cycle_len],
            'Maximum cycle length': [self.max_cycle_len],
            'aLAI': [self.aLAI],
            'bLAI':[self.bLAI],
            'aHI':[self.aHI],
            'bHI':[self.bHI],
            'Reference_cycle_len':[self.cycle_len],
            'Effective cycle length rainfed':[cycle_len_rain],
            'Effective cycle length irrigated':[cycle_len_irr],
            'Original LAI rain':[self.LAi],
            'Original HI rain':[self.HI],
            'Original LAI irr':[self.LAi],
            'Original HI irr':[self.HI],
            'Adjusted LAI rain':[LAI_rain],
            'Adjusted HI rain':[HI_rain],
            'Adjusted LAI irr':[LAI_irr],
            'Adjusted HI irr':[HI_irr],
            }
        
        climate = {
            'min_temp(DegC)':minT_daily_point,
            'max_temp(DegC)':maxT_daily_point,
            'mean_temp(DegC)':meanT_daily_point,
            'shortrad(Wm-2)':shortRad_daily_point,
            'shortrad(MJ/m2/day)': (shortRad_daily_point * 3600 * 24)/1000000,
            'shortrad(calcm-2day-1)':shortRad_daily_point * 2.06362854686156,
            'windspeed(ms-1)':wind2m_daily_point,
            'precipitation(mmday-1)':totalPrec_daily_point,
            'rel_humid(decimal)':rel_humidity_daily_point,
            'ETo (mmday-1)':pet_daily_point
            }
        
        cycle = {
            'Cycles': np.arange(1,367) if leap_year else np.arange(1,366),
            'Rainfed Yield ': yield_of_all_crop_cycles_rainfed,
            'Irrigated Yield': yield_of_all_crop_cycles_irrig,
            'fc1_rain': fc1_rain_lst,
            'fc1_irr': fc1_irr_lst,
            'fc2': fc2_lst
        }
        final = {
            'Maximum Rainfed Yield': [final_yield_rainfed],
            'Maximum Irrigated Yield':[final_yield_irrig],
            'fc1_irr': [fc1_irr],
            'fc1_rain': [fc1_rain],
            'fc2': [fc2],
            'crop_calendar_rain':[float(crop_calender_rain)],
            'crop_calendar_irr':[float(crop_calender_irr)]
        }

        if LAI_irr <= 0 or HI_irr <=0:
            biomassi = {'Note': 'LAI_irr or HI_irr is zero. Simulation is not done.'}
        else:
            # extract climate data within the season to pass in to calculation classes
            minT_daily_season_irr = minT_daily_2year[ccdi_idx: ccdi_idx +int(cycle_len_irr)]
            maxT_daily_season_irr = maxT_daily_2year[ccdi_idx: ccdi_idx +int(cycle_len_irr)]
            meanT_daily_season_irr = meanT_daily_2year[ccdi_idx: ccdi_idx +int(cycle_len_irr)]
            shortRad_daily_season_irr = shortRad_daily_2year[ccdi_idx: ccdi_idx +int(cycle_len_irr)]
            pet_daily_season_irr = pet_daily_2year[ccdi_idx: ccdi_idx +int(cycle_len_irr)]
            totalPrec_daily_season_irr = totalPrec_daily_2year[ccdi_idx: ccdi_idx +int(cycle_len_irr)]

            # Biomass Calculation for irrigated condition (GAEZ specific starting date)
            bio = BioMassCalc.BioMassCalc(cycle_begin= ccdi_idx+1, cycle_end= ccdi_idx+1+int(cycle_len_irr), latitude= self.latitude[row, col])

            bio.setClimateData(min_temp= minT_daily_season_irr, 
                            max_temp= maxT_daily_season_irr, 
                            short_rad= shortRad_daily_season_irr)

            bio.setCropParameters(LAI= LAI_irr, HI= HI_irr, legume= self.legume, adaptability= self.adaptability)

            bio_lst = bio.biomassinter()

            biomassi = {
            'adaptability': [bio.adaptability +1],
            'legume': [bio.legume],
            'cycle_start': [bio.cycle_begin],
            'cycle_end': [bio.cycle_end],
            'LAI': [bio.LAi],
            'HI': [bio.HI],
            'Ac_mean': [bio_lst[0]],
            'Bc_mean': [bio_lst[1]],
            'Bo_mean': [bio_lst[2]],
            'meanT_mean': [bio_lst[3]],
            'dT_mean': [bio_lst[4]],
            'Rg':[bio_lst[5]],
            'f_day_clouded':[bio_lst[6]],
            'pm': [bio_lst[7]],
            'ct': [bio_lst[9]],
            'growth ratio(l)': [bio_lst[10]],
            'bgm': [bio_lst[11]],
            'Bn': [bio_lst[12]],
            'By': [np.round(bio_lst[12] * bio.HI, 0).astype(int)],
            'final irrigated yield':[np.round(bio_lst[12] * bio.HI * fc1_irr_lst[ccdi_idx], 0)]
            }
        
        if LAI_rain <= 0 or HI_rain <=0:
            biomassr = {'Note': 'LAI_rain or HI_rain is zero. Simulation is not done.'}
            cropwat = {'Note': 'LAI_rain or HI_rain is zero. Simulation is not done'}
        else:
            # extract climate data within the season to pass in to calculation classes
            minT_daily_season_rain = minT_daily_2year[ccdr_idx: ccdr_idx + int(cycle_len_rain)]
            maxT_daily_season_rain = maxT_daily_2year[ccdr_idx: ccdr_idx + int(cycle_len_rain)]
            meanT_daily_season_rain = meanT_daily_2year[ccdr_idx: ccdr_idx + int(cycle_len_rain)]
            shortRad_daily_season_rain = shortRad_daily_2year[ccdr_idx: ccdr_idx + int(cycle_len_rain)]
            pet_daily_season_rain = pet_daily_2year[ccdr_idx: ccdr_idx + int(cycle_len_rain)]
            totalPrec_daily_season_rain = totalPrec_daily_2year[ccdr_idx: ccdr_idx + int(cycle_len_rain)]
            wind_sp_daily_season_rain = wind2m_daily_2year[ccdr_idx: ccdr_idx + int(cycle_len_rain)]

            # Biomass Calculation for rainfed condition (GAEZ specific starting date)
            bio2 = BioMassCalc.BioMassCalc(cycle_begin= ccdr_idx+1, cycle_end= ccdr_idx+1+cycle_len_rain, latitude= self.latitude[row, col])

            bio2.setClimateData(min_temp= minT_daily_season_rain, 
                            max_temp= maxT_daily_season_rain, 
                            short_rad= shortRad_daily_season_rain)

            bio2.setCropParameters(LAI= LAI_rain, HI= HI_rain, legume= self.legume, adaptability= self.adaptability)

            bio_lst2 = bio2.biomassinter()

            # Crop Water Requirement calculation for rainfed condition (GAEZ specific starting date)
            cwat = CropWatCalc.CropWatCalc(cycle_begin= ccdr_idx+1, cycle_end= ccdr_idx+1+int(cycle_len_rain))
            cwat.setClimateData(pet_daily_season_rain,
                                totalPrec_daily_season_rain, 
                                wind_sp_daily_season_rain, 
                                minT_daily_season_rain, 
                                maxT_daily_season_rain)
            cwat.setCropParameters(stage_per=self.d_per,
                                kc=self.kc,
                                kc_all=self.kc_all,
                                yloss_f=self.yloss_f,
                                yloss_f_all= self.yloss_f_all,
                                est_yield= (np.round(bio_lst2[12] * bio2.HI, 0).astype(int)) * fc1_rain_lst[ccdr_idx],
                                D1=self.D1,
                                D2=self.D2,
                                Sa= self.Sa,
                                pc =self.pc,
                                height = self.plant_height)
            cwat_lst = cwat.getMoistureYieldNumba()

            biomassr ={
                'adaptability': [bio2.adaptability +1],
                'legume': [bio2.legume],
                'cycle_start': [bio2.cycle_begin],
                'cycle_end': [bio2.cycle_end],
                'LAI': [bio2.LAi],
                'HI': [bio2.HI],
                'Ac_mean': [bio_lst2[0]],
                'Bc_mean': [bio_lst2[1]],
                'Bo_mean': [bio_lst2[2]],
                'meanT_mean': [bio_lst2[3]],
                'dT_mean': [bio_lst2[4]],
                'Rg':[bio_lst2[5]],
                'f_day_clouded':[bio_lst2[6]],
                'pm': [bio_lst2[7]],
                'ct': [bio_lst2[9]],
                'growth ratio(l)': [bio_lst2[10]],
                'bgm': [bio_lst2[11]],
                'Bn': [bio_lst2[12]],
                'By': [np.round(bio_lst2[12] * bio2.HI, 0).astype(int)],
                'rainfed yield':[(np.round(bio_lst2[12] * bio2.HI, 0).astype(int)) * fc1_rain_lst[ccdr_idx]]
                }
            
            cropwat = {
                'cycle_start': [cwat.cycle_begin],
                'cycle_end': [cwat.cycle_end],
                'Original kc_initial': [self.kc[0]],
                'Original kc_reprodu':[self.kc[1]],
                'Original kc_maturity':[self.kc[2]],
                'Adjustedd kc_initial': [cwat.kc[0]],
                'Adjusted kc_reprodu':[cwat.kc[1]],
                'Adjusted kc_maturity':[cwat.kc[2]],
                'Soil Water Holding Capacity (Sa)':[cwat.Sa],
                'Soil Water Depletion Factor (pc)':[cwat.pc],
                'plant height': [cwat.height],
                'kc_all': [cwat.kc_all],
                'y_loss_init':[cwat.yloss_f[0]],
                'y_loss_vege':[cwat.yloss_f[1]],
                'y_loss_repro':[cwat.yloss_f[2]],
                'y_loss_maturity': [cwat.yloss_f[3]],
                'y_loss_all': [cwat.yloss_f_all],
                'potential yield': [(np.round(bio_lst2[12] * bio2.HI, 0).astype(int)) * fc1_rain_lst[ccdr_idx]],
                'root_depth_start': [cwat.D1],
                'root_depth_end':[cwat.D2],
                'Sa': [cwat.Sa],
                'pc': [cwat.pc],
                'fc2': [cwat_lst[1]],
                'water_lim_yield': [cwat_lst[0]]
            }

        ### temperature profile calculation (Irrigated)
        obj_i = ThermalScreening.ThermalScreening()
        if self.perennial:
            obj_i.setClimateData(meanT_daily_2year[ccdi_idx: ccdi_idx+365], meanT_daily_2year[ccdi_idx: ccdi_idx+365])
        else:
            obj_i.setClimateData(meanT_daily_2year[ccdi_idx: ccdi_idx+int(cycle_len_irr)], meanT_daily_2year[ccdi_idx: ccdi_idx+int(cycle_len_irr)])
        
        # Crop-Specific Rule Screening
        if self.setTypeBConstraint:
            obj_i.applyTypeBConstraint(data=self.data, input_temp_profile=obj_i.tprofile, perennial_flag= self.perennial)

        temp_profile_i = obj_i.getReductionFactor2() # 

        obj_i.setTypeBConstraint = False

        # TSUM Screening
        if self.set_Tsum_screening:
            obj_i.setTSumScreening(LnS=self.LnS, LsO=self.LsO, LO=self.LO, HnS=self.HnS, HsO=self.HsO, HO=self.HO)
        TSUM_i = obj_i.getReductionFactor2() #


        ### Temperature profile calculation(Rainfed)
        obj_r = ThermalScreening.ThermalScreening()
        if self.perennial:
            obj_r.setClimateData(meanT_daily_2year[ccdr_idx: ccdr_idx+365], meanT_daily_2year[ccdr_idx: ccdr_idx+365])
        else:
            obj_r.setClimateData(meanT_daily_2year[ccdr_idx: ccdr_idx+int(cycle_len_rain)], meanT_daily_2year[ccdr_idx: ccdr_idx+int(cycle_len_rain)])
        
        # Crop-Specific Rule Screening
        if self.setTypeBConstraint:
            obj_r.applyTypeBConstraint(data=self.data, input_temp_profile=obj_r.tprofile, perennial_flag= self.perennial)

        temp_profile_r = obj_r.getReductionFactor2() # 
        obj_r.setTypeBConstraint = False

        # TSUM Screening
        if self.set_Tsum_screening:
            obj_r.setTSumScreening(LnS=self.LnS, LsO=self.LsO, LO=self.LO, HnS=self.HnS, HsO=self.HsO, HO=self.HO)
        TSUM_r = obj_r.getReductionFactor2() # 

        ts_i = {
        'cycle_begin':[ccdi_idx+1],
        'cycle_end':[ccdi_idx+1+int(cycle_len_irr)],
        'cycle_len_TSUM':[obj_i.meanT_daily_tsum.shape[0]],
        'cycle_len_Tprofile':[obj_i.meanT_daily_tp.shape[0]],
        'TSUM0':[obj_i.tsum0],
        'TProfile':[obj_i.tprofile],
        'LnS':[self.LnS],
        'LsO':[self.LsO],
        'LO':[self.LO],
        'HO':[self.HO],
        'HsO':[self.HsO],
        'HnS':[self.HnS],
        'fc1_TSUM0':[TSUM_i],
        'fc1_Tprofile':[temp_profile_i],
        'final_fc1_irr':[np.nanmin([TSUM_i, temp_profile_i])]
        }

        ts_r = {
        'cycle_begin':[ccdr_idx+1],
        'cycle_end':[ccdr_idx+1+int(cycle_len_rain)],
        'cycle_len_TSUM':[obj_r.meanT_daily_tsum.shape[0]],
        'cycle_len_Tprofile':[obj_r.meanT_daily_tp.shape[0]],
        'TSUM0':[obj_r.tsum0],
        'TProfile':[obj_r.tprofile],
        'LnS':[self.LnS],
        'LsO':[self.LsO],
        'LO':[self.LO],
        'HO':[self.HO],
        'HsO':[self.HsO],
        'HnS':[self.HnS],
        'fc1_TSUM0':[TSUM_r],
        'fc1_Tprofile':[temp_profile_r],
        'final_fc1_rain':[np.nanmin([TSUM_r, temp_profile_r])]
        }

        print('\nSimulations Completed !')
        return [general ,climate, cycle, final ,biomassi, biomassr ,cropwat, ts_i, ts_r]
