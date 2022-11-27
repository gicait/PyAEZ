"""
PyAEZ version 2.0.0 (November 2022)
This ClimateRegime Class read/load and calculates the agro-climatic indicators
required to run PyAEZ.  
2021: N. Lakmal Deshapriya
2022: Swun Wunna Htet, K. Boonma
"""

import numpy as np
import UtilitiesCalc, ETOCalc, LGPCalc
from time import sleep
np.seterr(divide='ignore', invalid='ignore') # ignore "divide by zero" or "divide by NaN" warning

# Initiate ClimateRegime Class instance
class ClimateRegime(object):
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
        self.latitude = UtilitiesCalc.UtilitiesCalc().generateLatitudeMap(lat_min, lat_max, self.im_height, self.im_width)
        
        
    
    def setStudyAreaMask(self, admin_mask, no_data_value):
        """Set clipping mask of the area of interest (optional)

        Args:
            admin_mask (2D NumPy/Binary): mask to extract only region of interest
            no_data_value (int): pixels with this value will be omitted during PyAEZ calculations
        """    
        self.im_mask = admin_mask
        self.nodata_val = no_data_value
        self.set_mask = True

  

    def setMonthlyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity):
        """Load MONTHLY climate data into the Class and calculate the Reference Evapotranspiration (ETo)

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
        wind_speed[wind_speed < 0] = 0
        self.meanT_daily = np.zeros((self.im_height, self.im_width, 365))
        self.totalPrec_daily = np.zeros((self.im_height, self.im_width, 365))
        self.pet_daily = np.zeros((self.im_height, self.im_width, 365))
        self.minT_daily = np.zeros((self.im_height, self.im_width, 365))
        self.maxT_daily = np.zeros((self.im_height, self.im_width, 365))


        # Interpolate monthly to daily data
        obj_utilities = UtilitiesCalc.UtilitiesCalc()

        meanT_monthly = (min_temp+max_temp)/2

        for i_row in range(self.im_height):
            for i_col in range(self.im_width):

                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue

                self.meanT_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(meanT_monthly[i_row, i_col,:], 1, 365)
                self.totalPrec_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(precipitation[i_row, i_col,:], 1, 365, no_minus_values=True)
                self.minT_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(min_temp[i_row, i_col,:], 1, 365)
                self.maxT_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(max_temp[i_row, i_col,:], 1, 365)
                radiation_daily = obj_utilities.interpMonthlyToDaily(short_rad[i_row, i_col,:], 1, 365, no_minus_values=True)
                wind_daily = obj_utilities.interpMonthlyToDaily(wind_speed[i_row, i_col,:], 1, 365, no_minus_values=True)
                rel_humidity_daily = obj_utilities.interpMonthlyToDaily(rel_humidity[i_row, i_col,:], 1, 365, no_minus_values=True)

                # calculation of reference evapotranspiration (ETo)
                obj_eto = ETOCalc.ETOCalc(1, 365, self.latitude[i_row, i_col], self.elevation[i_row, i_col])
                shortrad_daily_MJm2day = (radiation_daily*3600*24)/1000000 # convert w/m2 to MJ/m2/day
                obj_eto.setClimateData(self.minT_daily[i_row, i_col, :], self.maxT_daily[i_row, i_col, :], wind_daily, shortrad_daily_MJm2day, rel_humidity_daily)
                self.pet_daily[i_row, i_col, :] = obj_eto.calculateETO()
                
        # Sea-level adjusted mean temperature
        self.meanT_daily_sealevel = self.meanT_daily + np.tile(np.reshape(self.elevation/100*0.55, (self.im_height,self.im_width,1)), (1,1,365))
        # P over PET ratio(to eliminate nan in the result, nan is replaced with zero)
        self.P_by_PET_daily = np.nan_to_num(self.totalPrec_daily / self.pet_daily)

    def setDailyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity):
        """Load DAILY climate data into the Class and calculate the Reference Evapotranspiration (ETo)

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
        short_rad[short_rad<0]=0
        # max wind is 20 m/sec
        wind_speed[wind_speed<0]=0
        
        
        self.meanT_daily = np.zeros((self.im_height, self.im_width, 365))
        self.totalPrec_daily = np.zeros((self.im_height, self.im_width, 365))
        self.pet_daily = np.zeros((self.im_height, self.im_width, 365))
        self.maxT_daily = max_temp
        

        for i_row in range(self.im_height):
            for i_col in range(self.im_width):

                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue

                self.meanT_daily[i_row, i_col, :] = 0.5*(min_temp[i_row, i_col, :]+max_temp[i_row, i_col, :])
                self.totalPrec_daily[i_row, i_col, :] = precipitation[i_row, i_col, :]
                
                # calculation of reference evapotranspiration (ETo)
                obj_eto = ETOCalc.ETOCalc(1, 365, self.latitude[i_row, i_col], self.elevation[i_row, i_col])
                shortrad_daily_MJm2day = (short_rad[i_row, i_col, :]*3600*24)/1000000 # convert w/m2 to MJ/m2/day
                obj_eto.setClimateData(min_temp[i_row, i_col, :], max_temp[i_row, i_col, :], wind_speed[i_row, i_col, :], shortrad_daily_MJm2day, rel_humidity[i_row, i_col, :])
                self.pet_daily[i_row, i_col, :] = obj_eto.calculateETO()
                
        # sea level temperature
        self.meanT_daily_sealevel = self.meanT_daily + np.tile(np.reshape(self.elevation/100*0.55, (self.im_height,self.im_width,1)), (1,1,365))
        # P over PET ratio (to eliminate nan in the result, nan is replaced with zero)
        self.P_by_PET_daily = np.nan_to_num(self.totalPrec_daily / self.pet_daily)

    def getThermalClimate(self):
        """Classification of rainfall and temperature seasonality into thermal climate classes

        Returns:
            2D NumPy: Thermal Climate classification
        """        
        # Note that currently, this thermal climate is designed only for the northern hemisphere, southern hemisphere is not implemented yet.
        thermal_climate = np.zeros((self.im_height, self.im_width), dtype= np.int8)

        for i_row in range(self.im_height):
            for i_col in range(self.im_width):

                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                
                # converting daily to monthly
                obj_utilities = UtilitiesCalc.UtilitiesCalc()
                meanT_monthly_sealevel = obj_utilities.averageDailyToMonthly(self.meanT_daily_sealevel[i_row,i_col,:])
                meanT_monthly = obj_utilities.averageDailyToMonthly(self.meanT_daily[i_row,i_col,:])
                P_by_PET_monthly = obj_utilities.averageDailyToMonthly(self.P_by_PET_daily[i_row,i_col,:])

                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                # Tropics
                if np.min(meanT_monthly_sealevel) > 18 and np.max(meanT_monthly)-np.min(meanT_monthly) < 15:
                    if np.mean(meanT_monthly) > 20:
                        thermal_climate[i_row, i_col] = 1  # Tropical lowland
                    else:
                        thermal_climate[i_row, i_col] = 2  # Tropical highland
                        
                # SubTropic
                elif np.min(meanT_monthly_sealevel) > 5 and np.sum(meanT_monthly_sealevel >= 10) >= 8:
                    if np.sum(self.totalPrec_daily[i_row,i_col,:]) < 250:
                        # 'Subtropics Low Rainfall
                        thermal_climate[i_row,i_col] = 3
                    elif self.latitude[i_row,i_col]>=0 and np.mean(P_by_PET_monthly[3:9]) >= np.mean([P_by_PET_monthly[9:12],P_by_PET_monthly[0:3]]):
                        # Subtropics Summer Rainfall
                        thermal_climate[i_row,i_col] = 4
                    elif self.latitude[i_row,i_col]<0 and np.mean(P_by_PET_monthly[3:9]) <= np.mean([P_by_PET_monthly[9:12],P_by_PET_monthly[0:3]]):
                        # Subtropics Summer Rainfall
                        thermal_climate[i_row,i_col] = 4
                    elif self.latitude[i_row,i_col]>=0 and np.mean(P_by_PET_monthly[3:9]) <= np.mean([P_by_PET_monthly[9:12],P_by_PET_monthly[0:3]]): 
                        # Subtropics Winter Rainfall
                        thermal_climate[i_row,i_col] = 5
                    elif self.latitude[i_row,i_col]<0 and np.mean(P_by_PET_monthly[3:9]) >= np.mean([P_by_PET_monthly[9:12],P_by_PET_monthly[0:3]]):
                        # Subtropics Winter Rainfall
                        thermal_climate[i_row,i_col] = 5

                # Temperate
                elif np.sum(meanT_monthly_sealevel >= 10) >= 4:
                    if np.max(meanT_monthly)-np.min(meanT_monthly) <= 20:
                        # Oceanic Temperate
                        thermal_climate[i_row, i_col] = 6
                    elif np.max(meanT_monthly)-np.min(meanT_monthly) <= 35:
                        # Sub-Continental Temperate
                        thermal_climate[i_row, i_col] = 7
                    else:
                        # Continental Temperate
                        thermal_climate[i_row, i_col] = 8

                elif np.sum(meanT_monthly_sealevel >= 10) >= 1:
                    # Boreal
                    if np.max(meanT_monthly)-np.min(meanT_monthly) <= 20:
                        # Oceanic Boreal
                        thermal_climate[i_row, i_col] = 9
                    elif np.max(meanT_monthly)-np.min(meanT_monthly) <= 35:
                        # Sub-Continental Boreal
                        thermal_climate[i_row, i_col] = 10
                    else:
                        # Continental Boreal
                        thermal_climate[i_row, i_col] = 11
                else:
                    # Arctic
                    thermal_climate[i_row, i_col] = 12
        if self.set_mask:
            return np.ma.masked_where(self.im_mask == 0, thermal_climate)
        else:
            return thermal_climate
    

    def getThermalZone(self):
        """The thermal zone is classified based on actual temperature which reflects 
        on the temperature regimes of major thermal climates

        Returns:
            2D NumPy: Thermal Zones classification
        """        
        thermal_zone = np.zeros((self.im_height, self.im_width))
    
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                
                obj_utilities = UtilitiesCalc.UtilitiesCalc()
                meanT_monthly = obj_utilities.averageDailyToMonthly(self.meanT_daily[i_row, i_col, :])
                meanT_monthly_sealevel =  obj_utilities.averageDailyToMonthly(self.meanT_daily_sealevel[i_row, i_col, :])
    
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
    
                if np.min(meanT_monthly_sealevel) >= 18 and np.max(meanT_monthly)-np.min(meanT_monthly) < 15:
                    if np.mean(meanT_monthly) > 20:
                        thermal_zone[i_row,i_col] = 1 # Tropics Warm
                    else:
                        thermal_zone[i_row,i_col] = 2 # Tropics cool/cold/very cold
                
                elif np.min(meanT_monthly_sealevel) > 5 and np.sum(meanT_monthly_sealevel > 10) >= 8:
                    if np.sum(meanT_monthly<5) >= 1 and np.sum(meanT_monthly>10) >= 4:
                        thermal_zone[i_row,i_col] =  4 # Subtropics, cool
                    elif np.sum(meanT_monthly<5) >= 1 and np.sum(meanT_monthly>10) >= 1:
                        thermal_zone[i_row,i_col] =  5 # Subtropics, cold
                    elif np.sum(meanT_monthly<10) == 12:
                        thermal_zone[i_row,i_col] =  6 # Subtropics, very cold
                    else:
                        thermal_zone[i_row,i_col] =  3 # Subtropics, warm/mod. cool
    
                elif np.sum(meanT_monthly_sealevel >= 10) >= 4:
                    if np.sum(meanT_monthly<5) >= 1 and np.sum(meanT_monthly>10) >= 4:
                        thermal_zone[i_row,i_col] =  7 # Temperate, cool
                    elif np.sum(meanT_monthly<5) >= 1 and np.sum(meanT_monthly>10) >= 1:
                        thermal_zone[i_row,i_col] =  8 # Temperate, cold
                    elif np.sum(meanT_monthly<10) == 12:
                        thermal_zone[i_row,i_col] =  9 # Temperate, very cold
    
                elif np.sum(meanT_monthly_sealevel >= 10) >= 1:
                    if np.sum(meanT_monthly<5) >= 1 and np.sum(meanT_monthly>10) >= 1:
                        thermal_zone[i_row,i_col] = 10 # Boreal, cold
                    elif np.sum(meanT_monthly<10) == 12:
                        thermal_zone[i_row,i_col] = 11 # Boreal, very cold
                else:
                        thermal_zone[i_row,i_col] = 12 # Arctic
    
        if self.set_mask:
            return np.ma.masked_where(self.im_mask == 0, thermal_zone)
        else:
            return thermal_zone

    def getThermalLGP0(self):
        """Calculate Thermal Length of Growing Period (LGPt) with 
        temperature threshold of 0 degree Celcius

        Returns:
            2D numpy: The accumulated number of days with daily mean 
                      temperature is above 0 degree Celcius
        """        
        # Adding interpolation to the dataset
        interp_daily_temp = np.zeros((self.im_height, self.im_width, 365))
        
        days = np.arange(1,366)
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                temp_1D = self.meanT_daily[i_row, i_col, :]
                # Creating quadratic spline fit to smoothen the time series along time dimension
                quad_spl = np.poly1d(np.polyfit(days, temp_1D,5))
                interp_daily_temp[i_row, i_col, :] = quad_spl(days)
    
        lgpt0 = np.sum(interp_daily_temp>=0, axis=2)
        if self.set_mask:
            lgpt0 = np.ma.masked_where(self.im_mask == 0, lgpt0)
        
        self.lgpt0=lgpt0.copy()
        return lgpt0


    def getThermalLGP5(self):
        """Calculate Thermal Length of Growing Period (LGPt) with 
        temperature threshold of 5 degree Celcius

        Returns:
            2D numpy: The accumulated number of days with daily mean 
                      temperature is above 5 degree Celcius
        """          
        # Adding interpolation to the dataset
        interp_daily_temp = np.zeros((self.im_height, self.im_width, 365))
        days = np.arange(1,366)
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                temp_1D = self.meanT_daily[i_row, i_col, :]
                # Creating quadratic spline fit to smoothen the time series along time dimension
                quad_spl = np.poly1d(np.polyfit(days, temp_1D,5))
                interp_daily_temp[i_row, i_col, :] = quad_spl(days)
       
        lgpt5 = np.sum(interp_daily_temp >= 5, axis=2)
        if self.set_mask:
            lgpt5 = np.ma.masked_where(self.im_mask == 0, lgpt5)

        self.lgpt5 = lgpt5.copy()
        return lgpt5

    def getThermalLGP10(self):
        """Calculate Thermal Length of Growing Period (LGPt) with
        temperature threshold of 10 degree Celcius

        Returns:
            2D numpy: The accumulated number of days with daily mean
                      temperature is above 10 degree Celcius
        """
        # Adding interpolation to the dataset
        interp_daily_temp = np.zeros((self.im_height, self.im_width, 365))
        days = np.arange(1,366)
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                temp_1D = self.meanT_daily[i_row, i_col, :]
                # Creating quadratic spline fit to smoothen the time series along time dimension
                quad_spl = np.poly1d(np.polyfit(days, temp_1D,5))
                interp_daily_temp[i_row, i_col, :] = quad_spl(days)

        lgpt10 = np.sum(interp_daily_temp >= 10, axis=2)
        if self.set_mask:
            lgpt10 = np.ma.masked_where(self.im_mask == 0, lgpt10)

        self.lgpt10 = lgpt10.copy()
        return lgpt10

    def getTemperatureSum0(self):
        """Calculate temperature summation at temperature threshold 
        of 0 degree Celcius

        Returns:
            2D numpy: Accumulative daily average temperature (Ta) for days
                      when Ta is above the thresholds of 0 degree Celcius
        """
        tempT = self.meanT_daily.copy()
        tempT[tempT<0] = 0
        tsum0 = np.round(np.sum(tempT, axis=2), decimals = 0) 
        # masking
        if self.set_mask:
            tsum0 = np.ma.masked_where(self.im_mask == 0, tsum0)
        return tsum0

    def getTemperatureSum5(self):
        """Calculate temperature summation at temperature threshold 
        of 5 degree Celcius

        Returns:
            2D numpy: Accumulative daily average temperature (Ta) for days
                      when Ta is above the thresholds of 5 degree Celcius
        """
        tempT = self.meanT_daily.copy()
        tempT[tempT<5] = 0
        tsum5 = np.round(np.sum(tempT, axis=2), decimals = 0) 
        # masking
        if self.set_mask: 
            tsum5 = np.ma.masked_where(self.im_mask == 0, tsum5)
        return tsum5
        

    def getTemperatureSum10(self):
        """Calculate temperature summation at temperature threshold 
        of 10 degree Celcius

        Returns:
            2D numpy: Accumulative daily average temperature (Ta) for days
                      when Ta is above the thresholds of 10 degree Celcius
        """
        tempT = self.meanT_daily.copy()
        tempT[tempT<10] = 0
        tsum10 = np.round(np.sum(tempT, axis=2), decimals = 0) 
        # masking
        if self.set_mask: 
            tsum10 = np.ma.masked_where(self.im_mask == 0, tsum10)
        return tsum10

    def getTemperatureProfile(self):
        """Classification of temperature ranges for temperature profile

        Returns:
            2D NumPy: 18 2D arrays [A1-A9, B1-B9] correspond to each Temperature Profile class [days]
        """        
        # Smoothening the temperature curve
        interp_daily_temp = np.zeros((self.im_height, self.im_width, 365))
        days = np.arange(1,366)
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                temp_1D = self.meanT_daily[i_row, i_col, :]
                # Creating quadratic spline fit to smoothen the time series along time dimension
                quad_spl = np.poly1d(np.polyfit(days, temp_1D, 5))
                interp_daily_temp[i_row, i_col, :] = quad_spl(days)
        
        # we will use the interpolated temperature time series to decide and count
        meanT_daily_add1day = np.concatenate((interp_daily_temp, interp_daily_temp[:,:,0:1]), axis=-1)
        meanT_first = meanT_daily_add1day[:,:,:-1]
        meanT_diff = meanT_daily_add1day[:,:,1:] - meanT_daily_add1day[:,:,:-1]

        A9 = np.sum( np.logical_and(meanT_diff>0, meanT_first<-5), axis=2 )
        A8 = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=-5, meanT_first<0)), axis=2 )
        A7 = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=0, meanT_first<5)), axis=2 )
        A6 = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=5, meanT_first<10)), axis=2 )
        A5 = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=10, meanT_first<15)), axis=2 )
        A4 = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=15, meanT_first<20)), axis=2 )
        A3 = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=20, meanT_first<25)), axis=2 )
        A2 = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=25, meanT_first<30)), axis=2 )
        A1 = np.sum( np.logical_and(meanT_diff>0, meanT_first>=30), axis=2 )

        B9 = np.sum( np.logical_and(meanT_diff<0, meanT_first<-5), axis=2 )
        B8 = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=-5, meanT_first<0)), axis=2 )
        B7 = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=0, meanT_first<5)), axis=2 )
        B6 = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=5, meanT_first<10)), axis=2 )
        B5 = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=10, meanT_first<15)), axis=2 )
        B4 = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=15, meanT_first<20)), axis=2 )
        B3 = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=20, meanT_first<25)), axis=2 )
        B2 = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=25, meanT_first<30)), axis=2 )
        B1 = np.sum( np.logical_and(meanT_diff<0, meanT_first>=30), axis=2 )

        if self.set_mask:
            return [np.ma.masked_where(self.im_mask == 0, A1),
                    np.ma.masked_where(self.im_mask == 0, A2),
                    np.ma.masked_where(self.im_mask == 0, A3),
                    np.ma.masked_where(self.im_mask == 0, A4),
                    np.ma.masked_where(self.im_mask == 0, A5),
                    np.ma.masked_where(self.im_mask == 0, A6),
                    np.ma.masked_where(self.im_mask == 0, A7),
                    np.ma.masked_where(self.im_mask == 0, A8),
                    np.ma.masked_where(self.im_mask == 0, A9),
                    np.ma.masked_where(self.im_mask == 0, B1),
                    np.ma.masked_where(self.im_mask == 0, B2),
                    np.ma.masked_where(self.im_mask == 0, B3),
                    np.ma.masked_where(self.im_mask == 0, B4),
                    np.ma.masked_where(self.im_mask == 0, B5),
                    np.ma.masked_where(self.im_mask == 0, B6),
                    np.ma.masked_where(self.im_mask == 0, B7),
                    np.ma.masked_where(self.im_mask == 0, B8),
                    np.ma.masked_where(self.im_mask == 0, B9)]
        else:
            return [A1, A2, A3, A4, A5, A6, A7, A8, A9, B1, B2, B3, B4, B5, B6, B7, B8, B9]


    def getLGP(self, Sa=100., D=1.):
        """Calculate length of growing period (LGP)

        Args:
            Sa (float, optional): Available soil moisture holding capacity [mm/m]. Defaults to 100..
            D (float, optional): Rooting depth. Defaults to 1..

        Returns:
           2D NumPy: Length of Growing Period
        """        
        #============================
        kc_list = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 0.5, 0.2])
        #============================
        Txsnm = 0.  # Txsnm - snow melt temperature threshold
        Fsnm = 5.5  # Fsnm - snow melting coefficient
        Sb_old = 0.
        Wb_old = 0.
        #============================
        Tx365 = self.maxT_daily.copy()
        Ta365 = self.meanT_daily.copy()
        Pcp365 = self.totalPrec_daily.copy()
        self.Eto365 = self.pet_daily  # Eto
        self.Etm365 = np.zeros(Tx365.shape)
        self.Eta365 = np.zeros(Tx365.shape)
        self.Sb365 = np.zeros(Tx365.shape)
        self.Wb365 = np.zeros(Tx365.shape)
        self.Wx365 = np.zeros(Tx365.shape)
        self.kc365 = np.zeros(Tx365.shape)
        #============================
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):

                lgpt5_point = self.lgpt5[i_row, i_col]

                totalPrec_monthly = UtilitiesCalc.UtilitiesCalc().averageDailyToMonthly(self.totalPrec_daily[i_row, i_col, :])
                meanT_daily_point = Ta365[i_row, i_col, :]
                meanT_daily_new, istart0, istart1 = LGPCalc.rainPeak(
                    totalPrec_monthly, meanT_daily_point, lgpt5_point)
                #----------------------------------
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue

                for doy in range(0, 365):
                    p = LGPCalc.psh(
                        0., self.Eto365[i_row, i_col, doy])
                    fromT0 = LGPCalc.isfromt0(meanT_daily_new, doy)
                    Eta_new, Etm_new, Wb_new, Wx_new, Sb_new, kc_new = LGPCalc.EtaCalc(
                        np.float64(Tx365[i_row, i_col, doy]), np.float64(
                            Ta365[i_row, i_col, doy]),
                        np.float64(Pcp365[i_row, i_col, doy]), Txsnm, Fsnm, np.float64(
                            self.Eto365[i_row, i_col, doy]),
                        Wb_old, Sb_old, doy, fromT0, istart0, istart1,
                        Sa, D, p, kc_list, lgpt5_point)

                    self.Eta365[i_row, i_col, doy] = Eta_new
                    self.Etm365[i_row, i_col, doy] = Etm_new
                    self.Wb365[i_row, i_col, doy] = Wb_new
                    self.Wx365[i_row, i_col, doy] = Wx_new
                    self.Sb365[i_row, i_col, doy] = Sb_new
                    self.kc365[i_row, i_col, doy] = kc_new

                    Wb_old = Wb_new
                    Sb_old = Sb_new
                    # Wx_old = Wx_new
        lgp = np.sum((self.Eta365/self.Etm365) >= 0.4, axis=2)
        
        if self.set_mask:
            return np.ma.masked_where(self.im_mask==0, lgp)
        else:
            return lgp
  
    def getLGPClassified(self, lgp): # Original PyAEZ source code
        """This function calculates the classification of moisture regime using LGP.

        Args:
            lgp (2D NumPy): Length of Growing Period

        Returns:
            2D NumPy: Classified Length of Growing Period
        """        
        # 

        lgp_class = np.zeros(lgp.shape)

        lgp_class[lgp>=365] = 7 # Per-humid
        lgp_class[np.logical_and(lgp>=270, lgp<365)] = 6 # Humid
        lgp_class[np.logical_and(lgp>=180, lgp<270)] = 5 # Sub-humid
        lgp_class[np.logical_and(lgp>=120, lgp<180)] = 4 # Moist semi-arid
        lgp_class[np.logical_and(lgp>=60, lgp<120)] = 3 # Dry semi-arid
        lgp_class[np.logical_and(lgp>0, lgp<60)] = 2 # Arid
        lgp_class[lgp<=0] = 1 # Hyper-arid

        if self.set_mask:
            return np.ma.masked_where(self.im_mask == 0, lgp_class)
        else:
            return lgp_class
        
        
    def getLGPEquivalent(self): 
        """Calculate the Equivalent LGP 

        Returns:
            2D NumPy: LGP Equivalent 
        """        
        moisture_index = np.sum(self.totalPrec_daily, axis=2)/np.sum(self.pet_daily, axis=2)
        # moisture_index = np.mean(self.totalPrec_daily/self.pet_daily, axis = 2)

        lgp_equv = 14.0 + 293.66*moisture_index - 61.25*moisture_index*moisture_index
        lgp_equv[ moisture_index > 2.4 ] = 366

        if self.set_mask:
            return np.ma.masked_where(self.im_mask == 0, lgp_equv)
        else:
            return lgp_equv


    
    def getMultiCroppingZones(self, t_climate, lgp, lgp_t5, lgp_t10, ts_t0, ts_t10):
        """
        This function refers to the assessment of multiple cropping potential
        across the area through matching both growth cycle and temperature
        requirements for individual suitable crops with time avaiability of 
        crop growth. The logic considers crop suitability for rainfed and 
        irrigated conditions.

        Args:
        ----------
        t_climate : a 2-D numpy array
            Thermal Climate.
        lgp : a 2-D numpy array
            Length of Growing Period.
        lgp_t5 : a 2-D numpy array
            Thermal growing period in days with mean daily temperatures above 5 degree Celsius.
        lgp_t10 : a 2-D numpy array
            Thermal growing period in days with mean daily temperatures above 10 degree Celsius.
        ts_t10 : a 2-D numpy array
            Accumulated temperature (degree-days) on days when mean daily temperature is greater or equal to 10 degree Celsius.
        ts_t0 : a 2-D numpy array
            Accumulated temperature (degree-days) on days when mean daily temperature is greater or equal to 0 degree Celsius.
        tsg_t5 : a 2-D numpy array
            Accumulated temperature on growing period days when mean daily temperature is greater or equal to 5 degree Celsius.
        tsg_t10 : a 2-D numpy array
            Accumulated temperature on growing period days when mean daily temperature is greater or equal to 10 degree Celsius.

        Returns
        -------
        A list of two 2-D numpy arrays. The first array refers to multi-cropping
        zone for rainfed condition, and the second refers to multi-cropping zone
        for irrigated condition.

        """
        
        # Definition of multi-cropping classes (Reference: GAEZ v4 Model Documentation)
        # 1 => A Zone (Zone of no cropping, too cold or too dry for rainfed crops)
        # 2 => B Zone (Zone of single cropping)
        # 3 => C Zone (Zone of limited double cropping, relay cropping; single wetland rice may be possible)
        # 4 => D Zone (Zone of double cropping; sequential double cropping including wetland rice is not possible)
        # 5 => E Zone (Zone of double cropping with rice; sequential double cropping with one wetland rice crop is possible)
        # 6 => F Zone (Zone of double rice cropping or limited triple cropping; may partly involve relay cropping; a third crop is not possible in case of two wetland rice crops)
        # 7 => G Zone (Zone of triple cropping; sequential cropping of three short-cycle crops; two wetland rice crops are possible)
        # 8 => H Zone (Zone of triple rice cropping; sequential cropping of three wetland rice crops is possible)
        
        # defining the constant arrays for rainfed and irrigated conditions, all pixel values start with 1
        multi_crop_rain = np.ones((self.im_height, self.im_width)) # all values started with Zone A
        multi_crop_irr = np.ones((self.im_height, self.im_width)) # all vauels starts with Zone A
        
        ts_g_t5 = np.ones((self.im_height, self.im_width))
        ts_g_t10 = np.ones((self.im_height, self.im_width))
        
        # Calculation of Accumulated temperature during the growing period at specific temperature thresholds: 5 and 10 degree Celsius
        
        for i_r in range(self.im_height):
            for i_c in range(self.im_width):

                if self.set_mask:
                    if self.im_mask[i_r, i_c] == self.nodata_val:
                        continue
                    
                temp_1D = self.meanT_daily[i_r, i_c, :]
                days = np.arange(0,365)
                
                deg = 5 # order of polynomical fit
                
                # creating the function of polyfit
                polyfit = np.poly1d(np.polyfit(days,temp_1D,deg))
                
                # getting the interpolated value at each DOY
                interp_daily_temp = polyfit(days)
                
                # Getting the start and end day of vegetative period
                # The crop growth requires minimum temperature of at least 5 deg Celsius
                # If not, the first DOY and the lst DOY of a year will be considered
                try:
                    veg_period = days[interp_daily_temp >=5]
                    start_veg = veg_period[0]
                    end_veg = veg_period[-1]
                except:
                    start_veg = 0
                    end_veg = 364
                
                # Slicing the temperature within the vegetative period
                interp_meanT_veg_T5 = interp_daily_temp[start_veg:end_veg]
                interp_meanT_veg_T10 =  interp_daily_temp[start_veg:end_veg] *1
                
                # Removing the temperature of 5 and 10 deg Celsius thresholds
                interp_meanT_veg_T5[interp_meanT_veg_T5 < 5] = 0
                interp_meanT_veg_T10[interp_meanT_veg_T10 <10] = 0
                
                # Calculation of Accumulated temperatures during growing period
                ts_g_t5[i_r, i_c] = np.sum(interp_meanT_veg_T5)
                ts_g_t10[i_r, i_c] = np.sum(interp_meanT_veg_T10)


        """Rainfed conditions"""
        """Lowland tropic climate"""
    
        zone_B = np.all([t_climate ==1, lgp>=45, lgp_t5>=120, lgp_t10>=90, ts_t0>=1600, ts_t10>=1200], axis=0)
        
        # three criteria for zone C conditions
        zone_C1 = np.all([t_climate ==1, lgp>=220, lgp_t5>=220, lgp_t10>=120, ts_t0>=5500, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        zone_C2 = np.all([t_climate ==1, lgp>=200, lgp_t5>=200, lgp_t10>=120, ts_t0>=6400, ts_g_t5>=3200, ts_g_t10>=2700], axis=0) # lgp_t5 is 200 instead of 210
        zone_C3 = np.all([t_climate ==1, lgp>=180, lgp_t5>=200, lgp_t10>=120, ts_t0>=7200, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        
        zone_C = np.logical_or.reduce([zone_C1==True, zone_C2==True, zone_C3==True], axis = 0)
        
        # three criteria for zone D conditions
        zone_D1 = np.all([t_climate ==1, lgp>=270, lgp_t5>=270, lgp_t10>=165, ts_t0>=5500, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_D2 = np.all([t_climate ==1, lgp>=240, lgp_t5>=240, lgp_t10>=165, ts_t0>=6400, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_D3 = np.all([t_climate ==1, lgp>=210, lgp_t5>=240, lgp_t10>=165, ts_t0>=7200, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        
        zone_D = np.logical_or.reduce([zone_D1==True, zone_D2==True, zone_D3==True], axis = 0)
        
        zone_F = np.all([t_climate ==1, lgp>=300, lgp_t5>=300, lgp_t10>=240, ts_t0>=7200, ts_g_t5>=5100, ts_g_t10>=4800], axis=0) # no criteria for ts_t10 in GAEZ
        zone_H = np.all([t_climate ==1, lgp>=360, lgp_t5>=360, lgp_t10>=360, ts_t0>=7200, ts_t10>=7000], axis=0) # lgp_t10 changed to 360
        
        
        """Other thermal climates"""
        zone_B_other = np.all([t_climate!= 1, lgp>=45, lgp_t5>=120, lgp_t10>=90, ts_t0>=1600, ts_t10>=1200], axis=0)
        zone_C_other = np.all([t_climate!= 1, lgp>=180, lgp_t5>=200, lgp_t10>=120, ts_t0>=3600, ts_t10>=3000, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        zone_D_other = np.all([t_climate!= 1, lgp>=210, lgp_t5>=240, lgp_t10>=165, ts_t0>=4500, ts_t10>=3600, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_E_other = np.all([t_climate!= 1, lgp>=240, lgp_t5>=270, lgp_t10>=180, ts_t0>=4800, ts_t10>=4500, ts_g_t5>=4300, ts_g_t10>=4000], axis=0)
        zone_F_other = np.all([t_climate!= 1, lgp>=300, lgp_t5>=300, lgp_t10>=240, ts_t0>=5400, ts_t10>=5100, ts_g_t5>=5100, ts_g_t10>=4800], axis=0)
        zone_G_other = np.all([t_climate!= 1, lgp>=330, lgp_t5>=330, lgp_t10>=270, ts_t0>=5700, ts_t10>=5500], axis=0)
        zone_H_other = np.all([t_climate!= 1, lgp>=360, lgp_t5>=360, lgp_t10>=330, ts_t0>=7200, ts_t10>=7000], axis=0)
        

        multi_crop_rain[np.logical_or.reduce([zone_B == True, zone_B_other == True], axis = 0)]= 2
        multi_crop_rain[np.logical_or.reduce([zone_C == True, zone_C_other == True], axis = 0)]= 3
        multi_crop_rain[np.logical_or.reduce([zone_D == True, zone_D_other == True], axis = 0)]= 4
        multi_crop_rain[zone_E_other == True]= 5
        multi_crop_rain[np.logical_or.reduce([zone_F == True, zone_F_other == True], axis = 0)]= 6
        multi_crop_rain[np.logical_or.reduce([zone_H == True, zone_H_other == True], axis = 0)]= 8
        
        

        
        ###########################################################################################################################
        """Irrigated conditions"""
        """Lowland tropic climate"""
        zone_B = np.all([t_climate ==1, lgp_t5>=120, lgp_t10>=90, ts_t0>=1600, ts_t10>=1200], axis=0)
        
        # three criteria for zone C conditions
        zone_C1 = np.all([t_climate ==1, lgp_t5>=220, lgp_t10>=120, ts_t0>=5500, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        zone_C2 = np.all([t_climate ==1, lgp_t5>=200, lgp_t10>=120, ts_t0>=6400, ts_g_t5>=3200, ts_g_t10>=2700], axis=0) # lgp_t5 is 200 instead of 210
        zone_C3 = np.all([t_climate ==1, lgp_t5>=200, lgp_t10>=120, ts_t0>=7200, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        
        zone_C = np.logical_or.reduce([zone_C1==True, zone_C2==True, zone_C3==True], axis = 0)
        
        # three criteria for zone D conditions
        zone_D1 = np.all([t_climate ==1, lgp_t5>=270, lgp_t10>=165, ts_t0>=5500, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_D2 = np.all([t_climate ==1, lgp_t5>=240, lgp_t10>=165, ts_t0>=6400, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_D3 = np.all([t_climate ==1, lgp_t5>=240, lgp_t10>=165, ts_t0>=7200, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        
        zone_D = np.logical_or.reduce([zone_D1==True, zone_D2==True, zone_D3==True], axis = 0)
        
        zone_F = np.all([t_climate ==1, lgp_t5>=300, lgp_t10>=240, ts_t0>=7200, ts_g_t5>=5100, ts_g_t10>=4800], axis=0) # no criteria for ts_t10 in GAEZ
        zone_H = np.all([t_climate ==1, lgp_t5>=360, lgp_t10>=360, ts_t0>=7200, ts_t10>=7000], axis=0) # lgp_t10 changed to 360
        
        
        """Other thermal climates"""
        zone_B_other = np.all([t_climate!= 1, lgp_t5>=120, lgp_t10>=90, ts_t0>=1600, ts_t10>=1200], axis=0)
        zone_C_other = np.all([t_climate!= 1, lgp_t5>=200, lgp_t10>=120, ts_t0>=3600, ts_t10>=3000, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        zone_D_other = np.all([t_climate!= 1, lgp_t5>=240, lgp_t10>=165, ts_t0>=4500, ts_t10>=3600, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_E_other = np.all([t_climate!= 1, lgp_t5>=270, lgp_t10>=180, ts_t0>=4800, ts_t10>=4500, ts_g_t5>=4300, ts_g_t10>=4000], axis=0)
        zone_F_other = np.all([t_climate!= 1, lgp_t5>=300, lgp_t10>=240, ts_t0>=5400, ts_t10>=5100, ts_g_t5>=5100, ts_g_t10>=4800], axis=0)
        zone_G_other = np.all([t_climate!= 1, lgp_t5>=330, lgp_t10>=270, ts_t0>=5700, ts_t10>=5500], axis=0)
        zone_H_other = np.all([t_climate!= 1, lgp_t5>=360, lgp_t10>=330, ts_t0>=7200, ts_t10>=7000], axis=0)
        
        
        multi_crop_irr[np.logical_or.reduce([zone_B == True, zone_B_other == True], axis = 0)]= 2
        multi_crop_irr[np.logical_or.reduce([zone_C == True, zone_C_other == True], axis = 0)]= 3
        multi_crop_irr[np.logical_or.reduce([zone_D == True, zone_D_other == True], axis = 0)]= 4
        multi_crop_irr[zone_E_other == True]= 5
        multi_crop_irr[np.logical_or.reduce([zone_F == True, zone_F_other == True], axis = 0)]= 6
        multi_crop_irr[np.logical_or.reduce([zone_F == True, zone_F_other == True], axis = 0)]= 6
        multi_crop_irr[np.logical_or.reduce([zone_H == True, zone_H_other == True], axis = 0)]= 8
        
        if self.set_mask:
            return [np.ma.masked_where(self.im_mask == 0, multi_crop_rain), np.ma.masked_where(self.im_mask == 0, multi_crop_irr)]
        else:
            return [multi_crop_rain, multi_crop_irr]
    
    def TZoneFallowRequirement(self, tzone):
        """
        The function calculates the temperature for fallow requirements which 
        requires thermal zone to classify. If mask is on, the function will
        mask out pixels by the mask layer. (NEW FUNCTION)

        Args:
        tzone : a 2-D numpy array
            THERMAL ZONE.

        Returns:
        A 2-D numpy array, corresponding to thermal zone for fallow requirement.

        """

        # the algorithm needs to calculate the annual mean temperature.
        tzonefallow = np.zeros((self.im_height, self.im_width), dtype= int)
        annual_Tmean = np.mean(self.meanT_daily, axis = 2)
        obj_utilities = UtilitiesCalc.UtilitiesCalc()

        # thermal zone class definitions for fallow requirement
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):

                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                # Checking tropics thermal zone
                if tzone[i_row, i_col] == 1 or tzone[i_row, i_col] == 2:
                    
                    # Class 1: tropics, mean annual T > 25 deg C
                    if annual_Tmean[i_row, i_col] > 25:
                        tzonefallow[i_row, i_col] = 1
                    
                    # Class 2: tropics, mean annual T 20-25 deg C
                    elif annual_Tmean[i_row, i_col] > 20:
                        tzonefallow[i_row, i_col] = 2
                    
                    # Class 3: tropics, mean annual T 15-20 deg C
                    elif annual_Tmean[i_row, i_col] > 15:
                        tzonefallow[i_row, i_col] = 3
                    
                    # Class 4: tropics, mean annual T < 15 deg C
                    else:
                        tzonefallow[i_row, i_col] = 4
                
                # Checking the non-tropical zones
                else:
                    meanT_monthly = obj_utilities.averageDailyToMonthly(self.meanT_daily[i_row, i_col, :])
                    # Class 5: mean T of the warmest month > 20 deg C
                    if np.max(meanT_monthly) > 20:
                        tzonefallow[i_row, i_col] = 5
                        
                    else:
                        tzonefallow[i_row, i_col] = 6
                            
        if self.set_mask:
            return np.ma.masked_where(self.im_mask == 0, tzonefallow)
        else:
            return tzonefallow
    
  
    
   
    
    def AirFrostIndexandPermafrostEvaluation(self):
        """
        The function calculates the air frost index which is used for evaluation of 
        occurrence of continuous or discontinuous permafrost condtions executed in 
        GAEZ v4. Two outputs of numerical air frost index and classified reference
        permafrost zones are returned. If mask layer is inserted, the function will
        automatically mask user-defined pixels out of the calculation 

        Returns:
        air_frost_index/permafrost : a python list: [air frost number, permafrost classes]

        """
        fi = np.zeros((self.im_height, self.im_width), dtype=float)
        permafrost = np.zeros((self.im_height, self.im_width), dtype=int)
        ddt = np.zeros((self.im_height, self.im_width), dtype=float) # thawing index
        ddf = np.zeros((self.im_height, self.im_width), dtype=float) # freezing index
        meanT_gt_0 = self.meanT_daily.copy()
        meanT_le_0 = self.meanT_daily.copy()
        
        meanT_gt_0[meanT_gt_0 <=0] = 0 # removing all negative temperatures for summation
        meanT_le_0[meanT_gt_0 >0] = 0 # removing all positive temperatures for summation 
        ddt = np.sum(meanT_gt_0, axis = 2)
        ddf = - np.sum(meanT_le_0, axis = 2)  
        fi = np.sqrt(ddf)/(np.sqrt(ddf) + np.sqrt(ddt)) 
        # now, we will classify the permafrost zones (Reference: GAEZ v4 model documentation: Pg35 -37)
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue         
                # Continuous Permafrost Class
                if fi[i_row, i_col]> 0.625:
                    permafrost[i_row, i_col] = 1
                
                # Discontinuous Permafrost Class
                if fi[i_row, i_col]> 0.57 and fi[i_row, i_col]< 0.625:
                    permafrost[i_row, i_col] = 2
                
                # Sporadic Permafrost Class
                if fi[i_row, i_col]> 0.495 and fi[i_row, i_col]< 0.57:
                    permafrost[i_row, i_col] = 3
                
                # No Permafrost Class
                if fi[i_row, i_col]< 0.495:
                    permafrost[i_row, i_col] = 4
        # to remove the division by zero, the nan values will be converted into
        fi = np.nan_to_num(fi)

        if self.set_mask:
            return [np.ma.masked_where(self.im_mask == 0, fi), np.ma.masked_where(self.im_mask == 0, permafrost)]
        else:
            return [fi, permafrost]
        
  

    
    def AEZClassification(self, tclimate, lgp, lgp_equv, lgpt_5, soil_terrain_lulc, permafrost):
        """The AEZ inventory combines spatial layers of thermal and moisture regimes 
        with broad categories of soil/terrain qualities.

        Args:
            tclimate (2D NumPy): Thermal Climate classes
            lgp (2D NumPy): Length of Growing Period
            lgp_equv (2D NumPy): LGP Equivalent
            lgpt_5 (2D NumPy): Thermal LGP of Ta>5˚C
            soil_terrain_lulc (2D NumPy): soil/terrain/special land cover classes (8 classes)
            permafrost (2D NumPy): Permafrost classes

        Returns:
           2D NumPy: 57 classes of AEZ
        """        
        
    #1st step: reclassifying the existing 12 classes of thermal climate into 6 major thermal climate.
    # Class 1: Tropics, lowland
    # Class 2: Tropics, highland
    # Class 3: Subtropics
    # Class 4: Temperate Climate
    # Class 5: Boreal Climate
    # Class 6: Arctic Climate
    
        aez_tclimate = np.zeros((self.im_height, self.im_width), dtype = int)
        
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                
                # keep the tropics lowland
                if tclimate[i_row, i_col] == 1:
                    aez_tclimate[i_row, i_col] = 1
                
                # keep the tropics highland
                if tclimate[i_row, i_col] == 2:
                    aez_tclimate[i_row, i_col] = 2
                
                # grouping all the subtropics climates into a single class 3
                if tclimate[i_row, i_col] == 3:
                    aez_tclimate[i_row, i_col] = 3
                
                if tclimate[i_row, i_col] == 4:
                    aez_tclimate[i_row, i_col] = 3
                
                if tclimate[i_row, i_col] == 5:
                    aez_tclimate[i_row, i_col] = 3
                
                # grouping all the temperate classes into a single class 4
                if tclimate[i_row, i_col] == 6:
                    aez_tclimate[i_row, i_col] = 4
                
                if tclimate[i_row, i_col] == 7:
                    aez_tclimate[i_row, i_col] = 4
                    
                if tclimate[i_row, i_col] == 8:
                    aez_tclimate[i_row, i_col] = 4
                    
                # grouping all the boreal classes into a single class 5
                if tclimate[i_row, i_col] == 9:
                    aez_tclimate[i_row, i_col] = 5
                
                if tclimate[i_row, i_col] == 10:
                    aez_tclimate[i_row, i_col] = 5
                
                if tclimate[i_row, i_col] == 11:
                    aez_tclimate[i_row, i_col] = 5
                
                # changing the arctic class into class 6
                if tclimate[i_row, i_col] == 12:
                    aez_tclimate[i_row, i_col] = 6
        
        # 2nd Step: Classification of Thermal Zones
        aez_tzone = np.zeros((self.im_height, self.im_width), dtype = int)
        
        obj_utilities = UtilitiesCalc.UtilitiesCalc()
        
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                
                meanT_monthly =  obj_utilities.averageDailyToMonthly(self.meanT_daily[i_row, i_col, :])
                # one conditional parameter for temperature accumulation 
                temp_acc_10deg = self.meanT_daily[i_row, i_col, :]
                temp_acc_10deg[temp_acc_10deg < 10] = 0
                
                # Warm Tzone (TZ1)
                if np.logical_or(aez_tclimate[i_row, i_col] == 1, aez_tclimate[i_row, i_col]==3) == True and np.sum(meanT_monthly >= 10) == 12 and np.mean(self.meanT_daily[i_row, i_col, :])>= 20:
                    aez_tzone[i_row, i_col] = 1
                
                # Moderately cool Tzone (TZ2)
                elif np.logical_or(aez_tclimate[i_row, i_col] == 2, aez_tclimate[i_row, i_col]==3) == True and np.sum(meanT_monthly >= 5) == 12 and np.sum(meanT_monthly >=10) >= 8:
                    aez_tzone[i_row, i_col] = 2
                
                # Moderate Tzone (TZ3)
                elif aez_tclimate[i_row, i_col] == 4 and np.sum(meanT_monthly >= 10) >=5 and np.sum(self.meanT_daily[i_row, i_col, :]>20) >= 75 and np.sum(temp_acc_10deg) > 3000:
                    aez_tzone[i_row, i_col] = 3
                
                # Cool Tzone (TZ4)
                elif np.logical_or.reduce((aez_tclimate[i_row, i_col] == 1, aez_tclimate[i_row, i_col] == 2, aez_tclimate[i_row, i_col] == 3, aez_tclimate[i_row, i_col] == 4)) == True and np.sum(meanT_monthly >= 10) >= 4 and np.mean(self.meanT_daily[i_row, i_col, :])>= 0:
                    aez_tzone[i_row, i_col] = 4
                
                # Cold Tzone (TZ5)
                elif np.logical_or.reduce((aez_tclimate[i_row, i_col] == 1, aez_tclimate[i_row, i_col] == 2, aez_tclimate[i_row, i_col] == 3, aez_tclimate[i_row, i_col] == 4, aez_tclimate[i_row, i_col] == 5)) == True and np.sum(meanT_monthly >= 10) <= 3 and np.mean(self.meanT_daily[i_row, i_col, :])>= 0:
                    aez_tzone[i_row, i_col] = 5
                
                # Very cold Tzone (TZ6)
                # else if np.sum(meanT_monthly_sealevel < 10) == 12 or np.mean(self.meanT_daily_sealevel[i_row, i_col, :]) < 0:
                elif np.sum(meanT_monthly < 10) <= 12 and np.mean(self.meanT_daily[i_row, i_col, :])< 0:
                    aez_tzone[i_row, i_col] = 6
        
        
        # 3rd Step: Creation of Temperature Regime Classes
        # Temperature Regime Class Definition
        # 1 = Tropics, lowland (TRC1)
        # 2 = Tropics, highland (TRC2)
        # 3 = Subtropics, warm (TRC3)
        # 4 = Subtropics, moderately cool (TRC4)
        # 5 = Subtropics, cool (TRC5)
        # 6 = Temperate, moderate (TRC6)
        # 7 = Temperate, cool (TRC7)
        # 8 = Boreal, cold, no continuous or discontinuous occurrence of permafrost (TRC8)
        # 9 = Boreal, cold, with continuous or discontinuous occurrence of permafrost (TRC9)
        # 10 = Arctic, very cold (TRC10)
        # aez_temp_regime = np.zeros((self.im_height, self.im_width), dtype = int)
        
        # Tropics, lowland (TRC1)
        # aez_temp_regime[np.all([aez_tclimate == 1, aez_tzone==1], axis = 0)] = 1
        
        # Tropics, highland (TRC2)
        # aez_temp_regime[np.all([aez_tclimate == 2, np.logical_or(aez_tzone ==2, aez_tzone == 4)], axis = 0)] = 2
        
        # Subtropics, warm (TRC3)
        # aez_temp_regime[np.all([aez_tclimate == 3, aez_tzone ==1], axis = 0)] = 3
        
        # Sub-tropics, moderately cool (TRC4)
        # aez_temp_regime[np.all([aez_tclimate == 3, aez_tzone ==2], axis = 0)] = 4
        
        # Sub-tropics, cool (TRC5)
        # aez_temp_regime[np.all([aez_tclimate == 3, aez_tzone ==4], axis = 0)] = 5
        
        # Temperate, moderate (TRC6)
        # aez_temp_regime[np.all([aez_tclimate == 4, aez_tzone ==3], axis = 0)] = 6
        
        # Temperate, cool (TRC7)
        # aez_temp_regime[np.all([aez_tclimate == 4, aez_tzone == 4], axis = 0)] = 7
        
        # Boreal, cold (no occurrence of continuous or discontinuous permafrost classes) I (TCR8)
        # aez_temp_regime[np.all([np.logical_or.reduce(np.array((aez_tclimate == 2, aez_tclimate == 3, aez_tclimate == 4, aez_tclimate == 5))), aez_tzone ==5], axis = 0)] = 8
        
        # Boreal, cold (occurrence of continuous or discontinuous permafrost classes) II (TCR9) # need to add permafrost evaluation
        # aez_temp_regime[np.all([np.logical_or.reduce(np.array((aez_tclimate == 2, aez_tclimate == 3, aez_tclimate == 4, aez_tclimate == 5))), aez_tzone ==5], axis = 0)] = 9
        
        # Arctic, very cold
        # aez_temp_regime[np.all([np.logical_or.reduce(np.array((aez_tclimate == 2, aez_tclimate == 3, aez_tclimate == 4, aez_tclimate == 5))), aez_tzone ==6], axis = 0)] = 10
        
        # another way of classification of thermal regime (going pixel by pixel)
        
        aez_temp_regime = np.zeros((self.im_height, self.im_width))

        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                if aez_tclimate[i_row, i_col] == 1 and aez_tzone[i_row, i_col]==1:
                    aez_temp_regime[i_row, i_col]=1 # Tropics, lowland

                elif aez_tclimate[i_row, i_col] == 2 and np.logical_or(aez_tzone[i_row, i_col] == 2, aez_tzone[i_row, i_col] == 4) == True:
                    aez_temp_regime[i_row, i_col] = 2  # Tropics, highland
                    
                elif aez_tclimate[i_row, i_col] == 3 and aez_tzone[i_row, i_col] == 1:
                    aez_temp_regime[i_row, i_col] = 3  # Subtropics, warm
                    
                elif aez_tclimate[i_row, i_col] == 3 and aez_tzone[i_row, i_col] == 2:
                    aez_temp_regime[i_row, i_col] = 4  # Subtropics,moderate cool
                    
                elif aez_tclimate[i_row, i_col] == 3 and aez_tzone[i_row, i_col] == 4:
                    aez_temp_regime[i_row, i_col] = 5  # Subtropics,cool
                    
                elif aez_tclimate[i_row, i_col] == 4 and aez_tzone[i_row, i_col] == 3:
                    aez_temp_regime[i_row, i_col] = 6  # Temperate, moderate
                    
                elif aez_tclimate[i_row, i_col] == 4 and aez_tzone[i_row, i_col] == 4:
                    aez_temp_regime[i_row, i_col] = 7  # Temperate, cool
                    
                elif np.logical_or.reduce((aez_tclimate[i_row, i_col] == 2, aez_tclimate[i_row, i_col] == 3, aez_tclimate[i_row, i_col] == 4, aez_tclimate[i_row, i_col] == 5))== True and aez_tzone[i_row, i_col] == 5 and np.logical_or(permafrost[i_row, i_col] == 1, permafrost[i_row, i_col] == 2) == False:
                    aez_temp_regime[i_row, i_col] = 8  # Boreal/Cold, no permafrost

                elif np.logical_or.reduce((aez_tclimate[i_row, i_col] == 2, aez_tclimate[i_row, i_col] == 3, aez_tclimate[i_row, i_col] == 4, aez_tclimate[i_row, i_col] == 5))== True and aez_tzone[i_row, i_col] == 5 and np.logical_or(permafrost[i_row, i_col] == 1, permafrost[i_row, i_col] == 2) == True:
                    aez_temp_regime[i_row, i_col] = 9  # Boreal/Cold, with permafrost
                    
                elif np.logical_or.reduce((aez_tclimate[i_row, i_col] == 2, aez_tclimate[i_row, i_col] == 3, aez_tclimate[i_row, i_col] == 4, aez_tclimate[i_row, i_col] == 5, aez_tclimate[i_row, i_col] == 6))== True and aez_tzone[i_row, i_col] == 6:
                    aez_temp_regime[i_row, i_col] = 10  # Arctic/Very Cold
                    
        # 4th Step: Moisture Regime classes
        # Moisture Regime Class Definition
        # 1 = M1 (desert/arid areas, 0 <= LGP* < 60)
        # 2 = M2 (semi-arid/dry areas, 60 <= LGP* < 180)
        # 3 = M3 (sub-humid/moist areas, 180 <= LGP* < 270)
        # 4 = M4 (humid/wet areas, LGP* >= 270)
        
        aez_moisture_regime = np.zeros((self.im_height, self.im_width), dtype = int)
        
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                # check if LGP t>5 is greater or less than 330 days. If greater, LGP will be used; otherwise, LGP_equv will be used.
                if lgpt_5[i_row, i_col] > 330:
                    
                    # Class 1 (M1)
                    if lgp[i_row, i_col] >= 0 and lgp[i_row, i_col] < 60:
                        aez_moisture_regime[i_row, i_col] = 1
                    
                    # Class 2 (M2)
                    elif lgp[i_row, i_col] >= 60 and lgp[i_row, i_col] < 180:
                        aez_moisture_regime[i_row, i_col] = 2
                        
                    #Class 3 (M3)   
                    elif lgp[i_row, i_col] >= 180 and lgp[i_row, i_col] < 270:
                        aez_moisture_regime[i_row, i_col] = 3
                        
                    # Class 4 (M4)
                    elif lgp[i_row, i_col] >= 270:
                        aez_moisture_regime[i_row, i_col] = 4
                
                elif lgpt_5[i_row, i_col] <= 330:
                    
                    # Class 1 (M1)
                    if lgp_equv[i_row, i_col] >= 0 and lgp_equv[i_row, i_col] < 60:
                        aez_moisture_regime[i_row, i_col] = 1
                    
                    # Class 2 (M2)
                    elif lgp_equv[i_row, i_col] >= 60 and lgp_equv[i_row, i_col] < 180:
                        aez_moisture_regime[i_row, i_col] = 2
                        
                    #Class 3 (M3)   
                    elif lgp_equv[i_row, i_col] >= 180 and lgp_equv[i_row, i_col] < 270:
                        aez_moisture_regime[i_row, i_col] = 3
                        
                    # Class 4 (M4)
                    elif lgp_equv[i_row, i_col] >= 270:
                        aez_moisture_regime[i_row, i_col] = 4
                
        # Now, we will classify the agro-ecological zonation
        # By GAEZ v4 Documentation, there are prioritized sequential assignment of AEZ classes in order to ensure the consistency of classification 
        aez = np.zeros((self.im_height, self.im_width), dtype = int)
        
        
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                
                # if it's urban built-up lulc
                if soil_terrain_lulc[i_row, i_col]== 8:
                    aez[i_row, i_col] = 56
                
                # if it's water
                elif soil_terrain_lulc[i_row, i_col]== 7:
                    aez[i_row, i_col] = 57
                
                # if it's dominantly very steep terrain
                elif soil_terrain_lulc[i_row, i_col]== 1:
                    aez[i_row, i_col] = 49
                
                # if it's irrigated soils
                elif soil_terrain_lulc[i_row, i_col]== 6:
                    aez[i_row, i_col] = 51
                
                # if it's hydromorphic soils
                elif soil_terrain_lulc[i_row, i_col]== 2:
                    aez[i_row, i_col] = 52
                    
                elif aez_moisture_regime[i_row, i_col]==1:
                    aez[i_row, i_col] = 53
                    
                elif aez_temp_regime[i_row, i_col]==9 and aez_moisture_regime[i_row, i_col] in [1,2,3,4] == True:
                    aez[i_row, i_col] = 54
                    
                elif soil_terrain_lulc[i_row, i_col]==5:
                    aez[i_row, i_col] = 50
                
                elif aez_temp_regime[i_row, i_col]==10 and aez_moisture_regime[i_row, i_col] in [1,2,3,4] == True:
                    aez[i_row, i_col] = 55
                    
                #######
                elif aez_temp_regime[i_row, i_col]==1 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 1
                
                elif aez_temp_regime[i_row, i_col]==1 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 2
                
                elif aez_temp_regime[i_row, i_col]==1 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 3
                
                elif aez_temp_regime[i_row, i_col]==1 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 4
                
                elif aez_temp_regime[i_row, i_col]==1 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 5
                
                elif aez_temp_regime[i_row, i_col]==1 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 6
                ####
                elif aez_temp_regime[i_row, i_col]==2 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 7
                
                elif aez_temp_regime[i_row, i_col]==2 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 8
                
                elif aez_temp_regime[i_row, i_col]==2 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 9
                
                elif aez_temp_regime[i_row, i_col]==2 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 10
                
                elif aez_temp_regime[i_row, i_col]==2 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 11
                
                elif aez_temp_regime[i_row, i_col]==2 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 12
                ###
                elif aez_temp_regime[i_row, i_col]==3 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 13
                
                elif aez_temp_regime[i_row, i_col]==3 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 14
                
                elif aez_temp_regime[i_row, i_col]==3 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 15
                
                elif aez_temp_regime[i_row, i_col]==3 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 16
                
                elif aez_temp_regime[i_row, i_col]==3 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 17
                
                elif aez_temp_regime[i_row, i_col]==3 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 18
                #####
                elif aez_temp_regime[i_row, i_col]==4 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 19
                
                elif aez_temp_regime[i_row, i_col]==4 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 20
                
                elif aez_temp_regime[i_row, i_col]==4 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 21
                
                elif aez_temp_regime[i_row, i_col]==4 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 22
                
                elif aez_temp_regime[i_row, i_col]==4 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 23
                
                elif aez_temp_regime[i_row, i_col]==4 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 24
                #####
                elif aez_temp_regime[i_row, i_col]==5 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 25
                
                elif aez_temp_regime[i_row, i_col]==5 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 26
                
                elif aez_temp_regime[i_row, i_col]==5 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 27
                
                elif aez_temp_regime[i_row, i_col]==5 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 28
                
                elif aez_temp_regime[i_row, i_col]==5 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 29
                
                elif aez_temp_regime[i_row, i_col]==5 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 30
                ######
                
                elif aez_temp_regime[i_row, i_col]==6 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 31
                
                elif aez_temp_regime[i_row, i_col]==6 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 32
                
                elif aez_temp_regime[i_row, i_col]==6 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 33
                
                elif aez_temp_regime[i_row, i_col]==6 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 34
                
                elif aez_temp_regime[i_row, i_col]==6 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 35
                
                elif aez_temp_regime[i_row, i_col]==6 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 36
                
                ###
                elif aez_temp_regime[i_row, i_col]==7 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 37
                
                elif aez_temp_regime[i_row, i_col]==7 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 38
                
                elif aez_temp_regime[i_row, i_col]==7 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 39
                
                elif aez_temp_regime[i_row, i_col]==7 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 40
                
                elif aez_temp_regime[i_row, i_col]==7 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 41
                
                elif aez_temp_regime[i_row, i_col]==7 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 42
                #####
                
                elif aez_temp_regime[i_row, i_col]==8 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 43
                
                elif aez_temp_regime[i_row, i_col]==8 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 44
                
                elif aez_temp_regime[i_row, i_col]==8 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 45
                
                elif aez_temp_regime[i_row, i_col]==8 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 46
                
                elif aez_temp_regime[i_row, i_col]==8 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 47
                
                elif aez_temp_regime[i_row, i_col]==8 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 48
                

        if self.set_mask:
            return np.ma.masked_where(self.im_mask==0, aez)
        else:        
            return aez
    
    
#----------------- End of file -------------------------#