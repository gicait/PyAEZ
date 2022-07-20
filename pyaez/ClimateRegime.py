"""
PyAEZ
Written by N. Lakmal Deshapriya and Thaileng Thol
"""

import numpy as np
np.seterr(divide='ignore', invalid='ignore') # ignore "divide by zero" or "divide by NaN" warning
from . import UtilitiesCalc
from . import ETOCalc

class ClimateRegime(object):

    # set mask of study area, this is optional
    def setStudyAreaMask(self, admin_mask, no_data_value):
        self.im_mask = admin_mask
        self.nodata_val = no_data_value

        self.set_mask = True

    def setLocationTerrainData(self, lat_min, lat_max, elevation):
        self.elevation = elevation
        self.im_height = elevation.shape[0]
        self.im_width = elevation.shape[1]
        self.latitude = UtilitiesCalc.UtilitiesCalc().generateLatitudeMap(lat_min, lat_max, self.im_height, self.im_width)

    def setMonthlyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity):

        self.meanT_daily = np.zeros((self.im_height, self.im_width, 365))
        self.totalPrec_daily = np.zeros((self.im_height, self.im_width, 365))
        self.pet_daily = np.zeros((self.im_height, self.im_width, 365))

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

                minT_daily = obj_utilities.interpMonthlyToDaily(min_temp[i_row, i_col,:], 1, 365)
                maxT_daily = obj_utilities.interpMonthlyToDaily(max_temp[i_row, i_col,:], 1, 365)
                radiation_daily = obj_utilities.interpMonthlyToDaily(short_rad[i_row, i_col,:], 1, 365, no_minus_values=True)
                wind_daily = obj_utilities.interpMonthlyToDaily(wind_speed[i_row, i_col,:], 1, 365, no_minus_values=True)
                rel_humidity_daily = obj_utilities.interpMonthlyToDaily(rel_humidity[i_row, i_col,:], 1, 365, no_minus_values=True)

                # calculation of ET
                obj_eto = ETOCalc.ETOCalc(1, 365, self.latitude[i_row, i_col], self.elevation[i_row, i_col])
                shortrad_daily_MJm2day = (radiation_daily*3600*24)/1000000 # convert w/m2 to MJ/m2/day
                obj_eto.setClimateData(minT_daily, maxT_daily, wind_daily, shortrad_daily_MJm2day, rel_humidity_daily)
                self.pet_daily[i_row, i_col, :] = obj_eto.calculateETO()

        # sea level temperature
        self.meanT_daily_sealevel = self.meanT_daily + np.tile(np.reshape(self.elevation/100*0.55, (self.im_height,self.im_width,1)), (1,1,365))

        # P over PET ratio
        self.P_by_PET_daily = self.totalPrec_daily / self.pet_daily

    def setDailyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity):

        self.meanT_daily = np.zeros((self.im_height, self.im_width, 365))
        self.totalPrec_daily = np.zeros((self.im_height, self.im_width, 365))
        self.pet_daily = np.zeros((self.im_height, self.im_width, 365))

        # calculation of ET
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):

                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue

                self.meanT_daily[i_row, i_col, :] = (min_temp[i_row, i_col, :]+max_temp[i_row, i_col, :])/2
                self.totalPrec_daily[i_row, i_col, :] = precipitation[i_row, i_col, :]

                obj_eto = ETOCalc.ETOCalc(1, 365, self.latitude[i_row, i_col], self.elevation[i_row, i_col])
                shortrad_daily_MJm2day = (short_rad[i_row, i_col, :]*3600*24)/1000000 # convert w/m2 to MJ/m2/day
                obj_eto.setClimateData(min_temp[i_row, i_col, :], max_temp[i_row, i_col, :], wind_speed[i_row, i_col, :], shortrad_daily_MJm2day, rel_humidity[i_row, i_col, :])
                self.pet_daily[i_row, i_col, :] = obj_eto.calculateETO()

        # sea level temperature
        self.meanT_daily_sealevel = self.meanT_daily + np.tile(np.reshape(self.elevation/100*0.55, (self.im_height,self.im_width,1)), (1,1,365))

        # P over PET ratio
        self.P_by_PET_daily = self.totalPrec_daily / self.pet_daily

    def getThermalClimate(self):

        thermal_climate = np.zeros((self.im_height, self.im_width))

        for i_r in range(self.im_height):
            for i_c in range(self.im_width):

                if self.set_mask:
                    if self.im_mask[i_r, i_c] == self.nodata_val:
                        continue

                # converting daily to monthly
                obj_utilities = UtilitiesCalc.UtilitiesCalc()
                meanT_monthly_sealevel_v = obj_utilities.averageDailyToMonthly(self.meanT_daily_sealevel[i_r,i_c,:])
                meanT_monthly_v = obj_utilities.averageDailyToMonthly(self.meanT_daily[i_r,i_c,:])
                P_by_PET_monthly_v = obj_utilities.averageDailyToMonthly(self.P_by_PET_daily[i_r,i_c,:])

                if np.min(meanT_monthly_sealevel_v) >= 18:
                    # Tropics
                    #if np.mean(meanT_monthly_v) > 20:
                    if np.min(meanT_monthly_v) > 20:
                        # Tropical lowland
                        thermal_climate[i_r,i_c] = 1
                    else:
                        # Tropical highland
                        thermal_climate[i_r,i_c] = 2

                elif np.min(meanT_monthly_sealevel_v) >= 5 and np.sum(meanT_monthly_sealevel_v>=10) >= 8:
                    # SubTropic
                    if np.sum(self.totalPrec_daily[i_r,i_c,:]) < 250:
                        # 'Subtropics Low Rainfall
                        thermal_climate[i_r,i_c] = 3
                    elif self.latitude[i_r,i_c]>=0 and np.mean(P_by_PET_monthly_v[3:9]) >= np.mean([P_by_PET_monthly_v[9:12],P_by_PET_monthly_v[0:3]]):
                        # Subtropics Summer Rainfall
                        thermal_climate[i_r,i_c] = 4
                    elif self.latitude[i_r,i_c]<0 and np.mean(P_by_PET_monthly_v[3:9]) <= np.mean([P_by_PET_monthly_v[9:12],P_by_PET_monthly_v[0:3]]):
                        # Subtropics Summer Rainfall
                        thermal_climate[i_r,i_c] = 4
                    elif self.latitude[i_r,i_c]>=0 and np.mean(P_by_PET_monthly_v[3:9]) <= np.mean([P_by_PET_monthly_v[9:12],P_by_PET_monthly_v[0:3]]):
                        # Subtropics Winter Rainfall
                        thermal_climate[i_r,i_c] = 5
                    elif self.latitude[i_r,i_c]<0 and np.mean(P_by_PET_monthly_v[3:9]) >= np.mean([P_by_PET_monthly_v[9:12],P_by_PET_monthly_v[0:3]]):
                        # Subtropics Winter Rainfall
                        thermal_climate[i_r,i_c] = 5

                elif np.sum(meanT_monthly_sealevel_v>=10) >= 4:
                    # Temperate
                    if np.max(meanT_monthly_v)-np.min(meanT_monthly_v) < 20:
                        # Oceanic Temperate
                        thermal_climate[i_r,i_c] = 6
                    elif np.max(meanT_monthly_v)-np.min(meanT_monthly_v) < 35:
                        # Sub-Continental Temperate
                        thermal_climate[i_r,i_c] = 7
                    else:
                        # Continental Temperate
                        thermal_climate[i_r,i_c] = 8

                elif np.sum(meanT_monthly_sealevel_v>=10) >= 1:
                    # Boreal
                    if np.max(meanT_monthly_v)-np.min(meanT_monthly_v) < 20:
                        # Oceanic Boreal
                        thermal_climate[i_r,i_c] = 9
                    elif np.max(meanT_monthly_v)-np.min(meanT_monthly_v) < 35:
                        # Sub-Continental Boreal
                        thermal_climate[i_r,i_c] = 10
                    else:
                        # Continental Boreal
                        thermal_climate[i_r,i_c] = 11
                else:
                    # Arctic
                    thermal_climate[i_r,i_c] = 12

        return thermal_climate

    def getThermalZone(self):

        thermal_zone = np.zeros((self.im_height, self.im_width))

        for i_r in range(self.im_height):
            for i_c in range(self.im_width):

                if self.set_mask:
                    if self.im_mask[i_r, i_c] == self.nodata_val:
                        continue

                # converting daily to monthly
                obj_utilities = UtilitiesCalc.UtilitiesCalc()
                meanT_monthly_sealevel_v = obj_utilities.averageDailyToMonthly(self.meanT_daily_sealevel[i_r,i_c,:])
                meanT_monthly_v = obj_utilities.averageDailyToMonthly(self.meanT_daily[i_r,i_c,:])

                if np.min(meanT_monthly_sealevel_v) >= 18:
                    # Tropics
                    if np.mean(meanT_monthly_v) > 20:
                        # Warm
                        thermal_zone[i_r,i_c] = 1
                    elif np.sum(meanT_monthly_v<18) >= 1 and np.min(meanT_monthly_v) > 5 and np.sum(meanT_monthly_v>10) >= 8:
                        # Moderately Cool
                        thermal_zone[i_r,i_c] = 2
                    elif np.sum(meanT_monthly_v<5) >= 1 and np.sum(meanT_monthly_v>10) >= 4:
                        # Cool
                        thermal_zone[i_r,i_c] = 3
                    elif np.sum(meanT_monthly_v<5) >= 1 and np.sum(meanT_monthly_v>10) >= 1:
                        # Cold
                        thermal_zone[i_r,i_c] = 4
                    elif np.sum(meanT_monthly_v<10) == 12:
                        # Very Cold
                        thermal_zone[i_r,i_c] = 5

                elif np.min(meanT_monthly_sealevel_v) >= 5 and np.sum(meanT_monthly_sealevel_v>=10) >= 8:
                    # SubTropic
                    if np.mean(meanT_monthly_v) > 20:
                        # Warm
                        thermal_zone[i_r,i_c] = 6
                    elif np.sum(meanT_monthly_v<18) >= 1 and np.min(meanT_monthly_v) > 5 and np.sum(meanT_monthly_v>10) >= 8:
                        # Moderately Cool
                        thermal_zone[i_r,i_c] = 7
                    elif np.sum(meanT_monthly_v<5) >= 1 and np.sum(meanT_monthly_v>10) >= 4:
                        # Cool
                        thermal_zone[i_r,i_c] = 8
                    elif np.sum(meanT_monthly_v<5) >= 1 and np.sum(meanT_monthly_v>10) >= 1:
                        # Cold
                        thermal_zone[i_r,i_c] = 9
                    elif np.sum(meanT_monthly_v<10) == 12:
                        # Very Cold
                        thermal_zone[i_r,i_c] = 10

                elif np.sum(meanT_monthly_sealevel_v>=10) >= 4:
                    # Temperate
                    if np.mean(meanT_monthly_v) > 20:
                        # Warm
                        thermal_zone[i_r,i_c] = 11
                    elif np.sum(meanT_monthly_v<18) >= 1 and np.min(meanT_monthly_v) > 5 and np.sum(meanT_monthly_v>10) >= 8:
                        # Moderately Cool
                        thermal_zone[i_r,i_c] = 12
                    elif np.sum(meanT_monthly_v<5) >= 1 and np.sum(meanT_monthly_v>10) >= 4:
                        # Cool
                        thermal_zone[i_r,i_c] = 13
                    elif np.sum(meanT_monthly_v<5) >= 1 and np.sum(meanT_monthly_v>10) >= 1:
                        # Cold
                        thermal_zone[i_r,i_c] = 14
                    elif np.sum(meanT_monthly_v<10) == 12:
                        # Very Cold
                        thermal_zone[i_r,i_c] = 15

                elif np.sum(meanT_monthly_sealevel_v>=10) >= 1:
                    # Boreal
                    if np.mean(meanT_monthly_v) > 20:
                        # Warm
                        thermal_zone[i_r,i_c] = 16
                    elif np.sum(meanT_monthly_v<18) >= 1 and np.min(meanT_monthly_v) > 5 and np.sum(meanT_monthly_v>10) >= 8:
                        # Moderately Cool
                        thermal_zone[i_r,i_c] = 17
                    elif np.sum(meanT_monthly_v<5) >= 1 and np.sum(meanT_monthly_v>10) >= 4:
                        # Cool
                        thermal_zone[i_r,i_c] = 18
                    elif np.sum(meanT_monthly_v<5) >= 1 and np.sum(meanT_monthly_v>10) >= 1:
                        # Cold
                        thermal_zone[i_r,i_c] = 19
                    elif np.sum(meanT_monthly_v<10) == 12:
                        # Very Cold
                        thermal_zone[i_r,i_c] = 20
                else:
                    # Arctic
                    thermal_zone[i_r,i_c] = 21

        return thermal_zone

    def getThermalLGP0(self):
        return np.sum(self.meanT_daily>0, axis=2)

    def getThermalLGP5(self):
        return np.sum(self.meanT_daily>5, axis=2)

    def getThermalLGP10(self):
        return np.sum(self.meanT_daily>10, axis=2)

    def getTemperatureSum0(self):
        tempT = self.meanT_daily
        tempT[tempT<=0] = 0
        return np.sum(tempT, axis=2)

    def getTemperatureSum5(self):
        tempT = self.meanT_daily
        tempT[tempT<=5] = 0
        return np.sum(tempT, axis=2)

    def getTemperatureSum10(self):
        tempT = self.meanT_daily
        tempT[tempT<=10] = 0
        return np.sum(tempT, axis=2)

    def getTemperatureProfile(self):

        meanT_daily_add1day = np.concatenate((self.meanT_daily, self.meanT_daily[:,:,0:1]), axis=-1)
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

        return [A9,A8,A7,A6,A5,A4,A3,A2,A1,B1,B2,B3,B4,B5,B6,B7,B8,B9]


    def getLGP(self, Sa = 100, pc = 0.5, kc = 1, D = 1):

        # Sa: available soil moisture holding capacity (mm/m) , usually assume as 100
        # kc: crop water requirements for entire growth cycle
        # D: rooting depth (m)
        # pc: soil water depletion fraction below which ETa < ETo (from literature)

        petc = np.zeros(self.pet_daily.shape)
        peta = np.zeros(self.pet_daily.shape)
        petc = self.pet_daily * 1

        for i_r in range(self.im_height):
            for i_c in range(self.im_width):

                if self.set_mask:
                    if self.im_mask[i_r, i_c] == self.nodata_val:
                        continue

                if len(np.array(Sa).shape) == 2:
                    Sa_temp = Sa[i_r, i_c]
                else:
                    Sa_temp = Sa

                W = 0

                for ii in range(self.pet_daily.shape[-1]):
                    if self.totalPrec_daily[i_r,i_c,ii] >= petc[i_r,i_c,ii]:
                        peta[i_r,i_c,ii] = petc[i_r,i_c,ii]
                    elif self.totalPrec_daily[i_r,i_c,ii] + W >= Sa_temp*D*(1-pc):
                        peta[i_r,i_c,ii] = petc[i_r,i_c,ii]
                    else:
                        kk = (W+self.totalPrec_daily[i_r,i_c,ii]) / (Sa_temp*D*(1-pc))
                        peta[i_r,i_c,ii] = kk * petc[i_r,i_c,ii]

                    W = np.min([W+self.totalPrec_daily[i_r,i_c,ii]-peta[i_r,i_c,ii], Sa_temp*D])
                    if W<0: W=0

        return np.sum( (peta/self.pet_daily)>0.5, axis=2 )

    def getLGP_G_5_and_10(self):

        kc = 1 # crop water requirements for entire growth cycle
        Sa = 100 # available soil moisture holding capacity (mm/m) , usually assume as 100
        D = 1 # rooting depth (m)
        pc = 0.5 # soil water depletion fraction below which ETa < ETo (from literature)

        petc = np.zeros(self.pet_daily.shape)
        peta = np.zeros(self.pet_daily.shape)
        petc = self.pet_daily * 1

        for i_r in range(self.im_height):
            for i_c in range(self.im_width):

                if self.set_mask:
                    if self.im_mask[i_r, i_c] == self.nodata_val:
                        continue

                W = 0

                for ii in range(self.pet_daily.shape[-1]):
                    if self.totalPrec_daily[i_r,i_c,ii] >= petc[i_r,i_c,ii]:
                        peta[i_r,i_c,ii] = petc[i_r,i_c,ii]
                    elif self.totalPrec_daily[i_r,i_c,ii] + W >= Sa*D*(1-pc):
                        peta[i_r,i_c,ii] = petc[i_r,i_c,ii]
                    else:
                        kk = (W+self.totalPrec_daily[i_r,i_c,ii]) / (Sa*D*(1-pc))
                        peta[i_r,i_c,ii] = kk * petc[i_r,i_c,ii]

                    W = np.min([W+self.totalPrec_daily[i_r,i_c,ii]-peta[i_r,i_c,ii], Sa*D])
                    if W<0: W=0

        tempT = self.meanT_daily
        tempT[np.logical_and(tempT<5, (peta/self.pet_daily)<=0.5)] = 0
        ts_g_t5 = np.sum(tempT, axis=2)

        tempT = self.meanT_daily
        tempT[np.logical_and(tempT<10, (peta/self.pet_daily)<=0.5)] = 0
        ts_g_t10 = np.sum(tempT, axis=2)

        return [ts_g_t5, ts_g_t10]

    def getLGPClassified(self, lgp):

        lgp_class = np.zeros(lgp.shape)

        lgp_class[lgp>=365] = 7 # Per-humid
        lgp_class[np.logical_and(lgp>=270, lgp<365)] = 6 # Humid
        lgp_class[np.logical_and(lgp>=180, lgp<270)] = 5 # Sub-humid
        lgp_class[np.logical_and(lgp>=120, lgp<180)] = 4 # Moist semi-arid
        lgp_class[np.logical_and(lgp>=60, lgp<120)] = 3 # Dry semi-arid
        lgp_class[np.logical_and(lgp>0, lgp<60)] = 2 # Arid
        lgp_class[lgp<=0] = 1 # Hyper-arid

        return lgp_class

    def getLGPEquivalent(self):

        moisture_index = np.sum(self.totalPrec_daily, axis=2)/np.sum(self.pet_daily, axis=2)

        lgp_equv = 14.0 + 293.66*moisture_index - 61.25*moisture_index*moisture_index
        lgp_equv[ moisture_index > 2.4 ] = 366

        return lgp_equv


    def getMultiCroppingZones(self, t_climate, lgp, lgp_t5, lgp_t10, ts_t0, ts_t10):

        ts_g_t5_and_10 = self.getLGP_G_5_and_10()
        ts_g_t5 = ts_g_t5_and_10[0]
        ts_g_t10 = ts_g_t5_and_10[1]

        multi_cropping = np.zeros(lgp.shape)

        zone_B = np.all([lgp>=45, lgp_t5>=120, lgp_t10>=90, ts_t0>=1600, ts_t10>=1200], axis=0)
        zone_C1 = np.all([lgp>=220, lgp_t5>=220, lgp_t10>=120, ts_t0>=5500, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        #zone_C2 = np.all([lgp>=200, lgp_t5>=210, lgp_t10>=120, ts_t0>=6400, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        zone_C3 = np.all([lgp>=180, lgp_t5>=200, lgp_t10>=120, ts_t0>=7200, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        zone_D1 = np.all([lgp>=270, lgp_t5>=270, lgp_t10>=165, ts_t0>=5500, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        #zone_D2 = np.all([lgp>=240, lgp_t5>=240, lgp_t10>=165, ts_t0>=6400, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_D3 = np.all([lgp>=210, lgp_t5>=240, lgp_t10>=165, ts_t0>=7200, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_F = np.all([lgp>=300, lgp_t5>=300, lgp_t10>=240, ts_t0>=7200, ts_t10>=7000, ts_g_t5>=5100, ts_g_t10>=4800], axis=0)
        zone_H = np.all([lgp>=360, lgp_t5>=360, lgp_t10>=330, ts_t0>=7200, ts_t10>=7000], axis=0)

        multi_cropping[np.all([zone_B,t_climate<=2], axis=0)] = 2
        multi_cropping[np.all([zone_C1,t_climate==1], axis=0)] = 3
        multi_cropping[np.all([zone_C3,t_climate==2], axis=0)] = 3
        multi_cropping[np.all([zone_D1,t_climate==1], axis=0)] = 4
        multi_cropping[np.all([zone_D3,t_climate==2], axis=0)] = 4
        multi_cropping[np.all([zone_F,t_climate<=2], axis=0)] = 6
        multi_cropping[np.all([zone_H,t_climate<=2], axis=0)] = 8

        zone_B = np.all([lgp>=45, lgp_t5>=120, lgp_t10>=90, ts_t0>=1600, ts_t10>=1200], axis=0)
        zone_C = np.all([lgp>=180, lgp_t5>=200, lgp_t10>=120, ts_t0>=3600, ts_t10>=3000, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        zone_D = np.all([lgp>=210, lgp_t5>=240, lgp_t10>=165, ts_t0>=4500, ts_t10>=3600, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_E = np.all([lgp>=240, lgp_t5>=270, lgp_t10>=180, ts_t0>=4800, ts_t10>=4500, ts_g_t5>=4300, ts_g_t10>=4000], axis=0)
        zone_F = np.all([lgp>=300, lgp_t5>=300, lgp_t10>=240, ts_t0>=5400, ts_t10>=5100, ts_g_t5>=5100, ts_g_t10>=4800], axis=0)
        zone_G = np.all([lgp>=330, lgp_t5>=330, lgp_t10>=270, ts_t0>=5700, ts_t10>=5500], axis=0)
        zone_H = np.all([lgp>=360, lgp_t5>=360, lgp_t10>=330, ts_t0>=7200, ts_t10>=7000], axis=0)

        multi_cropping[np.all([zone_B,t_climate>=3,t_climate<=8], axis=0)] = 2
        multi_cropping[np.all([zone_C,t_climate>=3,t_climate<=8], axis=0)] = 3
        multi_cropping[np.all([zone_D,t_climate>=3,t_climate<=8], axis=0)] = 4
        multi_cropping[np.all([zone_E,t_climate>=3,t_climate<=8], axis=0)] = 5
        multi_cropping[np.all([zone_F,t_climate>=3,t_climate<=8], axis=0)] = 6
        multi_cropping[np.all([zone_G,t_climate>=3,t_climate<=8], axis=0)] = 7
        multi_cropping[np.all([zone_H,t_climate>=3,t_climate<=8], axis=0)] = 8

        return multi_cropping
