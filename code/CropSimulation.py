"""
PyAEZ
Written by N. Lakmal Deshapriya
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
# import gdal
from osgeo import gdal
import pandas as pd
import io

import UtilitiesCalc
import BioMassCalc
import ETOCalc
import CropWatCalc
import ThermalScreening_mod_Sh
import ClimateRegime 

class CropSimulation(object):

    def __init__(self):

        self.set_mask = False

        self.set_tclimate_screening = False
        self.set_lgpt_screening = False
        self.set_Tsum_screening = False
        self.set_Tprofile_screening = False

    def setMonthlyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity):
        # either one of monthly or daily climate data should be set
        self.minT_monthly = min_temp
        self.maxT_monthly = max_temp
        self.totalPrec_monthly = precipitation
        self.shortRad_monthly = short_rad
        self.wind2m_monthly = wind_speed
        self.rel_humidity_monthly = rel_humidity

        self.set_monthly = True

    def setDailyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity):
        # either one of monthly or daily climate data should be set
        self.minT_daily = min_temp
        self.maxT_daily = max_temp
        self.totalPrec_daily = precipitation
        self.shortRad_daily = short_rad
        self.wind2m_daily = wind_speed
        self.rel_humidity_daily = rel_humidity

        self.set_monthly = False

    def setLocationTerrainData(self, lat_min, lat_max, elevation):
        self.elevation = elevation
        self.im_height = elevation.shape[0]
        self.im_width = elevation.shape[1]
        self.latitude_map = UtilitiesCalc.UtilitiesCalc().generateLatitudeMap(lat_min, lat_max, self.im_height, self.im_width)

    def setCropParameters(self, LAI, HI, legume, adaptability, cycle_len, D1, D2, min_temp):
        self.LAi = LAI # leaf area index
        self.HI = HI # harvest index
        self.legume = legume # binary value
        self.adaptability = adaptability  #one of [1,2,3,4] classes
        self.cycle_len = cycle_len  #length of growing period

        self.D1 = D1  # rooting depth 1 (m)
        self.D2 = D2  # rooting depth 2 (m)

        self.min_temp = min_temp

    def setCropCycleParameters(self, stage_per, kc, kc_all, yloss_f, yloss_f_all):
        self.d_per = stage_per # Percentage for D1, D2, D3, D4 stages
        self.kc = kc # 3 crop water requirements for initial, reproductive, the end of the maturation stages
        self.kc_all = kc_all # crop water requirements for entire growth cycle
        self.yloss_f = yloss_f  # yield loss for D1, D2, D3, D4
        self.yloss_f_all = yloss_f_all  # yield loss for entire growth cycle

    def setAccTsum(self, LnS, LsO, LO,HnS, HsO,HO):
        self.LnS=LnS
        self.LsO=LsO
        self.LO=LO
        self.HnS=HnS
        self.HsO= HsO
        self.HO= HO

        self.set_Tsum_screening = True

    def getRainfedCycEff(self, LGP, cycle_len):
        self.cyc_eff_rainfed = np.minimum(LGP, cycle_len)

    def getIrrigatedCycEff(self, min_temp, LGPT5, LGPT10, cycle_len):
        print(min_temp)
        if min_temp == 5:
            self.cyc_eff_irrigated = np.minimum(LGPT5, cycle_len)
        else:
            self.cyc_eff_irrigated = np.minimum(LGPT10, cycle_len)

    def setPerennialCropParametersFromCSV(self, file_path, crop_name):
        self.crop_name = crop_name
        df = pd.read_csv(file_path)
        crop_df_index = df.index[df['Crop_name'] == crop_name].tolist()[0]
        crop_df = df.loc[df['Crop_name'] == crop_name]
        print("index:", crop_df_index)
        print(crop_df['D2'][crop_df_index])
        self.setCropParameters(LAI=crop_df['LAI'][crop_df_index], HI=crop_df['HI'][crop_df_index], legume=crop_df['legume'][crop_df_index], adaptability=int(crop_df['adaptability'][crop_df_index]), cycle_len=int(crop_df['cycle_len'][crop_df_index]), D1=crop_df['D1'][crop_df_index], D2=crop_df['D2'][crop_df_index], min_temp=crop_df['min_temp'][crop_df_index])

        LnS, LsO, LO, HO, HsO, HnS = ([] for i in range(6))        

        LnS.append(crop_df['LnS_0'][crop_df_index]) 
        LsO.append(crop_df['LsO_0'][crop_df_index])
        LO.append(crop_df['LO_0'][crop_df_index])
        HnS.append(crop_df['HnS_0'][crop_df_index]) 
        HsO.append(crop_df['HsO_0'][crop_df_index])
        HO.append(crop_df['HO_0'][crop_df_index])

        LnS.append(crop_df['LnS_5'][crop_df_index]) 
        LsO.append(crop_df['LsO_5'][crop_df_index])
        LO.append(crop_df['LO_5'][crop_df_index])
        HnS.append(crop_df['HnS_5'][crop_df_index]) 
        HsO.append(crop_df['HsO_5'][crop_df_index])
        HO.append(crop_df['HO_5'][crop_df_index])

        LnS.append(crop_df['LnS_10'][crop_df_index]) 
        LsO.append(crop_df['LsO_10'][crop_df_index])
        LO.append(crop_df['LO_10'][crop_df_index])
        HnS.append(crop_df['HnS_10'][crop_df_index]) 
        HsO.append(crop_df['HsO_10'][crop_df_index])
        HO.append(crop_df['HO_10'][crop_df_index])

        self.setAccTsum(LnS, LsO, LO, HO, HsO, HnS)

        self.setCropCycleParameters(stage_per=[crop_df['stage_per_1'][crop_df_index], crop_df['stage_per_2'][crop_df_index], crop_df['stage_per_3'][crop_df_index], crop_df['stage_per_4'][crop_df_index]], kc=[crop_df['kc_1'][crop_df_index], crop_df['kc_2'][crop_df_index], crop_df['kc_3'][crop_df_index]], kc_all=crop_df['kc_all'][crop_df_index], yloss_f=[crop_df['yloss_f1'][crop_df_index], crop_df['yloss_f2'][crop_df_index], crop_df['yloss_f3'][crop_df_index], crop_df['yloss_f4'][crop_df_index]], yloss_f_all=crop_df['yloss_f_all'][crop_df_index])
       

        self.is_perennial = crop_df['annual/perennial flag'][crop_df_index]
        # print(self.is_perennial)
        if self.is_perennial:

            climate = ClimateRegime.ClimateRegime()
            climate.elevation = self.elevation
            climate.im_height = self.im_height
            climate.im_width = self.im_width
            climate.latitude = self.latitude_map

            # climate.setLocationTerrainData(self.lat_min, self.lat_max, self.elevation)
            climate.im_mask = self.im_mask
            climate.nodata_val = self.nodata_val
            climate.set_mask = True
            
            # climate.setStudyAreaMask(self.im_mask, self.nodata_val)
            if self.set_monthly:
                climate.setMonthlyClimateData( self.minT_monthly, self.maxT_monthly, self.totalPrec_monthly, self.shortRad_monthly, self.wind2m_monthly, self.rel_humidity_monthly)
            else:
                climate.setDailyClimateData( self.minT_daily,  self.maxT_daily, self.totalPrec_daily, self.shortRad_daily, self.wind2m_daily, self.rel_humidity_daily)  

            self.LGP= climate.getLGP()
            self.LGPT5= climate.getThermalLGP5() 
            self.LGPT10= climate.getThermalLGP10()

            self.getIrrigatedCycEff(self.min_temp, self.LGPT5, self.LGPT10, self.cycle_len)
            self.getRainfedCycEff(self.LGP, self.cycle_len)

            perennial_file_path = './sample_data/input/Adjustment_factors_for_perennial.csv'
            p_df = pd.read_csv(perennial_file_path)
            perennial_df_index = p_df.index[p_df['Crop_name'] == crop_name].to_list()[0]
            perennial_df = p_df.loc[p_df['Crop_name'] == crop_name]
            self.adjustForPerennialCrop(aLAI=perennial_df['aLAI'][perennial_df_index], bLAI=perennial_df['bLAI'][perennial_df_index], aHI=perennial_df['aHI'][perennial_df_index], bHI=perennial_df['bHI'][perennial_df_index])

    def setSoilWaterParameters(self, Sa, pc):
        self.Sa = Sa  # available soil moisture holding capacity (mm/m) , assumption
        self.pc = pc  # soil water depletion fraction below which ETa < ETo (from literature)

    '''All of bellow settings are optional'''

    # set mask of study area, this is optional
    def setStudyAreaMask(self, admin_mask, no_data_value):
        self.im_mask = admin_mask
        self.nodata_val = no_data_value

        self.set_mask = True

    def adjustForPerennialCrop(self,  aLAI, bLAI, aHI, bHI):
        self.LAi_rainfed = self.LAi * ((self.cyc_eff_rainfed-aLAI)/bLAI) # leaf area index adjustment for perennial crops
        self.HI_rainfed = self.HI * ((self.cyc_eff_rainfed-aHI)/bHI) # harvest index adjustment for perennial crops
        self.LAi_irrigated = self.LAi * ((self.cyc_eff_irrigated-aLAI)/bLAI) # leaf area index adjustment for perennial crops
        self.HI_irrigated = self.HI * ((self.cyc_eff_irrigated-aHI)/bHI) # harvest index adjustment for perennial crops

    def setThermalClimateScreening(self, t_climate, no_t_climate):
        self.t_climate = t_climate
        self.no_t_climate = no_t_climate # list of unsuitable thermal climate

        self.set_tclimate_screening = True

    # set suitability screening, this is also optional
    def setLGPTScreening(self, no_lgpt, optm_lgpt):
        self.no_lgpt = no_lgpt
        self.optm_lgpt = optm_lgpt

        self.set_lgpt_screening = True

    # def setTSumScreening(self, no_Tsum, optm_Tsum):
    #     self.no_Tsum = no_Tsum
    #     self.optm_Tsum = optm_Tsum

    #     self.set_Tsum_screening = True

    def setTProfileScreening(self, no_Tprofile, optm_Tprofile):
        self.no_Tprofile = no_Tprofile
        self.optm_Tprofile = optm_Tprofile

        self.set_Tprofile_screening = True
    
    def ReadThermalScreeningRukesFromCSV(self, path):
        a = 1
        # must be called after cropname is ddeclared
        # with open(path, 'r') as cs

        #self.thermalscreeningrules =  [   
        #     [constraint1, opr1, op1, subop1, notsu1], 
        #     ..., 
        #     [constraint6, opr6, op6, subop6, notsu6]]
        # ]


    def simulateCropCycle(self, start_doy=1, end_doy=365, step_doy=1, leap_year=False):

        # just a counter to keep track of progress
        count_pixel_completed = 0

        # this stores final result
        self.final_yield_rainfed = np.zeros((self.im_height, self.im_width));
        self.final_yield_irrig = np.zeros((self.im_height, self.im_width));
        self.crop_calender = np.zeros((self.im_height, self.im_width));

        for i_row in range(self.im_height):

            if self.set_mask:
                print('Done: ' + str(int((count_pixel_completed/np.sum(self.im_mask!=self.nodata_val))*100)) + ' %')
            else:
                print('Done: ' + str(int((count_pixel_completed/(self.im_height*self.im_width))*100)) + ' %')

            for i_col in range(self.im_width):

                # check current location (pixel) is outside of study area or not. if it's outside of study area goes to next location (pixel)
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
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

                    minT_daily_point = obj_utilities.interpMonthlyToDaily(self.minT_monthly[i_row, i_col,:], 1, days_in_year)
                    maxT_daily_point = obj_utilities.interpMonthlyToDaily(self.maxT_monthly[i_row, i_col,:], 1, days_in_year)
                    shortRad_daily_point = obj_utilities.interpMonthlyToDaily(self.shortRad_monthly[i_row, i_col,:],  1, days_in_year, no_minus_values=True)
                    wind2m_daily_point = obj_utilities.interpMonthlyToDaily(self.wind2m_monthly[i_row, i_col,:],  1, days_in_year, no_minus_values=True)
                    totalPrec_daily_point = obj_utilities.interpMonthlyToDaily(self.totalPrec_monthly[i_row, i_col,:],  1, days_in_year, no_minus_values=True)
                    rel_humidity_daily_point = obj_utilities.interpMonthlyToDaily(self.rel_humidity_monthly[i_row, i_col,:],  1, days_in_year, no_minus_values=True)
                else:
                    minT_daily_point = self.minT_daily[i_row, i_col,:]
                    maxT_daily_point = self.maxT_daily[i_row, i_col,:]
                    shortRad_daily_point = self.shortRad_daily[i_row, i_col,:]
                    wind2m_daily_point = self.wind2m_daily[i_row, i_col,:]
                    totalPrec_daily_point = self.totalPrec_daily[i_row, i_col,:]
                    rel_humidity_daily_point = self.rel_humidity_daily[i_row, i_col,:]

                # calculate ETO for full year for particular location (pixel)
                obj_eto = ETOCalc.ETOCalc(1, minT_daily_point.shape[0], self.latitude_map[i_row, i_col], self.elevation[i_row, i_col])
                shortRad_dailyy_point_MJm2day = (shortRad_daily_point*3600*24)/1000000 # convert w/m2 to MJ/m2/day
                obj_eto.setClimateData(minT_daily_point, maxT_daily_point, wind2m_daily_point, shortRad_dailyy_point_MJm2day, rel_humidity_daily_point)
                pet_daily_point = obj_eto.calculateETO()

                # list that stores yield estimations of all cycles per particular location (pixel)
                yield_of_all_crop_cycles_rainfed = []
                yield_of_all_crop_cycles_irrig = []

                for i_cycle in range(start_doy, end_doy+1, step_doy):

                    # just repeat data for year 2 times, to simulate for a entire year. just for computational convenient
                    minT_daily_2year = np.tile(minT_daily_point, 2)
                    maxT_daily_2year = np.tile(maxT_daily_point, 2)
                    shortRad_daily_2year = np.tile(shortRad_daily_point, 2)
                    wind2m_daily_2year = np.tile(wind2m_daily_point, 2)
                    totalPrec_daily_2year = np.tile(totalPrec_daily_point, 2)
                    pet_daily_2year = np.tile(pet_daily_point, 2)

                    # extract climate data within the season to pass in to calculation classes
                    minT_daily_season = minT_daily_2year[i_cycle : i_cycle+self.cycle_len]
                    maxT_daily_season = maxT_daily_2year[i_cycle : i_cycle+self.cycle_len]
                    shortRad_daily_season = shortRad_daily_2year[i_cycle : i_cycle+self.cycle_len]
                    wind2m_daily_season = wind2m_daily_2year[i_cycle : i_cycle+self.cycle_len]
                    totalPrec_daily_season = totalPrec_daily_2year[i_cycle : i_cycle+self.cycle_len]
                    pet_daily_season = pet_daily_2year[i_cycle : i_cycle+self.cycle_len]

                    # conduct tests to check simulation should be carried out or not based on growing period threshold. if not, goes to next location (pixel)
                    obj_screening = ThermalScreening_mod_Sh.ThermalScreening()
                    obj_screening.setClimateData(minT_daily_season, maxT_daily_season)

                    if self.set_tclimate_screening:
                        obj_screening.setThermalClimateScreening(self.t_climate[i_row, i_col], self.no_t_climate)
                    if self.set_lgpt_screening:
                        obj_screening.setLGPTScreening(self.no_lgpt, self.optm_lgpt)
                    if self.set_Tsum_screening:
                        # print("hey alright till")
                        # obj_screening.setTSumScreening(self.no_Tsum, self.optm_Tsum)
                        obj_screening.SetTSumScreening(self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO)
                    if self.set_Tprofile_screening:

                        obj_screening.setTypeB(self.thermalscreeningrules)

                        # obj_screening.setTProfileScreening(self.no_Tprofile, self.optm_Tprofile)
                        # chnage to read from csv
                        

                    thermal_screening_f = 1
                    if not obj_screening.getSuitability():
                        continue
                    else:
                        # print("going apply reduction factor")
                        thermal_screening_f = obj_screening.getReductionFactor()

                    # calculate biomass
                    obj_maxyield = BioMassCalc.BioMassCalc(i_cycle, i_cycle+self.cycle_len-1, self.latitude_map[i_row, i_col])
                    obj_maxyield.setClimateData(minT_daily_season, maxT_daily_season, shortRad_daily_season)

                    if self.is_perennial:
                        # Calculation biomass for rainfed condition
                        # print("--------------")
                        # print(np.mean(self.LAi_rainfed), np.mean(self.HI_rainfed), self.legume, self.adaptability)
                        obj_maxyield.setCropParameters(self.LAi_rainfed[i_row, i_col]+1, self.HI_rainfed[i_row, i_col]+1, self.legume, self.adaptability)
                        obj_maxyield.calculateBioMass()
                        est_yield_rainfed = obj_maxyield.calculateYield()

                        # reduce thermal screening factor
                        est_yield_rainfed = est_yield_rainfed * thermal_screening_f

                        # apply cropwat
                        obj_cropwat = CropWatCalc.CropWatCalc(i_cycle, i_cycle+self.cycle_len-1)
                        obj_cropwat.setClimateData(pet_daily_season, totalPrec_daily_season)
                        # check Sa is a raster or single value and extract Sa value accordingly
                        if len(np.array(self.Sa).shape) == 2:
                            Sa_temp = self.Sa[i_row, i_col]
                        else:
                            Sa_temp = self.Sa
                        obj_cropwat.setCropParameters(self.d_per, self.kc, self.kc_all, self.yloss_f, self.yloss_f_all, est_yield_rainfed, self.D1, self.D2, Sa_temp, self.pc)
                        est_yield_moisture_limited = obj_cropwat.calculateMoistureLimitedYield()

                        # append current cycle yield to a list
                        yield_of_all_crop_cycles_rainfed.append( est_yield_moisture_limited )
                        
                        # Calculation biomass for rainfed condition
                        obj_maxyield.setCropParameters(self.LAi_irrigated[i_row, i_col]+1, self.HI_irrigated[i_row, i_col]+1, self.legume, self.adaptability)
                        obj_maxyield.calculateBioMass()
                        est_yield_irrigated = obj_maxyield.calculateYield()

                        # reduce thermal screening factor
                        est_yield_irrigated = est_yield_irrigated * thermal_screening_f
                        yield_of_all_crop_cycles_irrig.append( est_yield_irrigated )
                    
                    else:
                        obj_maxyield.setCropParameters(self.LAi, self.HI, self.legume, self.adaptability)
                        obj_maxyield.calculateBioMass()
                        est_yield = obj_maxyield.calculateYield()

                        # reduce thermal screening factor
                        est_yield = est_yield * thermal_screening_f

                        # apply cropwat
                        obj_cropwat = CropWatCalc.CropWatCalc(i_cycle, i_cycle+self.cycle_len-1)
                        obj_cropwat.setClimateData(pet_daily_season, totalPrec_daily_season)
                        # check Sa is a raster or single value and extract Sa value accordingly
                        if len(np.array(self.Sa).shape) == 2:
                            Sa_temp = self.Sa[i_row, i_col]
                        else:
                            Sa_temp = self.Sa
                        obj_cropwat.setCropParameters(self.d_per, self.kc, self.kc_all, self.yloss_f, self.yloss_f_all, est_yield, self.D1, self.D2, Sa_temp, self.pc)
                        est_yield_moisture_limited = obj_cropwat.calculateMoistureLimitedYield()

                        # append current cycle yield to a list
                        yield_of_all_crop_cycles_rainfed.append( est_yield_moisture_limited )
                        yield_of_all_crop_cycles_irrig.append( est_yield )

                    # get maximum yield from all simulation for a particular location (pixel) and assign to final map
                    if len(yield_of_all_crop_cycles_irrig) > 0:
                        self.final_yield_rainfed[i_row, i_col] = np.max(yield_of_all_crop_cycles_rainfed)
                        self.final_yield_irrig[i_row, i_col] = np.max(yield_of_all_crop_cycles_irrig)
                        self.crop_calender[i_row, i_col] = np.where(yield_of_all_crop_cycles_rainfed==np.max(yield_of_all_crop_cycles_rainfed))[0][0] * step_doy
                        

        print('Simulations Completed !')

    def getEstimatedYieldRainfed(self):
        return self.final_yield_rainfed

    def getEstimatedYieldIrrigated(self):
        return self.final_yield_irrig

    def getOptimumCycleStartDate(self):
        return self.crop_calender
