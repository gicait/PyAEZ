"""
PyAEZ
Written by N. Lakmal Deshapriya
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio

#from osgeo import gdal
import pandas as pd
import io


import UtilitiesCalc
import BioMassCalc
import ETOCalc
import CropWatCalc
import ThermalScreening
import ClimateRegime 

class CropSimulation(object):

    def __init__(self):
        self.set_mask = False
        self.set_tclimate_screening = False
        self.set_lgpt_screening = False
        self.set_Tsum_screening = False
        self.set_Tprofile_screening = False
        self.adjustment = False
      

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

    
    def adjustmentParameterPerennial(self, path):
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
        #reading the csv file 
        p_df = pd.read_csv(path)

        # self.cyc_eff_rainfed= self.LGP
        # if self.min_temp == 5:
        #     self.cyc_eff_irrigated = self.LGPT5
        # elif self.min_temp == 10:
        #     self.cyc_eff_irrigated = self.LGPT10

        perennial_df_index = p_df.index[p_df['Crop_name'] == self.crop_name].to_list()[0]
        perennial_df = p_df.loc[p_df['Crop_name'] == self.crop_name]
        self.aLAI=perennial_df['aLAI'][perennial_df_index] 
        self.bLAI=perennial_df['bLAI'][perennial_df_index] 
        self.aHI=perennial_df['aHI'][perennial_df_index]
        self.bHI=perennial_df['bHI'][perennial_df_index]
        
            
        


    def setCropParametersFromCSV(self, file_path, crop_name, mode):
        self.crop_name = crop_name
        df = pd.read_csv(file_path)
        
        crop_df_index = df.index[df['Crop_name'] == crop_name].tolist()[0]
        crop_df = df.loc[df['Crop_name'] == crop_name]
        print("index:", crop_df_index)
        print(crop_df['D2'][crop_df_index])
        if mode == 'H':
            LAI =crop_df['H_LAI'][crop_df_index]
            HI=crop_df['H_HI'][crop_df_index]
        elif mode == 'M':
            LAI=crop_df['M_LAI'][crop_df_index]
            HI=crop_df['M_HI'][crop_df_index]
        elif mode == 'L':
            LAI=crop_df['L_LAI'][crop_df_index]
            HI=crop_df['L_HI'][crop_df_index]
        else:
            print('The output mode is incorrect. Please assign H for high, M for medium and L for low')
        self.setCropParameters(LAI, HI, legume=crop_df['legume'][crop_df_index], adaptability=int(crop_df['adaptability'][crop_df_index]), cycle_len=int(crop_df['cycle_len'][crop_df_index]), D1=crop_df['D1'][crop_df_index], D2=crop_df['D2'][crop_df_index], min_temp=crop_df['min_temp'][crop_df_index])

        LnS = crop_df['LnS'][crop_df_index]
        LsO = crop_df['LsO'][crop_df_index]
        Lo = crop_df['LO'][crop_df_index]  
        HnS = crop_df['HnS'][crop_df_index]  
        HsO = crop_df['HsO'][crop_df_index]  
        Ho = crop_df['HO'][crop_df_index]
        self.setAccTsum(LnS, LsO, Lo, HnS, HsO, Ho)
        self.setCropCycleParameters(stage_per=[crop_df['stage_per_1'][crop_df_index], crop_df['stage_per_2'][crop_df_index], crop_df['stage_per_3'][crop_df_index], crop_df['stage_per_4'][crop_df_index]], kc=[crop_df['kc_1'][crop_df_index], crop_df['kc_2'][crop_df_index], crop_df['kc_3'][crop_df_index]], kc_all=crop_df['kc_all'][crop_df_index], yloss_f=[crop_df['yloss_f1'][crop_df_index], crop_df['yloss_f2'][crop_df_index], crop_df['yloss_f3'][crop_df_index], crop_df['yloss_f4'][crop_df_index]], yloss_f_all=crop_df['yloss_f_all'][crop_df_index])
        self.is_perennial = crop_df['annual/perennial flag'][crop_df_index]


    def setSoilWaterParameters(self, Sa, pc):
        self.Sa = Sa  # available soil moisture holding capacity (mm/m) , assumption
        self.pc = pc  # soil water depletion fraction below which ETa < ETo (from literature)

    '''All of bellow settings are optional'''

    # set mask of study area, this is optional
    def setStudyAreaMask(self, admin_mask, no_data_value):
        self.im_mask = admin_mask
        self.nodata_val = no_data_value

        self.set_mask = True

    def adjustForPerennialCrop_rainfed(self,  aLAI, bLAI, aHI, bHI):
        
        self.LAi_rainfed = self.LAi * ((self.cyc_eff_rainfed-aLAI)/bLAI) # leaf area index adjustment for perennial crops
        self.HI_rainfed = self.HI * ((self.cyc_eff_rainfed-aHI)/bHI) # harvest index adjustment for perennial crops
    
    def adjustForPerennialCrop_irrigated(self,  aLAI, bLAI, aHI, bHI):
        
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
    
    def ReadThermalScreeningRulesFromCSV(self, path):
        df = pd.read_csv(path)
        crop_df_index = df.index[df['Crop'] == self.crop_name].tolist()
        crop_df = df.loc[df['Crop'] == self.crop_name]
        
        self.formula =crop_df['Constraint'][crop_df_index].to_numpy() 
        self.opr = crop_df['Type'][crop_df_index].to_numpy()
        self.optm = crop_df['Optimal'][crop_df_index].to_numpy()
        self.soptm = crop_df['Sub-Optimal'][crop_df_index].to_numpy()
        self.notsuitable= crop_df['Not-Suitable'][crop_df_index].to_numpy()

        self.set_type_B = True
        #self.thermalscreeningrules= crop_df.to_numpy()
        
        
    # def thermalscreeningfunction(self, cycle_len, start_doy, minT_daily_season, maxT_daily_season, i_row, i_col, rel_humidity_daily_point, minT_daily_point, maxT_daily_point):
    #     obj_screening = ThermalScreening.ThermalScreening()
    #     self.cyc_len =cycle_len
    #     start_doy = start_doy
    #     obj_screening.setparamerter(self.cyc_len, start_doy)
    #     obj_screening.setClimateData(minT_daily_season, maxT_daily_season)
    #     if self.set_tclimate_screening:
    #          obj_screening.setThermalClimateScreening(self.t_climate[i_row, i_col], self.no_t_climate)
        
    #     if self.set_type_B:
    #         if self.is_perennial:
    #             obj_screening.set_RH_and_DT(rel_humidity_daily_point, minT_daily_point, maxT_daily_point)
    #         obj_screening.setTypeB(self.formula, self.opr, self.optm, self.soptm, self.notsuitable, self.is_perennial)
        
        


       

    def simulateCropCycle(self, start_doy=1, end_doy=365, step_doy=1, leap_year=False):

        # just a counter to keep track of progress
        count_pixel_completed = 0
        # this allows handing leap and non-leap year differently. This is only relevant for monthly data because this value will be used in interpolations.
        # In case of daily data, length of vector will be taken as number of days in  a year.
        if leap_year:
            days_in_year = 366
        else:
            days_in_year = 365

        # this stores final result
        self.final_yield_rainfed = np.zeros((self.im_height, self.im_width));
        self.final_yield_irrig = np.zeros((self.im_height, self.im_width));
        self.crop_calender = np.zeros((self.im_height, self.im_width));
        self.reductionfactorF1= np.zeros((self.im_height,self.im_width));
        self.water_reductionF2= np.zeros((self.im_height, self.im_width));
       
        

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
                    self.totalPrec_daily_season = totalPrec_daily_2year[i_cycle : i_cycle+self.cycle_len]
                    self.pet_daily_season = pet_daily_2year[i_cycle : i_cycle+self.cycle_len]
                    
                    #checking if it is perinal crop or not 
                    if self.is_perennial:
                        if self.LGP[i_row, i_col] < self.cycle_len :
                            self.adjustment = True
                            self.cyc_eff_rainfed = self.LGP[i_row, i_col]
                            self.adjustForPerennialCrop_rainfed(self.aLAI, self.bLAI, self.aHI, self.bHI)
                        else:
                            self.cyc_eff_rainfed = self.cycle_len
                            self.LAi_rainfed = self.LAi
                            self.HI_rainfed = self.HI

                        if self.min_temp == 5:

                            if self.LGPT5[i_row, i_col] < self.cycle_len:
                                self.adjustment = True
                                self.cyc_eff_irrigated = self.LGPT5
                                self.adjustForPerennialCrop_irrigated(self.aLAI, self.bLAI, self.aHI, self.bHI)
                            else:
                                self.cyc_eff_irrigated = self.cycle_len
                                self.LAi_irrigated = self.LAi
                                self.HI_irrigated = self.HI

                        else:
                            if self.LGPT10[i_row, i_col] < self.cycle_len:
                                self.cyc_eff_irrigated = self.LGPT10
                                self.adjustForPerennialCrop_irrigated(self.aLAI, self.bLAI, self.aHI, self.bHI)
                            else:
                                self.cyc_eff_irrigated = self.cycle_len
                                self.LAi_irrigated = self.LAi
                                self.HI_irrigated = self.HI
                

                    # conduct tests to check simulation should be carried out or not based on growing period threshold. if not, goes to next location (pixel)
                    
                    obj_screening = ThermalScreening.ThermalScreening()
                    obj_screening.setparamerter(self.cycle_len, start_doy)
                    

                    if self.adjustment:         #if the perennial crop is adjusted then Cycle efficent might be different for irrigated and rainfed conditions so we pass both the value and check under thhe both situation
                        obj_screening.setparameteradjusted(self.cyc_eff_rainfed, self.cyc_eff_irrigated, start_doy)
                        
                   
                        

                    obj_screening.setClimateData(minT_daily_season, maxT_daily_season)
                    obj_screening.SetTSumScreening(self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO)
                    if self.set_tclimate_screening:
                        obj_screening.setThermalClimateScreening(self.t_climate[i_row, i_col], self.no_t_climate)
                    
                    if self.set_type_B:
                        if self.is_perennial:
                            obj_screening.set_RH_and_DT(rel_humidity_daily_point, minT_daily_point, maxT_daily_point)
                        obj_screening.setTypeB(self.formula, self.opr, self.optm, self.soptm, self.notsuitable, self.is_perennial)
                    
                    

                    self.reductionfactorF1[i_row, i_col] =1
                    if not  obj_screening.getSuitability():
                        print('value is not suitable thus yield is not calculated for the pixel')
                        continue
                    else:
                        #print("going apply reduction factor")
                        #thermal_screening_f = obj_screening.getReductionFactor()
                        self.reductionfactorF1[i_row, i_col] = obj_screening.getReductionFactor()
                        
                       

                    # calculate biomass
                    obj_maxyield = BioMassCalc.BioMassCalc(i_cycle, i_cycle+self.cycle_len-1, self.latitude_map[i_row, i_col])
                    obj_maxyield.setClimateData(minT_daily_season, maxT_daily_season, shortRad_daily_season)

                    if self.is_perennial:
                        obj_maxyield.setCropParameters(self.LAi_rainfed, self.HI_rainfed, self.legume, self.adaptability)
                        obj_maxyield.calculateBioMass()
                        est_yield_rainfed = obj_maxyield.calculateYield()
                        

                        # reduce thermal screening factor
                        est_yield_rainfed = est_yield_rainfed * self.reductionfactorF1[i_row, i_col]

                        # apply cropwat
                        obj_cropwat = CropWatCalc.CropWatCalc(i_cycle, i_cycle+self.cycle_len-1)
                        obj_cropwat.setClimateData(self.pet_daily_season, self.totalPrec_daily_season)

                        # check Sa is a raster or single value and extract Sa value accordingly
                        if len(np.array(self.Sa).shape) == 2:
                            Sa_temp = self.Sa[i_row, i_col]
                        else:
                            Sa_temp = self.Sa
                        obj_cropwat.setCropParameters(self.d_per, self.kc, self.kc_all, self.yloss_f, self.yloss_f_all, est_yield_rainfed, self.D1, self.D2, Sa_temp, self.pc)
                        est_yield_moisture_limited = obj_cropwat.calculateMoistureLimitedYield()
                        self.water_reductionF2 [i_row, i_col] = obj_cropwat.waterreduction()
                        self.Etc[i_row, i_col] = obj_cropwat.petcmap()

                        # append current cycle yield to a list
                        yield_of_all_crop_cycles_rainfed.append( est_yield_moisture_limited )
                        
                        # Calculation biomass for rainfed condition
                        obj_maxyield.setCropParameters(self.LAi_irrigated, self.HI_irrigated, self.legume, self.adaptability)
                        obj_maxyield.calculateBioMass()
                        est_yield_irrigated = obj_maxyield.calculateYield()

                        # reduce thermal screening factor
                        est_yield_irrigated = est_yield_irrigated * self.reductionfactorF1[i_row, i_col]
                        yield_of_all_crop_cycles_irrig.append( est_yield_irrigated )
                    
                    else:
                        obj_maxyield.setCropParameters(self.LAi, self.HI, self.legume, self.adaptability)
                        obj_maxyield.calculateBioMass()
                        est_yield = obj_maxyield.calculateYield()
                    

                        # reduce thermal screening factor
                        self.est_yield = est_yield * self.reductionfactorF1[i_row, i_col]
                        

                        # apply cropwat
                        obj_cropwat = CropWatCalc.CropWatCalc(i_cycle, i_cycle+self.cycle_len-1)
                        obj_cropwat.setClimateData(self.pet_daily_season, self.totalPrec_daily_season)
                        # check Sa is a raster or single value and extract Sa value accordingly
                        if len(np.array(self.Sa).shape) == 2:
                            Sa_temp = self.Sa[i_row, i_col]
                        else:
                            Sa_temp = self.Sa
                        obj_cropwat.setCropParameters(self.d_per, self.kc, self.kc_all, self.yloss_f, self.yloss_f_all, self.est_yield, self.D1, self.D2, Sa_temp, self.pc)
                        est_yield_moisture_limited = obj_cropwat.calculateMoistureLimitedYield()
                        self.water_reductionF2 [i_row, i_col] = obj_cropwat.waterreduction()
                        #self.Etc[i_row, i_col] = obj_cropwat.petcmap()
                        

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

    def getreductionFactor(self):
        reduction_factor = [self.reductionfactorF1 , self.water_reductionF2]
        return reduction_factor
    
    # def getEtocandEtcMap(self):
    #     sum_etomap= np.sum(self.Eto, axis = 2)
    #     sum_etc = np.sum(self.Etc, axis=2)
    #     etomap= [sum_etomap, sum_etc]
    #     #etomap = [self.Eto, self.Etc]
    #     return etomap
    def getEtoandEtcMap(self):
        crop_calender2 =( np.tile(self.crop_calender, 2))
        self.Eto = np.zeros((self.im_height, self.im_width, self.cycle_len+1));
        self.Etc = np.zeros((self.im_height, self.im_width, self.cycle_len+1));

        for i_row in range (self.im_height):
            for i_col in range (self.im_width):
                if self.set_monthly:
                    obj_utilities = UtilitiesCalc.UtilitiesCalc()

                    minT_daily_point = obj_utilities.interpMonthlyToDaily(self.minT_monthly[i_row, i_col,:],  crop_calender2[i_row, i_col], crop_calender2[i_row, i_col]+self.cycle_len)
                    maxT_daily_point = obj_utilities.interpMonthlyToDaily(self.maxT_monthly[i_row, i_col,:], crop_calender2[i_row, i_col], crop_calender2[i_row,i_col]+self.cycle_len)
                    shortRad_daily_point = obj_utilities.interpMonthlyToDaily(self.shortRad_monthly[i_row, i_col,:],  crop_calender2[i_row, i_col],crop_calender2[i_row,i_col]+ self.cycle_len, no_minus_values=True)
                    wind2m_daily_point = obj_utilities.interpMonthlyToDaily(self.wind2m_monthly[i_row, i_col,:], crop_calender2[i_row, i_col], crop_calender2[i_row,i_col]+self.cycle_len, no_minus_values=True)
                    totalPrec_daily_point = obj_utilities.interpMonthlyToDaily(self.totalPrec_monthly[i_row, i_col,:],  crop_calender2[i_row, i_col],crop_calender2[i_row,i_col]+ self.cycle_len, no_minus_values=True)
                    rel_humidity_daily_point = obj_utilities.interpMonthlyToDaily(self.rel_humidity_monthly[i_row, i_col,:], crop_calender2[i_row, i_col], crop_calender2[i_row,i_col]+self.cycle_len, no_minus_values=True)
                    
                else:
                    minT_daily_point = self.minT_daily[i_row, i_col,:]
                    maxT_daily_point = self.maxT_daily[i_row, i_col,:]
                    shortRad_daily_point = self.shortRad_daily[i_row, i_col,:] 
                    wind2m_daily_point = self.wind2m_daily[i_row, i_col,:]
                    totalPrec_daily_point = self.totalPrec_daily[i_row, i_col,:]
                    rel_humidity_daily_point = self.rel_humidity_daily[i_row, i_col,:]
                    
                
                # calculate ETO for cycle length period for particular location (pixel)
                obj_eto = ETOCalc.ETOCalc(int(crop_calender2[i_row, i_col]), int(crop_calender2[i_row,i_col])+self.cycle_len, self.latitude_map[i_row, i_col], self.elevation[i_row, i_col])
                shortRad_dailyy_point_MJm2day = (shortRad_daily_point*3600*24)/1000000 # convert w/m2 to MJ/m2/day
                obj_eto.setClimateData(minT_daily_point, maxT_daily_point, wind2m_daily_point, shortRad_dailyy_point_MJm2day, rel_humidity_daily_point)
                #pet_daily_point = obj_eto.calculateETO()
                #print (len(pet_daily_point))
                self.Eto[i_row, i_col] = obj_eto.calculateETO()
                
                #Calculating Value of Ect for cycle lenght period for particular location(pixel)
                for i_cycle in range(int(crop_calender2[i_row, i_col])):
                    # reduce thermal screening factor
                                               
                        # apply cropwat
                        obj_cropwat = CropWatCalc.CropWatCalc(i_cycle, i_cycle+self.cycle_len-1)
                        obj_cropwat.setClimateData(self.pet_daily_season, self.totalPrec_daily_season)
                        # check Sa is a raster or single value and extract Sa value accordingly
                        if len(np.array(self.Sa).shape) == 2:
                            Sa_temp = self.Sa[i_row, i_col]
                        else:
                            Sa_temp = self.Sa
                        obj_cropwat.setCropParameters(self.d_per, self.kc, self.kc_all, self.yloss_f, self.yloss_f_all, self.est_yield, self.D1, self.D2, Sa_temp, self.pc)
                        est_yield_moisture_limited = obj_cropwat.calculateMoistureLimitedYield()
                        self.water_reductionF2 [i_row, i_col] = obj_cropwat.waterreduction()
                        self.Etc[i_row, i_col] = obj_cropwat.petcmap()
        
        self.eto_sum = np.sum(self.Eto, axis = 2)
        self.etc_sum = np.sum(self.Etc, axis= 2)
        eto_etc_map = [self.eto_sum, self.etc_sum]
        return(eto_etc_map)

    def getreduction(self):
        return self.reductionfactorF1
