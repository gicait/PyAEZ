"""
PyAEZ
Written by N. Lakmal Deshapriya
"""


import numpy as np
import UtilitiesCalc
import pandas as pd

class ThermalScreening(object):

    def __init__(self ):        
        self.set_tclimate_screening = False
        self.set_lgpt_screening = False
        self.set_Tsum_screening = False
        self.set_Tprofile_screening = False
        self.set_typeBconstraint = False
        self.set_parameter_adjusted = False

    def setparamerter(self, cycle_len, Start_day):
        self.cycle_len = cycle_len
        self.start_day = Start_day

    def setparameteradjusted(self, cycle_len_rain, cycle_len_irri, Start_day):
        self.cycle_len_r = cycle_len_rain
        self.cycle_len_i = cycle_len_irri
        self.start_day = Start_day
        self.set_parameter_adjusted = True

    def getThermalLGP0(self):
        return np.sum(self.meanT_daily>0)

    def getThermalLGP5(self):
        return np.sum(self.meanT_daily>5)

    def getThermalLGP10(self):
        return np.sum(self.meanT_daily>10)

    def getTemperatureSum0(self, cycle_len):
        tempT = self.meanT_daily[self.start_day-1: self.start_day-1+cycle_len]
        # print(len(tempT))
        tempT[tempT<=0] = 0
        return np.sum(tempT)
      
     
    def getTemperatureProfile(self):

        meanT_daily_add1day = np.concatenate((self.meanT_daily, self.meanT_daily[0:1]))
        meanT_first = meanT_daily_add1day[:-1]
        meanT_diff = meanT_daily_add1day[1:] - meanT_daily_add1day[:-1]

        A9 = np.sum( np.logical_and(meanT_diff>0, meanT_first<-5) )
        A8 = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=-5, meanT_first<0)) )
        A7 = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=0, meanT_first<5)) )
        A6 = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=5, meanT_first<10)) )
        A5 = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=10, meanT_first<15)) )
        A4 = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=15, meanT_first<20)) )
        A3 = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=20, meanT_first<25)) )
        A2 = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=25, meanT_first<30)) )
        A1 = np.sum( np.logical_and(meanT_diff>0, meanT_first>=30) )

        B9 = np.sum( np.logical_and(meanT_diff<0, meanT_first<-5) )
        B8 = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=-5, meanT_first<0)) )
        B7 = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=0, meanT_first<5)) )
        B6 = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=5, meanT_first<10)) )
        B5 = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=10, meanT_first<15)) )
        B4 = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=15, meanT_first<20)) )
        B3 = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=20, meanT_first<25)) )
        B2 = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=25, meanT_first<30)) )
        B1 = np.sum( np.logical_and(meanT_diff<0, meanT_first>=30) )
 
        return [A9,A8,A7,A6,A5,A4,A3,A2,A1,B1,B2,B3,B4,B5,B6,B7,B8,B9]

    def setClimateData(self, minT_daily, maxT_daily):
       
        #self.cycle_len=cycle_len
        self.meanT_daily = (minT_daily + maxT_daily) / 2

        self.lgp0 = self.getThermalLGP0()
        self.lgp5 = self.getThermalLGP5()
        self.lgp10 = self.getThermalLGP10()
        self.tprofile = self.getTemperatureProfile()
        self.tsum0 = self.getTemperatureSum0(self.cycle_len)

        if self.set_parameter_adjusted: #if the perennial crop is adjusted we must check for both condition
            self.tsum0_r = self.getTemperatureSum0(self.cycle_len_r)
            self.tsum0_i = self.getTemperatureSum0(self.cycle_len_i)
        else:
            self.tsum0 = self.getTemperatureSum0(self.cycle_len)
            

    def setThermalClimateScreening(self, t_climate, no_t_climate):
        self.t_climate = t_climate
        self.no_t_climate = no_t_climate # list of unsuitable thermal climate

        self.set_tclimate_screening = True

    def setLGPTScreening(self, no_lgpt, optm_lgpt):
        self.no_lgpt = no_lgpt
        self.optm_lgpt = optm_lgpt

        self.set_lgpt_screening = True

 
    def SetTSumScreening(self, LnS, LsO, LO, HnS, HsO, HO):
        self.LnS=LnS
        self.LsO=LsO
        self.LO=LO
        self.HnS=HnS
        self.HsO=HsO
        self.HO=HO
        
        self.set_Tsum_screening = True
        #print ('working')

    
    def setTProfileScreening(self, no_Tprofile, optm_Tprofile):
        self.no_Tprofile = no_Tprofile
        self.optm_Tprofile = optm_Tprofile

        self.set_Tprofile_screening = True

#-----------------------------------------------------------------------------------------------------------------------------------

    def set_RH_and_DT(self, rel_humidity_daily_point, min_temp_daily, max_temp_daily):
         self.RH = rel_humidity_daily_point
         self.min_temp_daily = min_temp_daily
         self.max_temp_daily = max_temp_daily
         self.RHavg = np.average(self.RH)
         self.DTRavg = np.average(self.max_temp_daily) - np.average( self.min_temp_daily)
       


    #     #converting daily data into monthly using utilitiesCal module to calculate RHmin and DTRhigh

         obj_utilities = UtilitiesCalc.UtilitiesCalc()
         self.RHmonthly = obj_utilities.averageDailyToMonthly(self.RH)
         self.RHmin = np.amin( self.RHmonthly)

         self.DTR = obj_utilities.averageDailyToMonthly(self.max_temp_daily) - obj_utilities.averageDailyToMonthly(self.min_temp_daily)
         self.DTRhigh = np.amax( self.DTR)

        #print(self.RHavg, self.RHmin, self.DTRavg, self.DTRhigh)
        



#--------------------------------------------------------------------------------------------------------------------
    """Sriram you can find pesudocode below"""
    def setTypeB(self, formula, opr, optm, soptm, notsuitable, is_perennial):
        self.formula= formula
        self.opr = opr
        self.optm = []
        self.soptm = []
        self.notsuitable = []
        T_profile = self.getTemperatureProfile()
        
        #print('perrinal', is_perennial)
        #checking of variables 
        if is_perennial:           
            N9a = T_profile[0]
            N8a = T_profile[1]
            N7a = T_profile[2]
            N6a = T_profile[3]
            N5a = T_profile[4]
            N4a = T_profile[5]
            N3a = T_profile[6]
            N2a = T_profile[7]
            N1a = T_profile[8]

            N1b =T_profile[9]
            N2b = T_profile[10]
            N3b = T_profile[11]
            N4b = T_profile[12]
            N5b = T_profile[13]
            N6b = T_profile[14]
            N7b = T_profile[15]
            N8b = T_profile[16]
            N9b = T_profile[17]
            N1 = N1a + N1b
            N2 = N2a + N2b
            N3 = N3a + N3b
            N4 = N4a + N4b
            N5 = N5a + N5b
            N6 = N6a + N6b
            N7 = N7a + N7b
            N8 = N8a + N8b
            N9 = N9a + N9b
            RHavg = self.RH
            RHmin = self.RHmin
            DTRavg = self.DTRavg
            DTRhigh = self.DTRhigh
        else:
            L9a = T_profile[0]
            L8a = T_profile[1]
            L7a = T_profile[2]
            L6a = T_profile[3]
            L5a = T_profile[4]
            L4a = T_profile[5]
            L3a = T_profile[6]
            L2a = T_profile[7]
            L1a = T_profile[8]

            L1b =T_profile[9]
            L2b = T_profile[10]
            L3b = T_profile[11]
            L4b = T_profile[12]
            L5b = T_profile[13]
            L6b = T_profile[14]
            L7b = T_profile[15]
            L8b = T_profile[16]
            L9b = T_profile[17]

            L1 = L1a + L1b
            L2 = L2a + L2b
            L3 = L3a + L3b 
            L4 = L4a + L4b
            L5 = L5a + L5b
            L6 = L6a + L6b
            L7 = L7a + L7b
            L8 = L8a + L8b
            L9 = L9a + L9b   
            
        
        

        LGP0 = self.getThermalLGP0()
        LGP5 = self.getThermalLGP10()
        LGP10 = self.getThermalLGP10()
        self.cal_value = []
        for i in range (len(formula)):
            self.cal_value.append(eval(formula[i]))
            self.optm.append(float(optm[i]))
            self.soptm.append(float(soptm[i]))
            self.notsuitable.append(float(notsuitable[i]))

        #print(self.cal_value, T_profile[3])  
            
        self.set_typeBconstraint = True
    



        
#-----------------------------------------------------------------------------------------------------------------------------



    def getSuitability(self):

        if self.set_tclimate_screening:
            if self.t_climate in self.no_t_climate:
                print('tclimate')
                return False
    
        
        if self.set_Tsum_screening:
           # print('check tsum0')        
            # check with thieleng ****
            if (self.tsum0 > self.HnS or self.tsum0 < self.LnS) :
                print('tsum', self.tsum0, self.HnS, self.LnS)
                return False

        if self.set_typeBconstraint:
            for i1 in range(len(self.cal_value)):
                #print (self.opr[i1])
                if self.opr [i1] == '=' :
                    if self.cal_value[i1] != self.optm[i1]:
                        print('Type B', self.cal_value[i1], self.optm[i1])
                        
                        return False
                elif self.opr[i1] == '<=' :
                    if self.cal_value[i1]  <= self.optm[i1] or self.cal_value[i1] >= self.notsuitable[i1]:
                        print('Type B',self.optm[i1] ,'<=', self.cal_value[i1],'<=' , self.notsuitable[i1])
                        return False
                elif self.opr[i1] == '>=':
                    if self.cal_value[i1] >= self.optm[i1] or self.cal_value[i1] <= self.notsuitable[i1]:
                        print('Type B', self.optm[i1], self.cal_value[i1], self.notsuitable[i1])
                        return False
        
        return True    

                    
      
#-----------------------------------------------------------------------------------------------------------------------


            


    def getReductionFactor(self):

        thermal_screening_f = 1
        '''the modified reduction factor for T_sum'''
        
        if self.set_Tsum_screening:
            if self.set_parameter_adjusted:
                self.get_adjusted_reduction_factor(self.tsum0_i)
                self.get_adjusted_reduction_factor(self.tsum0_r)
            if not self.set_parameter_adjusted:
                print ('entering loop')
                if self.tsum0 >= self.LsO and self.tsum0 <= self.LO :
                    f1 = ((self.tsum0-self.LsO)/(self.LO-self.LsO)) * 0.25 + 0.75
                    thermal_screening_f = np.min([f1,thermal_screening_f])
                elif self.tsum0 >= self.HO and self.tsum0 <= self.HsO:
                    f1 = ((self.HsO-self.tsum0)/(self.HsO-self.HO)) * 0.25 + 0.75
                    thermal_screening_f = np.min([f1,thermal_screening_f])
                elif self.tsum0 >= self.LnS and self.tsum0 <= self.LsO:
                    f1 = ((self.tsum0-self.LnS)/(self.LsO-self.LnS)) * 0.75
                    thermal_screening_f = np.min([f1,thermal_screening_f])
                elif self.tsum0 >= self.HsO and self.tsum0 <= self.HnS:
                    f1=((self.HnS-self.tsum0)/(self.HnS-self.HsO)) * 0.75
                    thermal_screening_f = np.min([f1,thermal_screening_f])
                elif self.tsum0 >= self.LO and self.tsum0 <= self.HO:
                    f1 = 1
                thermal_screening_f = np.min([f1,thermal_screening_f])
                
                     
        if self.set_typeBconstraint:
            for i1 in range (len(self.cal_value)):
                if self.optm[i1] == self.notsuitable[i1]:
                    f1 = 1
                elif self.opr[i1] == '<=' :
                    if self.optm[i1] != self.soptm[i1] and self.soptm[i1] != self.notsuitable[i1]:
                        if self.cal_value[i1] >= self.optm [i1] and self.cal_value[i1] <= self.soptm[i1]:
                            f1 = ((self.cal_value[i1]-self.optm[i1])/(self.soptm[i1]-self.optm[i1])) * 0.25 + 0.75
                        elif self.cal_value[i1] >= self.soptm[i1] and self.cal_value[i1] <= self.notsuitable[i1]:
                            f1=((self.cal_value[i1]-self.soptm[i1])/(self.notsuitable[i1]-self.soptm[i1])) * 0.75
                    elif self.optm[i1] != self.soptm[i1] and self.soptm[i1] == self.notsuitable[i1]:
                        f1 = ((self.cal_value[i1]-self.optm[i1])/(self.soptm[i1]-self.optm[i1])) * 0.25 + 0.75
                
                elif self.opr[i1] == '>=':
                     if self.optm[i1] != self.soptm[i1] and self.soptm[i1] != self.notsuitable[i1]:
                         if self.cal_value[i1] <= self.optm [i1] and self.cal_value[i1] >= self.soptm[i1]:
                            f1 = ((self.cal_value[i1]-self.soptm[i1])/(self.optm[i1]-self.soptm[i1])) * 0.25 + 0.75
                         elif self.cal_value[i1] <= self.soptm[i1] and self.cal_value[i1] >= self.notsuitable[i1]:
                            f1=((self.cal_value[i1]-self.notsuitable[i1])/(self.soptm[i1]-self.notsuitable[i1])) * 0.75
                     elif self.optm[i1] != self.soptm[i1] and self.soptm[i1] == self.notsuitable[i1]:
                        f1 = ((self.cal_value[i1]-self.optm[i1])/(self.soptm[i1]-self.optm[i1])) * 0.25 + 0.75
                
                thermal_screening_f = np.min([f1,thermal_screening_f])
        return thermal_screening_f

#function to calculate F1 if the T_sum is different for adjusted perenial crop paramenter under irrigated and rainfed 
    def get_adjusted_reduction_factor(self, tsum0):
        if self.set_Tsum_screening:
            thermal_screening_f = 1
            
            if tsum0 > self.LsO and tsum0 < self.LO :
                f1 = ((tsum0-self.LsO)/(self.LO-self.LsO)) * 0.25 + 0.75
                
                thermal_screening_f = np.min([f1,thermal_screening_f])
            elif self.tsum0 >= self.HO and self.tsum0 <= self.HsO:
                f1 = ((self.HsO-self.tsum0)/(self.HsO-self.HO)) * 0.25 + 0.75
                
                thermal_screening_f = np.min([f1,thermal_screening_f])
            elif tsum0 > self.LnS and tsum0 < self.LsO:
                f1 = ((tsum0-self.LnS)/(self.LsO-self.LnS)) * 0.75
                
                thermal_screening_f = np.min([f1,thermal_screening_f])
            elif self.tsum0 >= self.HsO and self.tsum0 <= self.HnS:
                f1=((self.HnS-self.tsum0)/(self.HnS-self.HsO)) * 0.75
                thermal_screening_f = np.min([f1,thermal_screening_f])
                
                
            elif tsum0 > self.LO and tsum0 < self.HO:
                f1 = 1
            elif self.tsum0< self.LnS or self.tsum0 > self.HnS:
                f1=0
            
