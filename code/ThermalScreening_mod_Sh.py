"""
PyAEZ
Written by N. Lakmal Deshapriya
"""

from code.UtilitiesCalc import UtilitiesCalc
import numpy as np
import UtilitiesCalc as ut
import pandas as pd

class ThermalScreening(object):

    def __init__(self):
        self.set_tclimate_screening = False
        self.set_lgpt_screening = False
        self.set_Tsum_screening = False
        self.set_Tprofile_screening = False


    def getThermalLGP0(self):
        return np.sum(self.meanT_daily>0)

    def getThermalLGP5(self):
        return np.sum(self.meanT_daily>5)

    def getThermalLGP10(self):
        return np.sum(self.meanT_daily>10)

    def getTemperatureSum0(self):
        tempT = self.meanT_daily
        tempT[tempT<=0] = 0
        return np.sum(tempT)

    def getTemperatureSum5(self):
        tempT = self.meanT_daily
        tempT[tempT<=5] = 0
        return np.sum(tempT)

    def getTemperatureSum10(self):
        tempT = self.meanT_daily
        tempT[tempT<=10] = 0
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
        
        self.meanT_daily = (minT_daily + maxT_daily) / 2

        self.lgp0 = self.getThermalLGP0()
        self.lgp5 = self.getThermalLGP5()
        self.lgp10 = self.getThermalLGP10()

        self.tsum0 = self.getTemperatureSum0()
        self.tsum5 = self.getTemperatureSum5()
        self.tsum10 = self.getTemperatureSum10()

        self.tprofile = self.getTemperatureProfile()

    def setThermalClimateScreening(self, t_climate, no_t_climate):
        self.t_climate = t_climate
        self.no_t_climate = no_t_climate # list of unsuitable thermal climate

        self.set_tclimate_screening = True

    def setLGPTScreening(self, no_lgpt, optm_lgpt):
        self.no_lgpt = no_lgpt
        self.optm_lgpt = optm_lgpt

        self.set_lgpt_screening = True

    # def setTSumScreening(self, no_Tsum, optm_Tsum):
    #     self.no_Tsum = no_Tsum
    #     self.optm_Tsum = optm_Tsum

    #     self.set_Tsum_screening = True

    def SetTSumScreening(self, LnS, LsO, LO, HnS, HsO, HO):
        self.LnS=LnS
        self.LsO=LsO
        self.LO=LO
        self.HnS=HnS
        self.HsO=HsO
        self.HO=HO
        
        self.set_Tsum_screening = True

    
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

         monthly = UtilitiesCalc.UtilitiesCalc()
         self.RHmonthly = monthly.averageDailyToMonthly(self.RH)
         self.RHmin = np.amin( self.RHmonthly)

         self.DTR = monthly.averageDailyToMonthly(self.max_temp_daily) - monthly.averageDailyToMonthly(self.min_temp_daily)
         self.DTRhigh = np.amax( self.DTR)
        



#--------------------------------------------------------------------------------------------------------------------
    """Sriram you can find pesudocode below"""
    def setTypeB(self, formula, opr, optm, soptm, notsuitable, is_perennial):
        self.formula= formula
        self.opr = opr
        self.optm = optm
        self.soptm = soptm
        self.notsuitable = notsuitable 
        T_profile = self.getTemperatureProfile()
        if is_perennial:
            N9a = T_profile[0]
            N9b = T_profile[1]
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

        LGP0 = self.getThermalLGP0()
        LGP5 = self.getThermalLGP10()
        LGP10 = self.getThermalLGP10()

        self.cal_value = []
        for i in range (len(formula)):
            self.cal_value.append(eval(formula[i]))
            self.set_typeBconstraint= True
        #      read the comparsion operator
        #      return calculated values 


        
#-----------------------------------------------------------------------------------------------------------------------------



    def getSuitability(self):

        if self.set_tclimate_screening:
            if self.t_climate in self.no_t_climate:
                return False
    # will be screened with the help of CSV file
        # if self.set_lgpt_screening:
        #     if self.lgp0<=self.no_lgpt[0] or self.lgp5<=self.no_lgpt[1] or self.lgp10<=self.no_lgpt[2]:
        #         return False
        
        if self.set_Tsum_screening:
            
            # check with thieleng ****
            if (self.tsum0 > self.HnS[0] or self.tsum0 < self.LnS[0]) or (self.tsum5 > self.HnS[1] or self.tsum5 < self.LnS[1]) or (self.tsum10 > self.HnS[2] or self.tsum10 < self.LnS[2]):
                return False

    # will be screened with the help CSV file
        # if self.set_Tprofile_screening:
        #     for i1 in range(len(self.tprofile)):
        #         if self.tprofile[i1] <= self.no_Tprofile[i1]:
        #             return False

        return True

#------------------------------------------------------------------------------------------------------------------------

        # """Sriram this is for Suitablility test"""
    

                    
      
#-----------------------------------------------------------------------------------------------------------------------


            


    def getReductionFactor(self):

        thermal_screening_f = 1

        # print("reducition applying")
        # if self.set_lgpt_screening:

        #     # fall under typB so will be calcualated in next function

        #     if self.lgp0 < self.optm_lgpt[0]:
        #         f1 = ((self.lgp0-self.no_lgpt[0])/(self.optm_lgpt[0]-self.no_lgpt[0])) * 0.75 + 0.25
        #         thermal_screening_f = np.min([f1,thermal_screening_f])

        #     if self.lgp5 < self.optm_lgpt[1]:
        #         f1 = ((self.lgp5-self.no_lgpt[1])/(self.optm_lgpt[1]-self.no_lgpt[1])) * 0.75 + 0.25
        #         thermal_screening_f = np.min([f1,thermal_screening_f])

        #     if self.lgp10 < self.optm_lgpt[2]:
        #         f1 = ((self.lgp10-self.no_lgpt[2])/(self.optm_lgpt[2]-self.no_lgpt[2])) * 0.75 + 0.25
        #         thermal_screening_f = np.min([f1,thermal_screening_f])
        


        # if self.set_Tsum_screening:

        #     if self.tsum0 < self.optm_Tsum[0]:
        #         f1 = ((self.tsum0-self.no_Tsum[0])/(self.optm_Tsum[0]-self.no_Tsum[0])) * 0.75 + 0.25
        #         thermal_screening_f = np.min([f1,thermal_screening_f])

        #     if self.tsum5 < self.optm_Tsum[1]:
        #         f1 = ((self.tsum5-self.no_Tsum[1])/(self.optm_Tsum[1]-self.no_Tsum[1])) * 0.75 + 0.25
        #         thermal_screening_f = np.min([f1,thermal_screening_f])

        #     if self.tsum10 < self.optm_Tsum[2]:
        #         f1 = ((self.tsum10-self.no_Tsum[2])/(self.optm_Tsum[2]-self.no_Tsum[2])) * 0.75 + 0.25
        #         thermal_screening_f = np.min([f1,thermal_screening_f])
        

        '''the modified reduction factor for T_sum'''
        # print("checking for lololloo ")
        if self.set_Tsum_screening:
            # print("checking for 0, 5, and 10")
            if self.tsum0 > self.LsO[0] and self.tsum0 < self.LO[0] :
                f1 = ((self.tsum0-self.LsO[0])/(self.LO[0]-self.LsO[0])) * 0.25 + 0.75
            elif self.tsum0 > self.HO[0] and self.tsum0 < self.HsO[0]:
                f1 = ((self.tsum0-self.HO[0])/(self.HsO[0]-self.HO[0])) * 0.25 + 0.75
            elif self.tsum0 > self.LnS[0] and self.tsum0 < self.LsO[0]:
                f1 = ((self.tsum0-self.LnS[0])/(self.LsO[0]-self.LnS[0])) * 0.75
            elif self.tsum0 > self.HsO[0] and self.tsum0 < self.HnS[0]:
                f1=((self.tsum0-self.HsO[0])/(self.HnS[0]-self.HsO[0])) * 0.75
            elif self.tsum0 > self.LO[0] and self.tsum0 < self.HO[0]:
                f1 = 1
                # print("im right")

            # confirm with thieleng ***
            thermal_screening_f = np.min([f1,thermal_screening_f])
            
            if self.tsum5 > self.LsO[1] and self.tsum5 < self.LO[1] :
                f1 = ((self.tsum5-self.LsO[1])/(self.LO[1]-self.LsO[1])) * 0.25 + 0.75
            elif self.tsum5 > self.LnS[1] and self.tsum5 < self.LsO[1]:
                f1=((self.tsum5-self.LnS[1])/(self.LsO[1]-self.LnS[1])) * 0.75
            elif self.tsum5 > self.HO[1] and self.Tsum5 < self.HsO[1]:
                f1 = ((self.tsum5-self.HO[1])/(self.HsO[1]-self.HO[1])) * 0.25 + 0.75
            elif self.tsum5 > self.HsO[1] and self.tsum5 < self.HnS[1]:
                f1=((self.tsum5-self.HsO[1])/(self.HnS[1]-self.HsO[1])) * 0.75

            # confirm with thieleng ***
            thermal_screening_f = np.min([f1,thermal_screening_f])

            if self.tsum10 > self.LsO[2] and self.tsum10 < self.LO[2] :
                f1 = ((self.tsum10-self.LsO[2])/(self.LO[2]-self.LsO[2])) * 0.25 + 0.75
            elif self.tsum10 > self.LnS[2] and self.tsum10 < self.LsO[2]:
                f1=((self.tsum0-self.LnS[2])/(self.LsO[2]-self.LnS[2])) * 0.75
            elif self.tsum10 > self.HO[2] and self.Tsum10 < self.HsO[2]:
                f1 = ((self.tsum10-self.HO[2])/(self.HsO[2]-self.HO[2])) * 0.25 + 0.75
            elif self.tsum10 > self.HsO[2] and self.tsum10 < self.HnS[2]:
                f1=((self.tsum10-self.HsO[2])/(self.HnS[2]-self.HsO[2])) * 0.75
        
            # confirm with thieleng ***
            thermal_screening_f = np.min([f1,thermal_screening_f])        

        #if self.set_Tprofile_screening:

           # for i1 in range(len(tprofile)):
                #if self.tprofile[i1] < self.optm_Tprofile[i1]:
                    #f1 = ((self.tprofile[i1]-self.no_Tprofile[i1])/(self.optm_Tprofile[i1]-self.no_Tprofile[i1])) * 0.75 + 0.25
                    #thermal_screening_f = np.min([f1,thermal_screening_f])
        if self.setTypeB:
            for i1 in range (len(self.cal_value)):
                if self.optm[i1] == self.notsuitable[i1]:
                    f1 = 1
                elif self.opr[i1] == '<=' :
                    if self.optm[i1] != self.soptm[i1] and self.soptm[i1] != self.notsuitable[i1]:
                        if self.cal_value[i1] >= self.optm [i1] and self.cal_value[i1] <= self.soptm[i1]:
                            f1 = ((self.cal_value[i1]-self.optm[i1])/(self.soptm[i1]-self.optm[i1])) * 0.25 + 0.75
                        elif self.cal_value[i1] >= self.soptm and self.cal_value[i1] <= self.notsuitable[i1]:
                            f1=((self.cal_value[i1]-self.soptm[i1])/(self.notsuitable[i1]-self.soptm[i1])) * 0.75
                    elif self.optm[i1] != self.soptm[i1] and self.soptm[i1] == self.notsuitable[i1]:
                        f1 = ((self.cal_value[i1]-self.optm[i1])/(self.soptm[i1]-self.optm[i1])) * 0.25 + 0.75
                
                elif self.opr[i1] == '>=':
                     if self.optm[i1] != self.soptm[i1] and self.soptm[i1] != self.notsuitable[i1]:
                         if self.cal_value[i1] <= self.optm [i1] and self.cal_value[i1] >= self.soptm[i1]:
                            f1 = ((self.cal_value[i1]-self.soptm[i1])/(self.optm[i1]-self.soptm[i1])) * 0.25 + 0.75
                         elif self.cal_value[i1] <= self.soptm and self.cal_value[i1] >= self.notsuitable[i1]:
                            f1=((self.cal_value[i1]-self.notsuitable[i1])/(self.soptm[i1]-self.notsuitable[i1])) * 0.75
                     elif self.optm[i1] != self.soptm[i1] and self.soptm[i1] == self.notsuitable[i1]:
                        f1 = ((self.cal_value[i1]-self.optm[i1])/(self.soptm[i1]-self.optm[i1])) * 0.25 + 0.75
                thermal_screening_f = np.min([f1,thermal_screening_f])



        return thermal_screening_f

       
