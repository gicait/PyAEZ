"""
PyAEZ
Written by N. Lakmal Deshapriya and Thaileng Thol

Reference: http://oar.icrisat.org/198/1/316_2009_GTAE_55_Poten_obt_yield_in_SAT.pdf
"""

import numpy as np

class CropWatCalc(object):

    def __init__(self, cycle_begin, cycle_end):
        self.cycle_begin = cycle_begin
        self.cycle_end = cycle_end
        self.cycle_len = cycle_end - cycle_begin + 1

    def setClimateData(self, pet, precipitation):
        self.peto = pet
        self.Prec = precipitation

    def setCropParameters(self, stage_per, kc, kc_all, yloss_f, yloss_f_all, est_yield, D1, D2, Sa, pc):
        self.d_per = np.array(stage_per) # Percentage for D1, D2, D3, D4 stages
        self.kc = np.array(kc) # 3 crop water requirements for initial, reproductive, the end of the maturation stages
        self.kc_all = kc_all # crop water requirements for entire growth cycle
        self.yloss_f = np.array(yloss_f)  # yield loss for D1, D2, D3, D4
        self.yloss_f_all = yloss_f_all  # yield loss for entire growth cycle
        self.y_potential = est_yield  # potential yield
        self.D1 = D1  # rooting depth (m)
        self.D2 = D2  # rooting depth (m)
        self.Sa = Sa  # available soil moisture holding capacity (mm/m) , usually assume as 100
        self.pc = pc  # soil water depletion fraction below which ETa < ETo (from literature)

    def calculateMoistureLimitedYield(self):

        '''Convert Percentage of stage in the crop cycle to cumulative number of days'''

        d_days = np.round( self.cycle_len * (np.cumsum(self.d_per)/100) ).astype('int')

        '''Interpolate Rotting Depth (D)'''
        D = np.zeros(self.peto.size)
        D[d_days[1]:] = self.D2
        D[:d_days[1]] = self.D1 + ((self.D2-self.D1)/(d_days[1])) * np.arange(d_days[1])

        '''Lets calculate crop specific PET for each stage'''

        petc_stage = np.zeros(self.peto.size)
        petc_all = np.zeros(self.peto.size)

        for ii in range(self.cycle_len):

            ## each growing stage
            if ii <= d_days[0]:
                petc_stage[ii] = self.kc[0] * self.peto[ii]
            elif ii <= d_days[1]:
                kc_temp = self.kc[0] + (ii-d_days[0]) * ( (self.kc[1]-self.kc[0])/(d_days[1]-d_days[0]) )
                petc_stage[ii] = kc_temp * self.peto[ii]
            elif ii <= d_days[2]:
                petc_stage[ii] = self.kc[1] * self.peto[ii]
            else:
                kc_temp = self.kc[1] + (ii-d_days[2]) * ( (self.kc[2]-self.kc[1])/(d_days[3]-d_days[2]) )
                petc_stage[ii] = kc_temp * self.peto[ii]

            ## entire gorwing season
            petc_all[ii] = self.kc_all * self.peto[ii]


        '''Lets calculate Actual ET for each stage'''

        peta_stage = np.zeros(self.peto.size)
        peta_all = np.zeros(self.peto.size)

        ## each growing stage
        W = self.Sa*D[0] 
        for ii in range(self.cycle_len):

            if self.Prec[ii] >= petc_stage[ii]:
                peta_stage[ii] = petc_stage[ii]
            elif self.Prec[ii] + W >= self.Sa*D[ii]*(1-self.pc):
                peta_stage[ii] = petc_stage[ii]
            else:
                kk = (W+self.Prec[ii]) / (self.Sa*D[ii]*(1-self.pc))
                peta_stage[ii] = kk * petc_stage[ii]

            W = np.min([W+self.Prec[ii]-peta_stage[ii], self.Sa*D[ii]])
            if W<0: W=0

        ## entire gorwing season
        W = self.Sa*D[0] 
        for ii in range(self.cycle_len):

            if self.Prec[ii] >= petc_all[ii]:
                peta_all[ii] = petc_all[ii]
            elif self.Prec[ii] + W >= self.Sa*D[ii]*(1-self.pc):
                peta_all[ii] = petc_all[ii]
            else:
                kk = (W+self.Prec[ii]) / (self.Sa*D[ii]*(1-self.pc))
                peta_all[ii] = kk * petc_all[ii]

            W = np.min([W+self.Prec[ii]-peta_all[ii], self.Sa*D[ii]])
            if W<0: W=0

        '''Assess Yield Loss in entire growth cycle'''

        peta_all_sum = np.sum( peta_all )
        petc_all_sum = np.sum( petc_all )

        f0 = 1 - self.yloss_f_all * ( 1 - (peta_all_sum/petc_all_sum) )

        '''Assess Yield Loss in individual growth stages separately'''

        peta_d1 = np.sum( peta_stage[0:d_days[0]] )
        peta_d2 = np.sum( peta_stage[d_days[0]:d_days[1]] )
        peta_d3 = np.sum( peta_stage[d_days[1]:d_days[2]] )
        peta_d4 = np.sum( peta_stage[d_days[2]:d_days[3]] )

        petc_d1 = np.sum( petc_stage[0:d_days[0]] )
        petc_d2 = np.sum( petc_stage[d_days[0]:d_days[1]] )
        petc_d3 = np.sum( petc_stage[d_days[1]:d_days[2]] )
        petc_d4 = np.sum( petc_stage[d_days[2]:d_days[3]] )

        f1_d1 = 1 - self.yloss_f[0] * ( 1 - (peta_d1/petc_d1) )
        f1_d2 = 1 - self.yloss_f[1] * ( 1 - (peta_d2/petc_d2) )
        f1_d3 = 1 - self.yloss_f[2] * ( 1 - (peta_d3/petc_d3) )
        f1_d4 = 1 - self.yloss_f[3] * ( 1 - (peta_d4/petc_d4) )

        f1 = np.min([f1_d1,f1_d2,f1_d3,f1_d4]) # some references use product, some use minimum. here we use minimum as in Thailand report

        '''Use more severe of above two conditions determines final yield'''

        f_final = np.min([f0,f1])

        # to avoid, possible error
        if f_final < 0:
            f_final = 0
        if f_final > 1:
            f_final = 1

        y_water_limited = f_final * self.y_potential

        return y_water_limited
