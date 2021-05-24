"""
PyAEZ
Written by N. Lakmal Deshapriya and Thaileng Thol

Reference: http://oar.icrisat.org/198/1/316_2009_GTAE_55_Poten_obt_yield_in_SAT.pdf
"""

import numpy as np
import numba as nb

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

    @staticmethod
    @nb.jit(nopython=True)
    def calculateMoistureLimitedYieldNumba(cycle_begin, cycle_end, cycle_len, peto, Prec, d_per, kc, kc_all, yloss_f, yloss_f_all, y_potential, D1, D2, Sa, pc):
        '''Convert Percentage of stage in the crop cycle to cumulative number of days'''

        d_days = (cycle_len * (np.cumsum(d_per)/100)).astype(np.int_)
        #d_days = np.round( cycle_len * (np.cumsum(d_per)/100) ).astype(np.int_)

        '''Interpolate Rotting Depth (D)'''
        D = np.zeros(peto.size)
        D[d_days[1]:] = D2
        D[:d_days[1]] = D1 + ((D2-D1)/(d_days[1])) * np.arange(d_days[1])

        '''Lets calculate crop specific PET for each stage'''

        petc_stage = np.zeros(peto.size)
        petc_all = np.zeros(peto.size)

        for ii in range(cycle_len):

            ## each growing stage
            if ii <= d_days[0]:
                petc_stage[ii] = kc[0] * peto[ii]
            elif ii <= d_days[1]:
                kc_temp = kc[0] + (ii-d_days[0]) * ( (kc[1]-kc[0])/(d_days[1]-d_days[0]) )
                petc_stage[ii] = kc_temp * peto[ii]
            elif ii <= d_days[2]:
                petc_stage[ii] = kc[1] * peto[ii]
            else:
                kc_temp = kc[1] + (ii-d_days[2]) * ( (kc[2]-kc[1])/(d_days[3]-d_days[2]) )
                petc_stage[ii] = kc_temp * peto[ii]

            ## entire gorwing season
            petc_all[ii] = kc_all * peto[ii]

        '''Lets calculate Actual ET for each stage'''

        peta_stage = np.zeros(peto.size)
        peta_all = np.zeros(peto.size)

        ## each growing stage
        W = Sa*D[0]
        for ii in range(cycle_len):

            if Prec[ii] >= petc_stage[ii]:
                peta_stage[ii] = petc_stage[ii]
            elif Prec[ii] + W >= Sa*D[ii]*(1-pc):
                peta_stage[ii] = petc_stage[ii]
            else:
                kk = (W+Prec[ii]) / (Sa*D[ii]*(1-pc))
                peta_stage[ii] = kk * petc_stage[ii]

            W = np.min(np.array([W+Prec[ii]-peta_stage[ii], Sa*D[ii]]))
            if W<0: W=0

        ## entire gorwing season
        W = Sa*D[0]
        for ii in range(cycle_len):

            if Prec[ii] >= petc_all[ii]:
                peta_all[ii] = petc_all[ii]
            elif Prec[ii] + W >= Sa*D[ii]*(1-pc):
                peta_all[ii] = petc_all[ii]
            else:
                kk = (W+Prec[ii]) / (Sa*D[ii]*(1-pc))
                peta_all[ii] = kk * petc_all[ii]

            W = np.min(np.array([W+Prec[ii]-peta_all[ii], Sa*D[ii]]))
            if W<0: W=0

        '''Assess Yield Loss in entire growth cycle'''

        peta_all_sum = np.sum( peta_all )
        petc_all_sum = np.sum( petc_all )

        f0 = 1 - yloss_f_all * ( 1 - (peta_all_sum/petc_all_sum) )

        '''Assess Yield Loss in individual growth stages separately'''

        peta_d1 = np.sum( peta_stage[0:d_days[0]] )
        peta_d2 = np.sum( peta_stage[d_days[0]:d_days[1]] )
        peta_d3 = np.sum( peta_stage[d_days[1]:d_days[2]] )
        peta_d4 = np.sum( peta_stage[d_days[2]:d_days[3]] )

        petc_d1 = np.sum( petc_stage[0:d_days[0]] )
        petc_d2 = np.sum( petc_stage[d_days[0]:d_days[1]] )
        petc_d3 = np.sum( petc_stage[d_days[1]:d_days[2]] )
        petc_d4 = np.sum( petc_stage[d_days[2]:d_days[3]] )

        f1_d1 = 1 - yloss_f[0] * ( 1 - (peta_d1/petc_d1) )
        f1_d2 = 1 - yloss_f[1] * ( 1 - (peta_d2/petc_d2) )
        f1_d3 = 1 - yloss_f[2] * ( 1 - (peta_d3/petc_d3) )
        f1_d4 = 1 - yloss_f[3] * ( 1 - (peta_d4/petc_d4) )

        f1 = np.min(np.array([f1_d1,f1_d2,f1_d3,f1_d4])) # some references use product, some use minimum. here we use minimum as in Thailand report

        '''Use more severe of above two conditions determines final yield'''

        f_final = np.min(np.array([f0,f1]))

        # to avoid, possible error
        if f_final < 0:
            f_final = 0
        if f_final > 1:
            f_final = 1

        y_water_limited = f_final * y_potential

        return y_water_limited

    def calculateMoistureLimitedYield(self):
        return CropWatCalc.calculateMoistureLimitedYieldNumba(self.cycle_begin, self.cycle_end, self.cycle_len, self.peto, self.Prec, self.d_per, self.kc, self.kc_all, self.yloss_f, self.yloss_f_all, self.y_potential, self.D1, self.D2, self.Sa, self.pc)
