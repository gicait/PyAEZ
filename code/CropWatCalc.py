"""
PyAEZ
Written by N. Lakmal Deshapriya and Thaileng Thol
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

        petc = np.zeros(self.peto.size)

        for ii in range(self.cycle_len):
            if ii <= d_days[0]:
                petc[ii] = self.kc[0] * self.peto[ii]
            elif ii <= d_days[1]:
                kc_temp = self.kc[0] + (ii-d_days[0]) * ( (self.kc[1]-self.kc[0])/(d_days[1]-d_days[0]) )
                petc[ii] = kc_temp * self.peto[ii]
            elif ii <= d_days[2]:
                petc[ii] = self.kc[1] * self.peto[ii]
            else:
                kc_temp = self.kc[1] + (ii-d_days[2]) * ( (self.kc[2]-self.kc[1])/(d_days[3]-d_days[2]) )
                petc[ii] = kc_temp * self.peto[ii]

        '''Lets calculate Actual ET for each stage'''

        peta = np.zeros(self.peto.size)

        W = 0

        for ii in range(self.cycle_len):

            if self.Prec[ii] >= petc[ii]:
                peta[ii] = petc[ii]
            elif self.Prec[ii] + W >= self.Sa*D[ii]*(1-self.pc):
                peta[ii] = petc[ii]
            else:
                kk = (W+self.Prec[ii]) / (self.Sa*D[ii]*(1-self.pc))
                peta[ii] = kk * petc[ii]

            W = np.min([W+self.Prec[ii]-peta[ii], self.Sa*D[ii]])
            if W<0: W=0

        '''Assess Yield Loss in entire growth cycle'''

        peta_all = np.sum( peta )

        petc_all = np.sum( petc )

        f0 = 1 - self.yloss_f_all * ( 1 - (peta_all/petc_all) )

        '''Assess Yield Loss in individual growth stages separately'''

        peta_d1 = np.sum( peta[0:d_days[0]] )
        peta_d2 = np.sum( peta[d_days[0]:d_days[1]] )
        peta_d3 = np.sum( peta[d_days[1]:d_days[2]] )
        peta_d4 = np.sum( peta[d_days[2]:d_days[3]] )

        petc_d1 = np.sum( petc[0:d_days[0]] )
        petc_d2 = np.sum( petc[d_days[0]:d_days[1]] )
        petc_d3 = np.sum( petc[d_days[1]:d_days[2]] )
        petc_d4 = np.sum( petc[d_days[2]:d_days[3]] )

        f1_d1 = 1 - self.yloss_f[0] * ( 1 - (peta_d1/petc_d1) )
        f1_d2 = 1 - self.yloss_f[1] * ( 1 - (peta_d2/petc_d2) )
        f1_d3 = 1 - self.yloss_f[2] * ( 1 - (peta_d3/petc_d3) )
        f1_d4 = 1 - self.yloss_f[3] * ( 1 - (peta_d4/petc_d4) )

        f1 = f1_d1*f1_d2*f1_d3*f1_d4

        '''Use more severe of above two conditions determines final yield'''

        y_water_limited = np.min([f0,f1]) * self.y_potential

        return y_water_limited
