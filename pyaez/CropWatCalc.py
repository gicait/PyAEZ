"""
PyAEZ version 2.2 (Dec 2023)
Crop Water Calculation
2020: N. Lakmal Deshapriya and Thaileng Thol
2023: Swun Wunna Htet
Reference: http://oar.icrisat.org/198/1/316_2009_GTAE_55_Poten_obt_yield_in_SAT.pdf

Modifications
1. Fixed the issue of division by zero when crop potential evapotranspiration (PETC) for each
    individual growth stage returns zero.
2. Kc factor adjustment based on local climate is implemented based on GAEZ FORTRAN routine.
"""

import numpy as np
import numba as nb

class CropWatCalc(object):

    def __init__(self, cycle_begin, cycle_end, perennial_flag = False):
        self.cycle_begin = cycle_begin
        self.cycle_end = cycle_end
        self.cycle_len = cycle_end - cycle_begin +1
        self.perennial_flag = perennial_flag

    def setClimateData(self, pet, precipitation, wind_speed, min_temp, max_temp):
        self.peto = pet
        self.Prec = precipitation
        self.wind_sp = wind_speed
        self.min_temp = min_temp
        self.max_temp = max_temp

    def setCropParameters(self, stage_per, kc, kc_all, yloss_f, yloss_f_all, est_yield, D1, D2, Sa, pc, height):
        self.d_per = np.array(stage_per) # Percentage for D1, D2, D3, D4 stages
        self.height = height # canopy height of the crop(m)

        input_kc = np.array(kc) # 3 crop water requirements for initial, reproductive, the end of the maturation stages
        self.kc = self.Adjustkc_Factor(input_kc) # kc factor will be adjusted based on local climate conditions

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

        for ii in range(self.cycle_len-1):

            # print('i = ', ii)

            ## each growing stage
            if ii+1 <= d_days[0]:
                petc_stage[ii] = self.kc[0] * self.peto[ii]
            elif ii+1 <= d_days[1]:
                kc_temp = self.kc[0] + (ii+1-d_days[0]) * ( (self.kc[1]-self.kc[0])/(d_days[1]-d_days[0]) )
                petc_stage[ii] = kc_temp * self.peto[ii]
            elif ii+1 <= d_days[2]:
                petc_stage[ii] = self.kc[1] * self.peto[ii]
            else:
                kc_temp = self.kc[1] + (ii+1-d_days[2]) * ( (self.kc[2]-self.kc[1])/(d_days[3]-d_days[2]) )
                petc_stage[ii] = kc_temp * self.peto[ii]

            ## entire gorwing season
            petc_all[ii] = self.kc_all * self.peto[ii]


        '''Lets calculate Actual ET for each stage'''

        peta_stage = np.zeros(self.peto.size)
        peta_all = np.zeros(self.peto.size)

        ## each growing stage
        W = self.Sa*D[0] 
        for ii in range(self.cycle_len-1):

            if self.Prec[ii] >= petc_stage[ii]:
                peta_stage[ii] = petc_stage[ii]
            elif self.Prec[ii] + W >= self.Sa*D[ii]*(1-self.pc):
                peta_stage[ii] = petc_stage[ii]
            else:
                kk = (W+self.Prec[ii]) / (self.Sa*D[ii]*(1-self.pc))
                peta_stage[ii] = kk * petc_stage[ii]

            W = np.min([W+self.Prec[ii]-peta_stage[ii], self.Sa*D[ii]])
            if W<0: W=0

        ## entire growing season
        W = self.Sa*D[0] 
        for ii in range(self.cycle_len-1):

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
        if self.perennial_flag:
            f1 = 1.
        else:

            peta_d1 = np.sum( peta_stage[0:d_days[0]] )
            peta_d2 = np.sum( peta_stage[d_days[0]:d_days[1]] )
            peta_d3 = np.sum( peta_stage[d_days[1]:d_days[2]] )
            peta_d4 = np.sum( peta_stage[d_days[2]:d_days[3]] )

            petc_d1 = np.sum( petc_stage[0:d_days[0]] )
            petc_d2 = np.sum( petc_stage[d_days[0]:d_days[1]] )
            petc_d3 = np.sum( petc_stage[d_days[1]:d_days[2]] )
            petc_d4 = np.sum( petc_stage[d_days[2]:d_days[3]] )

            # 1. Modification (SWH)
            # Avoiding division by zero
            f1_d1 = 1 - self.yloss_f[0] if petc_d1==0 else 1 - self.yloss_f[0] * (1 - (peta_d1/petc_d1));
            f1_d2 = 1 - self.yloss_f[1] if petc_d2==0 else 1 - self.yloss_f[1] * (1 - (peta_d2/petc_d2));
            f1_d3 = 1 - self.yloss_f[2] if petc_d3==0 else 1 - self.yloss_f[2] * (1 - (peta_d3/petc_d3));
            f1_d4 = 1 - self.yloss_f[3] if petc_d4==0 else 1 - self.yloss_f[3] * (1 - (peta_d4/petc_d4));

            f1 = np.min([f1_d1,f1_d2,f1_d3,f1_d4]) # some references use product, some use minimum. here we use minimum as in Thailand report

        '''Use more severe of above two conditions determines final yield'''

        self.f_final = np.min([f0,f1])

        # to avoid, possible error
        if self.f_final < 0:
            self.f_final = 0
        if self.f_final > 1:
            self.f_final = 1

        y_water_limited = self.f_final * self.y_potential

        return y_water_limited

    def getfc2factormap(self):
        return self.f_final
    
    @staticmethod
    @nb.jit(nopython=True)
    def calculateMoistureLimitedYieldNumba(length, d_per, peto, D2, D1, kc, kc_all, yloss_f, yloss_f_all, Sa, pc, Prec, y_potential, perennial_flag):

        '''Convert Percentage of stage in the crop cycle to cumulative number of days'''

        ddays = np.round_( length * (d_per.cumsum()/100) ).astype('int')

        '''Interpolate Rotting Depth (D)'''
        D = np.zeros(peto.size)
        D[ddays[1]:] = D2
        D[:ddays[1]] = D1 + ((D2-D1)/(ddays[1])) * np.arange(ddays[1])

        '''Lets calculate crop specific PET for each stage'''

        petc_stage = np.zeros(peto.size)
        petc_all = np.zeros(peto.size)

        for ii in range(length):

            ## each growing stage
            if ii <= ddays[0]:
                petc_stage[ii] = kc[0] * peto[ii]
            elif ii <= ddays[1]:
                kc_temp = kc[0] + (ii-ddays[0]) * ( (kc[1]-kc[0])/(ddays[1]-ddays[0]) )
                petc_stage[ii] = kc_temp * peto[ii]
            elif ii <= ddays[2]:
                petc_stage[ii] = kc[1] * peto[ii]
            else:
                kc_temp = kc[1] + (ii-ddays[2]) * ( (kc[2]-kc[1])/(ddays[3]-ddays[2]) )
                petc_stage[ii] = kc_temp * peto[ii]

            ## entire gorwing season
            petc_all[ii] = kc_all * peto[ii]


        '''Lets calculate Actual ET for each stage'''

        peta_stage = np.zeros(peto.size)
        peta_all = np.zeros(peto.size)

        ## each growing stage
        W = Sa*D[0] 
        for ii in range(length):

            if Prec[ii] >= petc_stage[ii]:
                peta_stage[ii] = petc_stage[ii]
            elif Prec[ii] + W >= Sa*D[ii]*(1-pc):
                peta_stage[ii] = petc_stage[ii]
            else:
                kk = (W+Prec[ii]) / (Sa*D[ii]*(1-pc))
                peta_stage[ii] = kk * petc_stage[ii]

            A1 = W+Prec[ii]-peta_stage[ii]
            B1 = Sa*D[ii]
            W = min(A1, B1)

            if W<0: W=0

        ## entire gorwing season
        W = Sa*D[0] 
        for ii in range(length):

            if Prec[ii] >= petc_all[ii]:
                peta_all[ii] = petc_all[ii]
            elif Prec[ii] + W >= Sa*D[ii]*(1-pc):
                peta_all[ii] = petc_all[ii]
            else:
                kk = (W+Prec[ii]) / (Sa*D[ii]*(1-pc))
                peta_all[ii] = kk * petc_all[ii]

            A = W+Prec[ii]-peta_all[ii]
            B = Sa*D[ii]
            W = min(A, B)

            if W<0: W=0

        '''Assess Yield Loss in entire growth cycle'''

        peta_all_sum = np.sum( peta_all )
        petc_all_sum = np.sum( petc_all )

        f0 = 1 - yloss_f_all * ( 1 - (peta_all_sum/petc_all_sum) )

        '''Assess Yield Loss in individual growth stages separately'''
        if perennial_flag:
            f1 = 1.
        else:

            peta_d1 = np.sum( peta_stage[0:ddays[0]] )
            peta_d2 = np.sum( peta_stage[ddays[0]:ddays[1]] )
            peta_d3 = np.sum( peta_stage[ddays[1]:ddays[2]] )
            peta_d4 = np.sum( peta_stage[ddays[2]:ddays[3]] )

            petc_d1 = np.sum( petc_stage[0:ddays[0]] )
            petc_d2 = np.sum( petc_stage[ddays[0]:ddays[1]] )
            petc_d3 = np.sum( petc_stage[ddays[1]:ddays[2]] )
            petc_d4 = np.sum( petc_stage[ddays[2]:ddays[3]] )

            # 1. Modification (SWH)
            # Avoiding division by zero
            f1_d1 = 1 - yloss_f[0] if petc_d1==0 else 1 - yloss_f[0] * (1 - (peta_d1/petc_d1))
            f1_d2 = 1 - yloss_f[1] if petc_d2==0 else 1 - yloss_f[1] * (1 - (peta_d2/petc_d2))
            f1_d3 = 1 - yloss_f[2] if petc_d3==0 else 1 - yloss_f[2] * (1 - (peta_d3/petc_d3))
            f1_d4 = 1 - yloss_f[3] if petc_d4==0 else 1 - yloss_f[3] * (1 - (peta_d4/petc_d4))

            f1 = min(f1_d1,f1_d2,f1_d3,f1_d4) # some references use product, some use minimum. here we use minimum as in Thailand report

        '''Use more severe of above two conditions determines final yield'''

        f_final = min(f0,f1)

        # to avoid, possible error
        if f_final < 0:
            f_final = 0
        if f_final > 1:
            f_final = 1

        y_water_limited = f_final * y_potential

        return [y_water_limited, f_final]
    
    def getMoistureYieldNumba(self):
        result = self.calculateMoistureLimitedYieldNumba(self.cycle_len, self.d_per, self.peto, self.D2, self.D1, self.kc, self.kc_all, self.yloss_f, 
                                                         self.yloss_f_all, self.Sa, self.pc, self.Prec, self.y_potential, self.perennial_flag)
        return result
    ########################################## MAIN FUNCTION ENDS HERE ###################################################################################

    def Adjustkc_Factor(self, orig_kc):
        """
        Calculation of kc factor adjustment based on local climate. Sub-function of water
        requirement calculation. 
        Reference: Chapter 6 of FAO and Irrigation and Drainage Report 56, 1998.
        Algorithm adapted from GAEZv4 FORTRAN routine. Calculation exclusive to rainfed yield.
        
        Required arguments:
            1. in_kc : input kc factors (NumPy array or list). Format = [kc_init, kc_vegetative, kc_maturity]
            2. in_stage_per : input development duration of stages corresponding to kc factors (1D NumPy array or list).
                              Format:[initial, vegetative, reproductive, maturity]
            3. precip : precipitation within cycle duration (1D NumPy array)
            4. eto : reference evapotranspiration within cycle duration (1D NumPy array)
            5. min_temp : minimum temperature within cycle duration (1D NumPy array)
            6. max_temp : maximum temperature within cycle duration (1D NumPy array)
            7. cycle_start : starting DOY of the cycle (int)
            8. incycle_len : input cycle length (int)
            9. height : canopy height (meters)
            10. wind_sp : wind speed in 2 meters (1D NumPy array)
            """
        kc = orig_kc.copy()
        stage_per = self.d_per.copy()
        start, incycle_len = self.cycle_begin, self.cycle_len

        # Kc1 (kc factor for initial growth stage) procedure
        precip_cycle = self.Prec[start-1:start-1+incycle_len-1] # correct
        eto_cycle = self.peto[start-1: start-1+incycle_len-1] # correct
        minT_cycle = self.min_temp[start-1:start-1+incycle_len-1] # correct
        maxT_cycle = self.max_temp[start-1:start-1+incycle_len-1] # correct

        d_days = np.round( incycle_len * (np.cumsum(stage_per)/100) ).astype('int')

        precip_d1 = precip_cycle[:d_days[0]-1]
        eto_d1 = eto_cycle[:d_days[0]-1]

        cum_sum = (precip_d1 >= 0.2 )*1

        # counting the number of wet_events snippet
        wet_events = 0
        flg = 0

        for i in range(cum_sum.shape[0]):
            if cum_sum[i] == False:
                flg = 0
            elif flg == 1:
                continue
            else:
                wet_events = wet_events +1
                flg = 1;

        # average time between wetting events
        tw = d_days[0]/ (wet_events + 0.5) # correct

        # fraction of surfaced wetted by rain (By reference, the value is set to 1 for rainfed)
        fw = 1

        # average wetting depths (Pmean by FAO documentation)
        if wet_events >0:
            dw = (np.sum(precip_d1)/wet_events)/fw # correct
        else:
            dw = 0

        # light infiltration depths (<10mm) (procedure is correct)
        tew1 = 10
        avg_eto1= np.sum(eto_d1)/d_days[0]

        if avg_eto1 <= 1e-5:
            rew1 = 7
        else:
            rew1 = min(max(2.5, 6/np.sqrt(avg_eto1)), 7) # correct

        if rew1 > 7.:
            rew1= 7.

        # larprege infiltration depths (> 40mm)
        tew2 = min(15., 7.*np.sqrt(avg_eto1))
        rew2 = min(6., tew2 - 0.01)

        if dw <= 10.:
            rew = rew1
            tew = tew1
        elif dw >= 40.:
            rew = rew2
            tew = tew2
        else:
            rew = rew1 + ((rew2 - rew1) * (dw - 10.)/ 30.)
            tew = tew1 + ((tew2 - tew1) * (dw - 10.)/ 30.)

        rew = min(rew, tew - 0.01)

        # time needed for stage 1 drying
        # Eso = potential rate of evapotranspiration (mm/day)
        if avg_eto1 <= 1e-5:
            Eso = 0.
            t1 = 365.
        else:
            # coefficient of 1.15 indicates increased evaporation potential due to low albedo of 
            # wet soil and possibility of heat stored in the surface layer during previous dry periods.
            Eso = 1.15 * avg_eto1
            t1 = rew / Eso

        # redefining kc factors
        if tw <= t1 or abs(tw * avg_eto1) <= 0.:
            kc1 = np.round(1.15 * fw, 2)
        else:
            exp = np.exp((-(tw - t1) * Eso * (1 + (rew/(tew - rew))))/tew) # correct
            com = (tew - ((tew - rew) * exp))/(tw * avg_eto1)
            kc1 = np.round(min(1.15, com) * fw, 2)


        # kc1 finished

        #################################
        # plant high factor adjustment (must be included in crop parameterization file)

        # from object class
        hx = self.height

        # for experiment
        hx = 2.0

        # if canopy height is not enough, kc2 and kc3 will not be applied
        if hx <= 0.1:
            return np.array([kc1, kc[1], kc[2]])
        if hx > 10:
            hx = 10.

        hfct = (hx/3.)**0.3

        # length duration starts from previous DOY to length for D2+ D3
        d_days
        precip_d2 = precip_cycle[d_days[0]:d_days[2]-1]
        eto_d2 = eto_cycle[d_days[0]:d_days[2]-1]
        wd_d2 = self.wind_sp[d_days[0]:d_days[2]-1]

        minT_d2 = minT_cycle[d_days[0]:d_days[2]-1]
        maxT_d2 = maxT_cycle[d_days[0]:d_days[2]-1]

        e0n = np.exp((17.27 * minT_d2)/(237.3 + minT_d2))
        e0x = np.exp((17.27 * maxT_d2)/(237.3 + maxT_d2))

        div = np.divide(e0n, e0x, where = e0x!=0, out = np.zeros(e0n.shape[0]))
        e0n_sum = np.sum(div)
        wd_sum = np.sum(wd_d2)

        rhmn = 100 * (e0n_sum)/(stage_per[1]+stage_per[2])

        if rhmn < 20.:
            rhmn = 20.
        elif rhmn > 80:
            rhmn = 80.;

        u2 = wd_sum/((stage_per[1] + stage_per[2])* 86.4)
        if u2 <1.:
            u2 = 1.
        elif u2>6.:
            u2 = 6.;

        kc2 = np.round(kc[1] + (0.04 * (u2-2.) - 0.004*(rhmn - 45.)) * hfct, 2) # correct
        # print(kc2)
        ### kc2 completed
        """kc3 starting"""

        if kc[2] > 0.45:
            precip_d3 = precip_cycle[d_days[2]:]
            eto_d3 = eto_cycle[d_days[2]:]
            wd_d3 = self.wind_sp[d_days[2]:]

            minT_d3 = minT_cycle[d_days[2]:]
            maxT_d3 = maxT_cycle[d_days[2]:]

            e0n3 = np.exp((17.27 * minT_d3)/(237.3 + minT_d3))
            e0x3 = np.exp((17.27 * maxT_d3)/(237.3 + maxT_d3))

            div3 = np.divide(e0n3, e0x3, where = e0x3!=0, out = np.zeros(e0n3.shape[0]))
            e0n3_sum = np.sum(div3)
            wd3_sum = np.sum(wd_d3)


            rhmn3 = 100 * (e0n3_sum/e0n3.shape[0])

            if rhmn3 < 20.:
                rhmn3 = 20.
            elif rhmn3 >80.:
                rhmn3 = 80.
            else:
                rhmn = 45.;

            
            u23 = wd3_sum/ (wd_d3.shape[0] * 86.4)

            if u23 <1.:
                u23 = 1.
            elif u23 > 6.:
                u23 = 6.
            else:
                u23 = 2.;

            kc3 = np.round(kc[2] + (0.04 * (u23 - 2.) - (0.004 * (rhmn3 - 45.))) * hfct, 2)
        else:
            kc3 = kc[2]
            

        return np.array([kc1, kc2, kc3])
    
#---------------------------------- END OF CODE----------------------------------------------------------#
