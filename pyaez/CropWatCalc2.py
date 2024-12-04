"""
PyAEZ version 2.4 (Dec 2024)
Crop Water Calculation USING FORTRAN ROUTINE
2020: N. Lakmal Deshapriya and Thaileng Thol
2023: Swun Wunna Htet (Dec)
2024: Swun Wunna Htet (Dec)
Reference: http://oar.icrisat.org/198/1/316_2009_GTAE_55_Poten_obt_yield_in_SAT.pdf

Modifications
1. Fixed the issue of division by zero when crop potential evapotranspiration (PETC) for each
    individual growth stage returns zero.
2. Kc factor adjustment based on local climate is implemented based on GAEZ FORTRAN routine.
3. A major revision to water balanace calculation based on GAEZ FORTRAN routine.
"""
import numpy as np
import numba as nb
from pyaez.UtilitiesCalc import averageDailyToMonthly
from pyaez.LGPCalc import rainPeak, psh, EtaCalc

########################################## MAIN FUNCTION ENDS HERE ###################################################################################
# This code is now working properly. Do not use parallel setting as it will slow down. (Earlier case 1.6 seconds to 4.9 seconds execution)
@nb.jit(nopython=True)
def Adjustkc_Factor(orig_kc,d_per, cycle_len, Prec, eto, min_temp, max_temp, h, wind_sp, irr_or_rain):
    """
    Calculation of kc factor adjustment based on local climate. Sub-function of water
    requirement calculation. 
    Reference: Chapter 6 of FAO and Irrigation and Drainage Report 56, 1998.
    Algorithm adapted from GAEZv4 FORTRAN routine. Calculation exclusive to rainfed yield.
    
    Args:
        in_kc (1D NumPy array or list): input kc factors. Format = [kc_init, kc_vegetative, kc_maturity]
        in_stage_per (1D NumPy array or list): input development duration of stages corresponding to kc factors  Format:[initial, vegetative, reproductive, maturity]
        precip (1D NumPy array): precipitation within cycle duration 
        eto (1D NumPy array): reference evapotranspiration within cycle duration 
        min_temp (1D NumPy array): minimum temperature within cycle duration 
        max_temp (1D NumPy array): maximum temperature within cycle duration (1D NumPy array)
        cycle_start (int) : starting DOY of the cycle
        incycle_len (int): input cycle length
        height (float): canopy height (meters)
        wind_sp (1D NumPy array): wind speed in 2 meters
        irr_or_rain (string): Rainfed ('R')/ Irrigated ('I')
    """
    kc = orig_kc
    stage_per = d_per
    incycle_len = cycle_len

    # Kc1 (kc factor for initial growth stage) procedure
    precip_cycle = Prec # correct
    eto_cycle = eto # correct
    minT_cycle = min_temp # correct
    maxT_cycle = max_temp # correct

    # creating cumulative summation of the number of days for each growing stage (Because np.cumsum is not supported in Numba)

    sum = 0
    cumu_arr =  np.zeros(4,dtype= 'int8')

    for i in range(cumu_arr.shape[0]):
        sum += stage_per[i]
        cumu_arr[i] = sum

    d_days = np.round_( incycle_len * (cumu_arr/100) ).astype('int')

    precip_d1 = precip_cycle[:d_days[0]]
    eto_d1 = eto_cycle[:d_days[0]]

    cum_sum = (precip_d1 >= 0.2 )*1

    # (Confirmed by Gunther) For irrigated conditions, if the wet events are greater than 5, GAEZ set up wet events to 5 due to 
    # the fact that irrigated condition is expected constant water supply.
    if irr_or_rain == 'I':
        wet_events = 5
    else:
        # counting the number of wet_events snippet
        wet_events = np.nansum(cum_sum)

            
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
        rew1 = 6./np.sqrt(avg_eto1)

    if rew1 < 2.5:
        rew1 = 2.5
    elif rew1 >7:
        rew1 = 7. 

    # light infiltration depths (> 40mm)
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
    if tw <= t1 or abs(tw * avg_eto1) <= 1e-5:
        kc1 = round(1.15 * fw, 2)
    else:
        exp = np.exp((-(tw - t1) * Eso * (1 + (rew/(tew - rew))))/tew) # correct
        com = (tew - ((tew - rew) * exp))/(tw * avg_eto1)
        kc1 = round(min(1.15, com) * fw, 2)


    # kc1 finished

    #################################
    # plant high factor adjustment (must be included in crop parameterization file)

    # from object class
    hx = h

    # if canopy height is not enough, kc2 and kc3 will not be applied
    # if hx <= 0.1:
    #     return kc1, kc[1], kc[2]
    if hx > 10:
        hx = 10.

    hfct = (hx/3.)**0.3

    # length duration starts from previous DOY to length for D2+ D3

    # precip_d2 = precip_cycle[d_days[0]:d_days[2]-1]
    wd_d2 = wind_sp[d_days[0]:d_days[2]]

    minT_d2 = minT_cycle[d_days[0]:d_days[2]]
    maxT_d2 = maxT_cycle[d_days[0]:d_days[2]]


    e0n = np.exp((17.27 * minT_d2)/(237.3 + minT_d2))
    e0x = np.exp((17.27 * maxT_d2)/(237.3 + maxT_d2))


    div = e0n/ e0x
    div[e0x<=0] = 0

    e0n_sum = np.sum(div)
    wd_sum = np.sum(wd_d2)

    rhmn = 100 * (e0n_sum)/(stage_per[2]-stage_per[0])

    if rhmn < 20.:
        rhmn = 20.
    elif rhmn > 80:
        rhmn = 80.;

    u2 = wd_sum/((stage_per[2]-stage_per[0]))
    if u2 <1.:
        u2 = 1.
    elif u2>6.:
        u2 = 6.;

    kc2 = round(kc[1] + (0.04 * (u2-2.) - 0.004*(rhmn - 45.)) * hfct, 2) # correct
    # print(kc2)
    ### kc2 completed
    """kc3 starting"""

    if kc[2] > 0.45:
        # precip_d3 = precip_cycle[d_days[2]:]
        # eto_d3 = eto_cycle[d_days[2]:]
        wd_d3 = wind_sp[d_days[2]:]

        minT_d3 = minT_cycle[d_days[2]:]
        maxT_d3 = maxT_cycle[d_days[2]:]

        e0n3 = np.exp((17.27 * minT_d3)/(237.3 + minT_d3))
        e0x3 = np.exp((17.27 * maxT_d3)/(237.3 + maxT_d3))

        div3 = e0n3/ e0x3
        div3[e0x3 <=0] = 0

        e0n3_sum = np.sum(div3)
        wd3_sum = np.sum(wd_d3)


        rhmn3 = 100 * (e0n3_sum/e0n3.shape[0])

        if rhmn3 < 20.:
            rhmn3 = 20.
        elif rhmn3 >80.:
            rhmn3 = 80.
        else:
            rhmn = 45.;

        
        u23 = wd3_sum/ (wd_d3.shape[0])

        if u23 <1.:
            u23 = 1.
        elif u23 > 6.:
            u23 = 6.
        else:
            u23 = 2.;

        kc3 = round(kc[2] + (0.04 * (u23 - 2.) - (0.004 * (rhmn3 - 45.))) * hfct, 2)
    else:
        kc3 = kc[2]
        

    return kc1, kc2, kc3

#---------------------------------- END OF CODE----------------------------------------------------------#
@nb.jit(nopython=True)
def eta(wb_old, etm, Sa, D, p, rain):
    """SUBROUTINE: Calculate actual evapotranspiration (ETa) 

    Args:
        wb_old (float): daily water balance left from the previous day
        etm (float): maximum evapotranspiration
        Sa (float): Available soil moisture holding capacity [mm/m]
        D (float): rooting depth [m]
        p (float): soil moisture depletion fraction (0-1)
        rain (float): amount of rainfall


    Returns:
        wb (float): a value for daily water balance
        wx (float): a value for total available soil moisture
        eta (float): the calculated actual evapotranspiration
    """
    s = wb_old+rain
    wx = 0
    Salim = max(Sa*D, 1.)
    wr=min(100*(1-p),Salim)

    if rain >= etm:
        eta = etm
    elif s-wr >= etm:
        eta = etm
    else:
        rho = wb_old/wr
        eta = min(rain + rho*etm, etm)

    s = s - eta
    if s > Salim:
        wx = s - Salim
        s = Salim
    
    if s < 0: s = 0

    wb = s

    return wb, wx, eta

@nb.jit(nopython=True)
def YieldReductionByWaterDeficit(yloss_f, eta_cycle, etm_cycle, cycle_len, d_per):

    """
    This yield reduction due to moisture deficit does not need for irrigated annual crops and irrigated perennials.
    Exclusively for rainfed annual crops. Calculates yield reduction factor for each growth stage.
    """

    '''Convert Percentage of stage in the crop cycle to cumulative number of days'''
    sum = 0
    cumu_arr =  np.zeros(4,dtype= 'int8')

    for i in range(cumu_arr.shape[0]):
        sum += d_per[i]
        cumu_arr[i] = sum

    d_days = np.round(cycle_len * (np.cumsum(cumu_arr)/100) ).astype('int')

    eta_d1 = np.sum( eta_cycle[0:d_days[0]] )
    eta_d2 = np.sum( eta_cycle[d_days[0]:d_days[1]] )
    eta_d3 = np.sum( eta_cycle[d_days[1]:d_days[2]] )
    eta_d4 = np.sum( eta_cycle[d_days[2]:d_days[3]] )

    etm_d1 = np.sum( etm_cycle[0:d_days[0]] )
    etm_d2 = np.sum( etm_cycle[d_days[0]:d_days[1]] )
    etm_d3 = np.sum( etm_cycle[d_days[1]:d_days[2]] )
    etm_d4 = np.sum( etm_cycle[d_days[2]:d_days[3]] )

    # 1. Modification (SWH)
    # Avoiding division by zero
    f1_d1 = 1 - yloss_f[0] if etm_d1==0 else 1 - (yloss_f[0] * (1 - (eta_d1/etm_d1)));
    f1_d2 = 1 - yloss_f[1] if etm_d2==0 else 1 - (yloss_f[1] * (1 - (eta_d2/etm_d2)));
    f1_d3 = 1 - yloss_f[2] if etm_d3==0 else 1 - (yloss_f[2] * (1 - (eta_d3/etm_d3)));
    f1_d4 = 1 - yloss_f[3] if etm_d4==0 else 1 - (yloss_f[3] * (1 - (eta_d4/etm_d4)));

    f1 = min(f1_d1,f1_d2,f1_d3,f1_d4) # some references use product, some use minimum. here we use minimum as in Thailand report

    return f1


@nb.jit(nopython=True)
def WaterBalance(eto, kc, d_per, cycle_len, Sa, D1, D2, mean_temp, max_temp, Prec, crop_group):

    """Sub-function routine: apply water balance calculation for individual growth stage of development
    
    Args:
        eto (1-D NumPy array): cycle-length specific reference evapotranspiration (mm/day)
        kc (1-D NumPy array): crop water requirement factors for individual growth stages
        d_per (1-D NumPy array): percentages of days each individual growth stages compared to cycle length
        cycle_len (int): cycle length of the crop cycle [Days]
        Sa (int/float): soil water holding capacity (mm/m)
        D1 (int/float): effective soil depth at the cycle start (m)
        D2 (int/float): effective soil depth at the cycle end (m)
        mean_temp (1-D NumPy array): cycle-length specific mean temperature (Deg C)
        max_temp (1-D NumPy array): cycle-length specific maximum temperature (Deg C)
        Prec (1-D NumPy array): cycle-length specific precipitation (mm/day)
        crop_group (int): soil water depletion factor group

    Return:
        Sb_cycle (1-D NumPy Array): cycle-length specific daily snow balance 
        Wx_cycle (1-D NumPy Array): cycle-length specific daily available soil moisture
        Wb_cycle (1-D NumPy Array): cycle-length specific daily water balance amount (mm)
        Eta_cycle (1-D NumPy Array): cycle-length specific daily actual evapotranspiration (mm/day)
        Etm_cycle (1-D NumPy Array): cycle-length specific daily maximum evapotranpiration (mm/day)
    """
    # Creating constant values for Snow balance calculation
    Txsnm = 0.0 # snow melt temperature threshold
    Fsnm = 5.5 # snow melting coefficient

    #  creating cumulative summation of the number of days for each growing stage (Because np.cumsum is not supported in Numba)

    sum = 0
    cumu_arr =  np.zeros(4,dtype= 'int8')

    for i in range(cumu_arr.shape[0]):
        sum += d_per[i]
        cumu_arr[i] = sum
    
    d_days = np.round_( cycle_len * (cumu_arr/100) ).astype('int')

    '''Interpolate Effective Soil Depth / Rootable Depth (D) if D1 and D2 are not equal.'''
    D = np.zeros(eto.size)

    if D1 == D2:
        D = np.full(D.shape, D1)
    else:
        D[d_days[1]:] = D2
        D[:d_days[1]] = D1 + ((D2-D1)/(d_days[1])) * np.arange(d_days[1])
    
    # curtail D by setting limits on usable soil water
    D[D<0.1]= 0.1
    D[D>1.0] = 1.0

    # create output arrays
    Wx_cycle = np.zeros(eto.size)
    Wb_cycle = np.zeros(eto.size)
    Etm_cycle = np.zeros(eto.size)
    Sb_cycle = np.zeros(eto.size)
    Eta_cycle = np.zeros(eto.size)
    kc_daily = np.zeros(eto.size)
    pc_daily = np.zeros(eto.size)
    
    # kc factor array creation
    kc_daily[0:d_days[0]] = kc[0]
    kc_daily[d_days[0]:d_days[1]] = kc[0] + (np.arange(d_days[0], d_days[1])- d_days[0]) * ( (kc[1]-kc[0])/(d_days[1]-d_days[0]) )
    kc_daily[d_days[1]:d_days[2]] = kc[1]
    kc_daily[d_days[2]:] = kc[1] + (np.arange(d_days[2], d_days[3])- d_days[2]) * ( (kc[2]-kc[1])/(d_days[3]-d_days[2]) )


    # calculation of soil water depletion factor (pc) for the whole cycle length
    if crop_group == 0.:
        psh0 = 0.5
    else:
        psh0 = 0.3+(crop_group-1)*.05

    pc_daily = psh0 + .04 * (5.-eto)
    pc_daily[pc_daily < 0.1] = 0.1
    pc_daily[pc_daily > 0.8] = 0.8


    # Loop for each doy for water balance calculation (Snow balance calculation considered)
    # starting point variables
    wb = 0.
    sbx = 0.
    sb = 0.
    wx = 0.

    for i in range(eto.shape[0]):
        
        #"""Periods with Tmax <= Txsnm (precipitation falls as snow)"""
        if max_temp[i] <= Txsnm:
            etm = kc_daily[i] * eto[i]
            Etm_cycle[i] = etm
            sbx = sb + Prec[i]

            if sbx >= etm:
                Sb_cycle[i] = sbx - etm
                Eta_cycle[i] = etm
            else:
                Sb_cycle[i] = 0
                val = eta(wb, etm - sbx, Sa, D[i], pc_daily[i], 0.) 
                wb, wx, Eta_cycle[i]  = val[0], val[1], val[2]+ sbx
            Wb_cycle[i] = wb
            sb = Sb_cycle[i]
        
        #"""Periods with Txsnm < Tmax; Tx <=0 (Preciptiation is water; 100% runoff)"""

        elif mean_temp[i] <= 0:
            etm = kc_daily[i] * eto[i]
            Etm_cycle[i] = etm

            # Apply snow-melt function
            a = Fsnm * (max_temp[i] - Txsnm)
            snm = min(a, sb) # snowmelt
            sb = sb - snm
            wb = wb + snm
            sbx = sb

            if sbx >= etm:
                Sb_cycle[i] = sbx - etm
                Eta_cycle[i] = etm

                if wb > Sa:
                    wx = wb - Sa + Prec[i]
                    wb = Sa
                else:
                    wx = Prec[i]
            else:
                Sb_cycle[i] = 0.
                val =  eta(wb, etm, Sa, D[i], pc_daily[i], Prec[i]) 
                wb, wx, Eta_cycle[i] = val[0], val[1], val[2]+ sbx
            
            Wb_cycle[i] = wb
            sb = Sb_cycle[i]
            Wx_cycle[i] = wx
        
        #""""Periods with 0 < Ta < 5""""
        elif mean_temp[i] < 5:
            etm = kc_daily[i] * eto[i]
            Etm_cycle[i] = etm

            # if there is still snow
            if sb > 0:
                a = Fsnm * (max_temp[i] - Txsnm)
                snm = min(a, sb)
            else:
                snm = 0
            
            wb = wb + snm
            sb = sb - snm
            val = eta(wb, etm, Sa, D[i], pc_daily[i], Prec[i])
            wb, wx, Eta_cycle[i] = val[0], val[1], val[2]

            Sb_cycle[i] = sb
            Wx_cycle[i] = wx
            Wb_cycle[i] = wb

        #""""Periods with Ta >5 """"
        elif mean_temp[i] >= 5:
            etm = kc_daily[i] * eto[i]
            Etm_cycle[i] = etm

            # if there is still snow
            if sb > 0:
                a = Fsnm * (max_temp[i] - Txsnm)
                snm = min(a, sb)
            else:
                snm = 0
            wb = wb + snm
            sb = sb - snm
            val = eta(wb, etm, Sa, D[i], pc_daily[i], Prec[i])
            wb, wx, Eta_cycle[i] = val[0], val[1], val[2]
            Sb_cycle[i] = sb
            Wx_cycle[i] = wx
            Wb_cycle[i] = wb
    
    
    return Sb_cycle, Wx_cycle, Wb_cycle, Eta_cycle, Etm_cycle, kc_daily, pc_daily
    # Swun (14/9/2024) the above code will be used for the calculation of ETa and ETm, including the water

@nb.jit(nopython=True)
def WaterBalancewhole(eto, kc_daily, cycle_start, cycle_len, Sa, D1, D2, mean_temp, max_temp, Prec, crop_group):

    """Sub-function routine: apply water balance calculation for individual growth stage of development
    
    Args:
        eto (1-D NumPy array): cycle-length specific reference evapotranspiration (mm/day)
        kc (1-D NumPy array): crop water requirement factors for individual growth stages
        d_per (1-D NumPy array): percentages of days each individual growth stages compared to cycle length
        cycle_len (int): cycle length of the crop cycle [Days]
        Sa (int/float): soil water holding capacity (mm/m)
        D1 (int/float): effective soil depth at the cycle start (m)
        D2 (int/float): effective soil depth at the cycle end (m)
        mean_temp (1-D NumPy array): cycle-length specific mean temperature (Deg C)
        max_temp (1-D NumPy array): cycle-length specific maximum temperature (Deg C)
        Prec (1-D NumPy array): cycle-length specific precipitation (mm/day)
        crop_group (int): soil water depletion factor group

    Return:
        Sb_cycle (1-D NumPy Array): cycle-length specific daily snow balance 
        Wx_cycle (1-D NumPy Array): cycle-length specific daily available soil moisture
        Wb_cycle (1-D NumPy Array): cycle-length specific daily water balance amount (mm)
        Eta_cycle (1-D NumPy Array): cycle-length specific daily actual evapotranspiration (mm/day)
        Etm_cycle (1-D NumPy Array): cycle-length specific daily maximum evapotranpiration (mm/day)
    """
    # Creating constant values for Snow balance calculation
    Txsnm = 0.0 # snow melt temperature threshold
    Fsnm = 5.5 # snow melting coefficient

    # #  creating cumulative summation of the number of days for each growing stage (Because np.cumsum is not supported in Numba)
    
    # d_days = np.round_( cycle_len * (cumu_arr/100) ).astype('int')

    '''Interpolate Effective Soil Depth / Rootable Depth (D) if D1 and D2 are not equal.'''
    D = np.zeros(eto.size)

    if D1 == D2:
        D = np.full(D.shape, D1)
    # else:
    #     D[d_days[1]:] = D2
    #     D[:d_days[1]] = D1 + ((D2-D1)/(d_days[1])) * np.arange(d_days[1])
    
    # curtail D by setting limits on usable soil water
    D[D<0.1]= 0.1
    D[D>1.0] = 1.0

    # create output arrays
    Wx_cycle = np.zeros(eto.size)
    Wb_cycle = np.zeros(eto.size)
    Etm_cycle = np.zeros(eto.size)
    Sb_cycle = np.zeros(eto.size)
    Eta_cycle = np.zeros(eto.size)
    # kc_daily = np.zeros(eto.size)
    pc_daily = np.zeros(eto.size)
    
    # # kc factor array creation
    # kc_daily[0:d_days[0]] = kc[0]
    # kc_daily[d_days[0]:d_days[1]] = kc[0] + (np.arange(d_days[0], d_days[1])- d_days[0]) * ( (kc[1]-kc[0])/(d_days[1]-d_days[0]) )
    # kc_daily[d_days[1]:d_days[2]] = kc[1]
    # kc_daily[d_days[2]:] = kc[1] + (np.arange(d_days[2], d_days[3])- d_days[2]) * ( (kc[2]-kc[1])/(d_days[3]-d_days[2]) )

    # calculation of soil water depletion factor (pc) for the whole cycle length
    if crop_group == 0.:
        psh0 = 0.5
    else:
        psh0 = 0.3+(crop_group-1)*.05

    pc_daily = psh0 + .04 * (5.-eto)
    pc_daily[pc_daily < 0.1] = 0.1
    pc_daily[pc_daily > 0.8] = 0.8


    # Loop for each doy for water balance calculation (Snow balance calculation considered)
    # starting point variables
    wb = 0.
    sbx = 0.
    sb = 0.
    wx = 0.
    snm = None

    for i in range(eto.shape[0]):
        
        #"""Periods with Tmax <= Txsnm (precipitation falls as snow)"""
        if max_temp[i] <= Txsnm:
            etm = kc_daily[i] * eto[i]
            Etm_cycle[i] = etm
            sbx = sb + Prec[i]

            if sbx >= etm:
                Sb_cycle[i] = sbx - etm
                Eta_cycle[i] = etm
            else:
                Sb_cycle[i] = 0
                val = eta(wb, etm - sbx, Sa, D[i], pc_daily[i], 0.) 
                wb, wx, Eta_cycle[i]  = val[0], val[1], val[2]+ sbx
            Wb_cycle[i] = wb
            sb = Sb_cycle[i]
        
        #"""Periods with Txsnm < Tmax; Tx <=0 (Preciptiation is water; 100% runoff)"""

        elif mean_temp[i] <= 0:
            etm = kc_daily[i] * eto[i]
            Etm_cycle[i] = etm

            # Apply snow-melt function
            a = Fsnm * (max_temp[i] - Txsnm)
            snm = min(a, sb) # snowmelt
            sb = sb - snm
            wb = wb + snm
            sbx = sb

            if sbx >= etm:
                Sb_cycle[i] = sbx - etm
                Eta_cycle[i] = etm

                if wb > Sa:
                    wx = wb - Sa + Prec[i]
                    wb = Sa
                else:
                    wx = Prec[i]
            else:
                Sb_cycle[i] = 0.
                val =  eta(wb, etm, Sa, D[i], pc_daily[i], Prec[i]) 
                wb, wx, Eta_cycle[i] = val[0], val[1], val[2]+ sbx
            
            Wb_cycle[i] = wb
            sb = Sb_cycle[i]
            Wx_cycle[i] = wx
        
        #""""Periods with 0 < Ta < 5""""
        elif mean_temp[i] < 5:
            etm = kc_daily[i] * eto[i]
            Etm_cycle[i] = etm

            # if there is still snow
            if sb > 0:
                a = Fsnm * (max_temp[i] - Txsnm)
                snm = min(a, sb)
            else:
                snm = 0
            
            wb = wb + snm
            sb = sb - snm
            val = eta(wb, etm, Sa, D[i], pc_daily[i], Prec[i])
            wb, wx, Eta_cycle[i] = val[0], val[1], val[2]

            Sb_cycle[i] = sb
            Wx_cycle[i] = wx
            Wb_cycle[i] = wb

        #""""Periods with Ta >5 """"
        elif mean_temp[i] >= 5:
            etm = kc_daily[i] * eto[i]
            Etm_cycle[i] = etm

            # if there is still snow
            if sb > 0:
                a = Fsnm * (max_temp[i] - Txsnm)
                snm = min(a, sb)
            else:
                snm = 0
            wb = wb + snm
            sb = sb - snm
            val = eta(wb, etm, Sa, D[i], pc_daily[i], Prec[i])
            wb, wx, Eta_cycle[i] = val[0], val[1], val[2]
            Sb_cycle[i] = sb
            Wx_cycle[i] = wx
            Wb_cycle[i] = wb
    
    return Sb_cycle, Wx_cycle, Wb_cycle, Eta_cycle, Etm_cycle, kc_daily, pc_daily

@nb.jit(nopython=True)
def WaterBalancewholeII(eto, kc_daily, Sa, D1, D2, Prec, crop_group):

    """Sub-function routine: apply water balance calculation for individual growth stage of development
        Note: to test out the water balance based on WATREQ.F routine. But this approach is not correct.
    Args:
        eto (1-D NumPy array): cycle-length specific reference evapotranspiration (mm/day)
        kc (1-D NumPy array): crop water requirement factors for individual growth stages
        d_per (1-D NumPy array): percentages of days each individual growth stages compared to cycle length
        cycle_len (int): cycle length of the crop cycle [Days]
        Sa (int/float): soil water holding capacity (mm/m)
        D1 (int/float): effective soil depth at the cycle start (m)
        D2 (int/float): effective soil depth at the cycle end (m)
        mean_temp (1-D NumPy array): cycle-length specific mean temperature (Deg C)
        max_temp (1-D NumPy array): cycle-length specific maximum temperature (Deg C)
        Prec (1-D NumPy array): cycle-length specific precipitation (mm/day)
        crop_group (int): soil water depletion factor group

    Return:
        Sb_cycle (1-D NumPy Array): cycle-length specific daily snow balance 
        Wx_cycle (1-D NumPy Array): cycle-length specific daily available soil moisture
        Wb_cycle (1-D NumPy Array): cycle-length specific daily water balance amount (mm)
        Eta_cycle (1-D NumPy Array): cycle-length specific daily actual evapotranspiration (mm/day)
        Etm_cycle (1-D NumPy Array): cycle-length specific daily maximum evapotranpiration (mm/day)
    """

    '''Interpolate Effective Soil Depth / Rootable Depth (D) if D1 and D2 are not equal.'''
    D = np.zeros(eto.size)

    if D1 == D2:
        D = np.full(D.shape, D1)
    # else:
    #     D[d_days[1]:] = D2
    #     D[:d_days[1]] = D1 + ((D2-D1)/(d_days[1])) * np.arange(d_days[1])
    
    # curtail D by setting limits on usable soil water
    D[D<0.1]= 0.1
    D[D>1.0] = 1.0

    # create output arrays
    Wx_cycle = np.zeros(eto.size)
    Wb_cycle = np.zeros(eto.size)
    Etm_cycle = np.zeros(eto.size)
    Sb_cycle = np.zeros(eto.size)
    Eta_cycle = np.zeros(eto.size)
    pc_daily = np.zeros(eto.size)
    
    # calculation of soil water depletion factor (pc) for the whole cycle length
    if crop_group == 0.:
        psh0 = 0.5
    else:
        psh0 = 0.3+(crop_group-1)*.05

    pc_daily = psh0 + .04 * (5.-eto)
    pc_daily[pc_daily < 0.1] = 0.1
    pc_daily[pc_daily > 0.8] = 0.8

    # maximum evapotranspiration (ETm calculation)
    Etm_cycle = eto * kc_daily

    # Loop for each doy for water balance and actual evapotranspiration (ETa)
    # starting point variables
    wb = 0.
    wx = 0.

    for i in range(eto.shape[0]):
        dta = eta(wb, Etm_cycle[i], Sa, D[i], pc_daily[i], Prec[i])
        wb, wx, Eta_cycle[i] = dta
        Wx_cycle[i] = wx
        Wb_cycle[i] = wb
        
    
    return Wx_cycle, Wb_cycle, Eta_cycle, Etm_cycle, kc_daily, pc_daily

@nb.jit(nopython = True)
def calculateMoistureLimitedYieldNumba(irr_or_rain, kc, d_per, cycle_len, Prec, eto, min_temp, max_temp, height, wind_sp,
                                        Sa, D1, D2, mean_temp, crop_group, yloss_f_all, yloss_f, perennial_flag, y_potential):

    '''
    #3 Changes: Water balance calculation is now calculated for both rainfed and irrigated condition. The revised logic is that the assessment of yield loss for
        the entire growth stage are not applied to irrigated annual crops and irrigated perennial crops due to assumption of constant water would be suppied throughout 
        the entire growing cycle, thus no possible yield loss could occurr. However, you still need to assess yield loss coming from individual growith stages of development 
        caused by water deficit for each interval must be assessed because the deficit information is important for farmers to know how much net irrigation is needed.
    '''
    
    # create input variables
    wde = 0. # water deficit
    fc2_cycle = 0.
    fc2_all = 0.
    fc2_final = 0.
    eta_total = 0.


    # Kc factor adjustment based on local climate conditions is done for rainfed/irrigated condition

    adj_kc = Adjustkc_Factor(kc, d_per, cycle_len, Prec, eto, min_temp, max_temp, height, wind_sp, irr_or_rain)
    adj_kc = np.array(adj_kc)

    
    Sb_cycle = None
    Wx_cycle = None
    Wb_cycle = None
    eta_stage = None
    etm_stage = None
    
    # Start water balance calculation using crop-stage specific kc factors
    water_balance_results = WaterBalance(eto, adj_kc, d_per, cycle_len, Sa, D1, D2, mean_temp, max_temp, Prec, crop_group)
    Sb_cycle =  water_balance_results[0]
    Wx_cycle = water_balance_results[1]
    Wb_cycle = water_balance_results[2]
    eta_stage =water_balance_results[3]
    etm_stage = water_balance_results[4]
    kc_stage = water_balance_results[5]

    # calculate total cycle deficit and cumulative over crop stages
    sum_eta_stage = np.sum(eta_stage)
    sum_etm_stage = np.sum(etm_stage)

    ratio = 0. if sum_etm_stage <=0 else sum_eta_stage/sum_etm_stage

    wde = sum_etm_stage - sum_eta_stage
    eta_total = sum_eta_stage

    # Yield reduction factor for the overall cycle period
    fc2_all = 1 - (yloss_f_all *(1- ratio) )

    if fc2_all <0: fc2_all = 0. 

    # Any irrigated annuals or perennials won't go yield deficit assessment due to moisture 
    if perennial_flag:
        fc2_cycle = 1.
    else: 
        fc2_cycle = YieldReductionByWaterDeficit(yloss_f, eta_stage, etm_stage, cycle_len, d_per)
    
    if irr_or_rain == 'I':
        fc2_final = 1.
    else:
        fc2_final = min(fc2_cycle, fc2_all)

    yld_w = y_potential * fc2_final

    return wde, fc2_final, eta_total,  yld_w

@nb.jit(nopython = True)
def calculateMoistureLimitedYieldNumbaIntermediates(irr_or_rain, kc, d_per, cycle_len, Prec, eto, min_temp, max_temp, height, wind_sp,
                                        Sa, D1, D2, mean_temp, crop_group, yloss_f_all, yloss_f, perennial_flag, y_potential):

    '''
    #3 Changes: Water balance calculation is now calculated for both rainfed and irrigated condition. The revised logic is that the assessment of yield loss for
        the entire growth stage are not applied to irrigated annual crops and irrigated perennial crops due to assumption of constant water would be suppied throughout 
        the entire growing cycle, thus no possible yield loss could occurr. However, you still need to assess yield loss coming from individual growith stages of development 
        caused by water deficit for each interval must be assessed because the deficit information is important for farmers to know how much net irrigation is needed.
    '''
    
    # create input variables
    wde = 0. # water deficit
    fc2_cycle = 0.
    fc2_all = 0.
    fc2_final = 0.
    eta_total = 0.


    # Kc factor adjustment based on local climate conditions are done to rainfed condition only

    adj_kc = Adjustkc_Factor(kc, d_per, cycle_len, Prec, eto, min_temp, max_temp, height, wind_sp, irr_or_rain)
    adj_kc = np.array(adj_kc)

    
    Sb_cycle = None
    Wx_cycle = None
    Wb_cycle = None
    eta_stage = None
    etm_stage = None
    
    # Start water balance calculation using crop-stage specific kc factors
    water_balance_results = WaterBalance(eto, adj_kc, d_per, cycle_len, Sa, D1, D2, mean_temp, max_temp, Prec, crop_group)
    Sb_cycle =  water_balance_results[0]
    Wx_cycle = water_balance_results[1]
    Wb_cycle = water_balance_results[2]
    eta_stage =water_balance_results[3]
    etm_stage = water_balance_results[4]

    # calculate total cycle deficit and cumulative over crop stages
    sum_eta_stage = np.sum(eta_stage)
    sum_etm_stage = np.sum(etm_stage)

    ratio = 0. if sum_etm_stage <=0 else sum_eta_stage/sum_etm_stage

    wde = sum_etm_stage - sum_eta_stage
    eta_total = sum_eta_stage

    # Yield reduction factor for the overall cycle period
    fc2_all = 1 - (yloss_f_all *(1- ratio) )

    if fc2_all <0: fc2_all = 0. 

    # Any irrigated annuals or perennials won't go yield deficit assessment due to moisture 
    if perennial_flag:
        fc2_cycle = 1.
    else: 
        fc2_cycle = YieldReductionByWaterDeficit(yloss_f, eta_stage, etm_stage, cycle_len, d_per)
    
    if irr_or_rain == 'I':
        fc2_final = 1.
    else:
        fc2_final = min(fc2_cycle, fc2_all)

    yld_w = y_potential * fc2_final

    return wde, fc2_final, eta_total,  yld_w, fc2_cycle, fc2_all, adj_kc, ratio, Sb_cycle, Wx_cycle, Wb_cycle, eta_stage, etm_stage, water_balance_results[5], water_balance_results[6]

@nb.jit(nopython = True)
def calculateMoistureLimitedYieldNumbaIntermediatesII(irr_or_rain, kc, d_per, cycle_len, Prec, eto, min_temp, max_temp, height, wind_sp,
                                        Sa, D1, D2, mean_temp, crop_group, yloss_f_all, yloss_f, perennial_flag, y_potential):

    '''
    #3 Changes: Water balance calculation is now calculated for both rainfed and irrigated condition. The revised logic is that the assessment of yield loss for
        the entire growth stage are not applied to irrigated annual crops and irrigated perennial crops due to assumption of constant water would be suppied throughout 
        the entire growing cycle, thus no possible yield loss could occurr. However, you still need to assess yield loss coming from individual growith stages of development 
        caused by water deficit for each interval must be assessed because the deficit information is important for farmers to know how much net irrigation is needed.
    Note: This functions will be used to check the investigation of water balance. Not for the entire simulation.
    '''
    
    # create input variables
    wde = 0. # water deficit
    fc2_cycle = 0.
    fc2_all = 0.
    fc2_final = 0.
    eta_total = 0.


    # Kc factor adjustment based on local climate conditions are done to rainfed condition only

    adj_kc = Adjustkc_Factor(kc, d_per, cycle_len, Prec, eto, min_temp, max_temp, height, wind_sp, irr_or_rain)
    adj_kc = np.array(adj_kc)

    
    Sb_cycle = None
    Wx_cycle = None
    Wb_cycle = None
    eta_stage = None
    etm_stage = None
    
    # Start water balance calculation using crop-stage specific kc factors
    water_balance_results = WaterBalancewhole(eto, adj_kc, d_per, cycle_len, Sa, D1, D2, mean_temp, max_temp, Prec, crop_group)
    Sb_cycle =  water_balance_results[0]
    Wx_cycle = water_balance_results[1]
    Wb_cycle = water_balance_results[2]
    eta_stage =water_balance_results[3]
    etm_stage = water_balance_results[4]

    # calculate total cycle deficit and cumulative over crop stages
    sum_eta_stage = np.sum(eta_stage)
    sum_etm_stage = np.sum(etm_stage)

    ratio = 0. if sum_etm_stage <=0 else sum_eta_stage/sum_etm_stage

    wde = sum_etm_stage - sum_eta_stage
    eta_total = sum_eta_stage

    # Yield reduction factor for the overall cycle period
    fc2_all = 1 - (yloss_f_all *(1- ratio) )

    if fc2_all <0: fc2_all = 0. 

    # Any irrigated annuals or perennials won't go yield deficit assessment due to moisture 
    if perennial_flag:
        fc2_cycle = 1.
    else: 
        fc2_cycle = YieldReductionByWaterDeficit(yloss_f, eta_stage, etm_stage, cycle_len, d_per)
    
    if irr_or_rain == 'I':
        fc2_final = 1.
    else:
        fc2_final = min(fc2_cycle, fc2_all)

    yld_w = y_potential * fc2_final

    return wde, fc2_final, eta_total,  yld_w, fc2_cycle, fc2_all, adj_kc, ratio, Sb_cycle, Wx_cycle, Wb_cycle, eta_stage, etm_stage, water_balance_results[5], water_balance_results[6]


nb.jit(nopython = True)
def ReferenceWaterBalanceCalc(max_temp, mean_temp, precip, eto, Sa:float = 100., D:float = 1.):
    """Reference water balance calculation for a single pixel (Relevant to Module I).
    
    Args:
        max_temp (1D NumPy Array): daily maximum temperature [Deg Celsius]
        mean_temp (1D NumPy Array): daily mean temperature [Deg Celsius]
        precip (1D NumPy Array): daily precipitation [mm/day]
        eto (1D NumPy Array): daily reference evapotranspiration [mm/day]
        Sa (int/float): Soil Water Holding Capacity [mm/m]
        D (int/float): Rooting Depth [m]
    Return:
        
    """
    #============================
    # Calculation of REFERENCE Water Balance to estimate ETa, ETm
    #============================
    kc_list = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
    #============================
    Txsnm = 0.  # Txsnm - snow melt temperature threshold
    Fsnm = 5.5  # Fsnm - snow melting coefficient
    Sb_old = 0.
    Wb_old = 0.

    #============================
    Tx365 = max_temp.copy()
    Ta365 = mean_temp.copy()
    Pcp365 = precip.copy()
    Eto365 = eto.copy()  # Eto
    Etm365 = np.zeros(Tx365.shape)
    Eta365 = np.zeros(Tx365.shape)
    Sb365 = np.zeros(Tx365.shape)
    Wb365 = np.zeros(Tx365.shape)
    Wx365 = np.zeros(Tx365.shape)
    kc365 = np.zeros(Tx365.shape)

    lgpt5_point = np.sum(mean_temp >=5)
    totalPrec_monthly = averageDailyToMonthly(Pcp365)
    istart0, istart1 = rainPeak(totalPrec_monthly, Ta365, lgpt5_point)

    for doy in range(0, 365):
        p = psh(0., Eto365[doy])
        Eta_new, Etm_new, Wb_new, Wx_new, Sb_new, kc_new = EtaCalc(
                        np.float64(Tx365[doy]), np.float64(
                            Ta365[doy]),
                        np.float64(Pcp365[doy]), Txsnm, Fsnm, np.float64(
                            Eto365[doy]),
                        Wb_old, Sb_old, doy, istart0, istart1,
                        Sa, D, p, kc_list, lgpt5_point)
        
        if Eta_new <0.: Eta_new = 0.
        Eta365[doy] = Eta_new
        Etm365[doy] = Etm_new
        Wb365[doy] = Wb_new
        Wx365[doy] = Wx_new
        Sb365[doy] = Sb_new
        kc365[doy] = kc_new

        Wb_old = Wb_new
        Sb_old = Sb_new

    return Eta365, Etm365, Wb365, Wx365, Sb365, kc365
