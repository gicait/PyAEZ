"""
PyAEZ version 2.2 (Dec 2023)
PyAEZ: LGPCalc.py calculates the length of growing period (LGP)
2022/2023: Kittiphon Boonma
2025     : Swun Wunna Htet

Modifications
-------------

1. Revised 10-day moving average to return 365 days of input variable instead of 356.
"""

from numba import jit
import numpy as np


@jit(nopython=True)
def rainPeak(meanT_daily, lgpt5_point):
    """Scan the monthly precipitation for the month with peak rainfall over 3 months

    Args:
        meanT_daily (float): daily mean temperature for a pixel
        lgpt5_point (float): thermal length of growing period (T>5C) for a pixel

    Returns:
        meanT_daily(1D NumPy): the smoothened daily mean temperature curve
        istart0(int): the starting date of the growing period
        istart1(int): the ending date of the growing period
    """
    # FORTRAN routine evaluates the rain peak but this is not used in start and end date determination
    # of LGPT start and end
    # The rest of the code should work exact like python statements.
    # ============================================
    days_f = np.arange(0, 365)
    lgpt5_veg = days_f[meanT_daily >= 5]
    # ============================================
    if lgpt5_point < 365:
        istart0 = lgpt5_veg[0]
        istart1 = setdat(istart0) + lgpt5_point-1
    else:
        istart0 = 0
        istart1 = lgpt5_point-1

    return istart0, istart1
# ============================================


@jit(nopython=True)
def isfromt0(meanT_daily_new, doy):
    """Check if the Julian day is coming from the temperature
       upward or downward trend

    Args:
        meanT_daily_new (1D NumPy): 1-year time-series of daily mean temperature
        doy (int): Julian day

    Returns:
        _type_: _description_
    """
    if meanT_daily_new[doy]-meanT_daily_new[doy-1] > 0.:
        fromt0 = 1.
    else:
        fromt0 = 0.

    return fromt0

# ============================================


@jit(nopython=True)
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
        float: a value for daily water balance
        float: a value for total available soil moisture
        float: the calculated actual evapotranspiration
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
        eta = min(rain + (rho*etm), etm)

    wb=s-eta

    if wb > Salim:
        wx = wb-Salim
        wb = Salim
    
    if wb < 0:
        wb=0

    return wb, wx, eta


@jit(nopython=True)
def psh(ng, et0):
    """Calculate soil moisture depletion fraction (0-1)

    Args:
        ng (float): crop group
        et0 (float): potential evapotranspiration [mm/day]

    Returns:
        float: soil moisture depletion fraction
    """
    # ng = crop group
    # eto = potential evapotranspiration [mm/day]
    if ng == 0:
        psh0 = 0.5
    else:
        psh0 = 0.3+(ng-1)*.05

    psh = psh0 + .04 * (5.-et0)

    if psh < 0.1:
        psh = 0.1
    elif psh > 0.8:
        psh = 0.8

    return psh


@jit(nopython=True)
def val10day(Et):
    """Calculate 10-day moving average 

    Args:
        Et (1D NumPy Array): 1D np array (365 days) of any variables
    Return:
        10-day average(1D NumPy Array): 
    """
    # Program to calculate moving average
    arr = np.concatenate((Et[354:364], Et, Et[:10]))
    window_size = 10
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < 365:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i+10: i+10 + window_size]
        # Calculate the average of current window
        window_average = round((sum(window) / window_size), 2)
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        # Shift window to right by one position
        i += 1

    return np.array(moving_averages)


@jit(nopython=True)
def RefWaterBalanceCalc(Tx365, Ta365, Pcp365, Txsnm, Fsnm, Eto365, wb_old, sb_old, doy, istart0, istart1, Sa, D, p, lgpt5_point, istup):
    """Calculate reference water balance.
        This is a Numba routine, which means all the arguments are a single element -- not an array. 
    Args:
        Tx365 (float): a daily value of maximum temperature
        Ta365 (float): a daily value of average temperature
        Pcp365 (float): a daily value of precipitation
        Txsnm (float): the maximum temperature threshold, underwhich precip. falls as snow
        Fsnm (float): snow melt parameter
        Eto365 (float): a daily value of reference evapotranspiration
        wb_old (float): water bucket value from the previous day
        sb_old (float): snow bucket value from the previous day
        doy (int): day of year
        istart0 (int): the starting date of the growing period
        istart1 (int): the ending date of the growing period
        Sa (int): total available soil water holding capacity
        D (int): rooting depth
        p (int): the share of exess water, below which soil moisture starts to become difficult to extract
        lgpt5_point (float): numbers of days with mean daily tenmperature above 5 degC
        istup (int): temperature trend (upward = 1, downward = -1)

    Returns:
        Eta365 (float): a daily value of the 'Actual Evapotranspiration' (mm)
        Etm365 (float): a daily value of the 'Maximum Evapotranspiration' (mm)
        Wb365 (float): a daily value of the 'Soil Water Balance'
        Wx365 (float): a daily value of the 'Maximum water available to plants'
        Sb365 (float): a daily value of the 'Snow balance' (mm)
        kc365 (float): a daily value of the 'crop coefficients for water requirements'
    """
    if lgpt5_point >= 365:
        kc0 = 1.
    else:
        kc0 = 0.5
    kc1 = 0.
    kc2 = 0.1
    kc3, kc7 = 0.2, 0.2
    kc4, kc6 = 0.5, 0.5
    kc5 = 1.

    Wx365 = 0


    # Period with Tmax <= Txsnm (precipitaton falls as snow)
    if Tx365 <= Txsnm:
        kc365 = kc1
        etm = kc1 * Eto365

        Etm365 = etm

        sbx = sb_old+Pcp365

        if sbx >= etm:
            Sb365 = sbx - etm
            Eta365 = etm
        else:
            Sb365 = 0
            wb, wx, Eta = eta(wb_old, etm-sbx, Sa, D, p, 0.)
            Eta365 = Eta + sbx

        Wb365 = wb
        sb =Sb365

    # period with Txsnm < Tmax; Ta <=0 (precipitation is water; 100% runoff)
    # Snow-melt takes place; minor evapotranspiration
    elif Ta365 <= 0:
        kc365 = kc2
        etm = kc2 * Eto365
        Etm365 = etm

        # Snow-melt function
        snm = min(Fsnm*(Tx365-Txsnm), sb_old)
        sb = sb_old - snm
        wb = wb_old + snm
        sbx = sb

        if sbx >= etm:
            Sb365 = sbx-etm
            Eta365 = etm

            if wb > Sa:
                wx = wb - Sa + Pcp365
                wb = Sa
            else:
                wx = Pcp365
        else:
            Sb365 = 0
            wb, wx, Eta = eta(wb, etm - sbx, Sa, D, p, Pcp365)
            Eta365 = Eta + sbx
        
        Wb365 =wb
        sb = Sb365
        Wx365 = wx

    # Periods with 0 < Ta < 5
    elif Ta365 < 5.:

        # In FORTRAN, the routine check whether the temperature trend is upward or downward
        # but the kc determination is the same.
        kc365 = kc3
        # Biological activities before start of growing period OR
        # Reduced biological activitites before dormancy
        etm = kc3 * Eto365
        Etm365 = etm

        # In case there is still snow
        if sb_old > 0.:
            snm = min(Fsnm*(Tx365-Txsnm), sb_old)
        else:
            snm = 0.

        wb = wb_old + snm
        sb = sb_old - snm
        wb, wx, Eta = eta(wb, etm, Sa, D, p, Pcp365)

        Eta365 =Eta
        Sb365 = sb
        Wx365 = wx
        Wb365 = wb

    # periods with Ta >5
    elif Ta365 >= 5.:

        if istart0+1 >0 and istart1+1 <= 365:
            # case 2 -- kc increases from 0.5 to 1.0 during first month of LGP
            if doy+1 >= istart0+1 and doy+1 <= istart1+1:
                xx = min((doy+1-(istart0+1))/30., 1.)
                kc = (kc0*(1.-xx))+(kc5*xx)
            elif istup == 1:
                kc = kc4
            else:
                kc = kc6
        elif istart0+1 >0 and istart1+1 > 365:
            ii = (istart1 % 365) +1
            if doy+1 >= istart0+1:
                xx = min((doy+1-(istart0+1))/30., 1.)   
                kc = (kc0 * (1.-xx)) + (kc5*xx)
            elif doy+1 <= ii:
                xx = min((doy+1+365-(istart0+1))/ 30., 1.0)
                kc = kc0*(1.-xx)+(kc5*xx)
            elif istup == 1:
                kc = kc4
            else:
                kc = kc6
        else:
            kc = kc5 # kc5

        kc365 = kc
        etm = kc * Eto365
        Etm365 = etm
        # In case there is still snow
        if sb_old > 0.:
            snm = min(Fsnm*(Tx365-Txsnm), sb_old)
        else:
            snm = 0.
        
        wb = wb_old + snm
        sb = sb_old - snm

        wb, wx, Eta = eta(wb, etm, Sa, D, p, Pcp365)
        Eta365 = Eta
        Sb365 = sb
        Wx365 = wx
        Wb365 = wb

    # added logic for values less than zero
    if Eta365 <0: Eta365 = 0
    if Etm365 <0: Etm365 = 0
    if Wb365 <0: Wb365 = 0
    if Wx365 <0: Wx365 = 0
    if Sb365 <0: Sb365 = 0
    if kc365 <0: kc365 = 0.

    return Eta365, Etm365, Wb365, Wx365, Sb365, kc365


@jit(nopython=True)
def setdat(dat1):
    if dat1 > 365:
        dat1 = dat1-365
    return dat1


@jit(nopython=True)
def islgpt(Ta):
    ist5 = np.zeros((np.shape(Ta)))
    for i in range(len(Ta)):
        if Ta[i] >= 5:
            ist5[i] = 1
        else:
            ist5[i] = 0

    return ist5


def search_cycles(array):
    """
    LGP subroutine: find the possible cycles which ETa >= 0.4 ETm.
    The routine collect the total number of cycles, index of the DOY 
    of each LGP cycle.
    
    Args:
        array [1-D NumPy Array]: boolean array of ETa/ETm >= 0.4.
    Return:
        [cycle_list, DOY index list]: a python list containing cycles
                                      and beginning date index of each cycle.
    """
    final_cycles = [] # a list of collect all possible LGP cycles
    cycle_idx = [] # a list of each LGP cycle's beginning date index
    Onecycle = [] # a list for looping purposes; to collect the LGP days.

    stopdate = False

    for j in range(len(array)):
        i = array[j]
        if i == 1:
            Onecycle.append(i) # start day count for that particular cycle
            if stopdate == False:
                cycle_idx.append(j) # Attach the doy index of the detected cycle
                stopdate = True  # after attaching the date, stop counting until
            else:
                pass
        else:
            if len(Onecycle) == 0:
                continue
            else:
                final_cycles.append(Onecycle)
                Onecycle = []
                stopdate = False

    final_cycles.append(Onecycle)
    return final_cycles, cycle_idx

