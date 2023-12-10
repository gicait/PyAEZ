"""
PyAEZ version 2.2 (Dec 2023)
PyAEZ: LGPCalc.py calculates the length of growing period (LGP)
2022/2023: Kittiphon Boonma 
"""

from numba import jit
import numpy as np


@jit(nopython=True)
def rainPeak(totalPrec_monthly, meanT_daily, lgpt5_point):
    """Scan the monthly precipitation for the month with peak rainfall over 3 months

    Args:
        totalPrec_monthly (float): value of monthly precipitation for a pixel
        meanT_daily (float): daily mean temperature for a pixel
        lgpt5_point (float): thermal length of growing period (T>5C) for a pixel

    Returns:
        meanT_daily(1D NumPy): the smoothened daily mean temperature curve
        istart0(int): the starting date of the growing period
        istart1(int): the ending date of the growing period
    """
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
        eta = min(rain + rho*etm, etm)

    wb=s-eta

    if wb >= Salim:
        wx = wb-Salim
        wb = Salim
    else:
        wx = 0

    
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
    if ng == 0.:
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
        
    """
    # Program to calculate moving average
    arr = Et
    window_size = 10
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(arr) - window_size + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i: i + window_size]
        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        # Shift window to right by one position
        i += 1

    return np.array(moving_averages)


@jit(nopython=True)
def EtaCalc(Tx365, Ta365, Pcp365, Txsnm, Fsnm, Eto365, wb_old, sb_old, doy, istart0, istart1, Sa, D, p, kc_list, lgpt5_point):
    """Calculate actual evapotranspiration (ETa)
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
        kc_list (list): crop coefficients for water requirements
        lgpt5_point (float): numbers of days with mean daily tenmperature above 5 degC

    Returns:
        Eta365 (float): a daily value of the 'Actual Evapotranspiration' (mm)
        Etm365 (float): a daily value of the 'Maximum Evapotranspiration' (mm)
        Wb365 (float): a daily value of the 'Soil Water Balance'
        Wx365 (float): a daily value of the 'Maximum water available to plants'
        Sb365 (float): a daily value of the 'Snow balance' (mm)
        kc365 (float): a daily value of the 'crop coefficients for water requirements'
    """

    # Period with Tmax <= Txsnm (precipitaton falls as snow as is added to snow bucket)
    if Tx365 <= Txsnm and Ta365 <= 0.:
        etm = kc_list[0] * Eto365

        Etm365 = etm

        sbx = sb_old+Pcp365

        wb, wx, Eta = eta(wb_old-Pcp365, etm, Sa, D, p, Pcp365)

        Salim = Sa*D  

        if sbx >= etm:
            Sb365 = sbx-etm
            Eta365 = etm

            wb=wb_old-etm
            if wb > Salim:
                wx = wb- Salim 
                wb = Salim
            else:
                wx = 0
        else:
            Sb365 = 0.
            Eta365 = Eta

        if wb < 0:
            wb = 0

        Wb365 = wb
        Wx365 = wx
        kc365 = kc_list[0]

    # Snow-melt takes place; minor evapotranspiration
    elif Ta365 <= 0. and Tx365 >= 0.:
        etm = kc_list[1] * Eto365
        Etm365 = etm
        ks = 0.1
        # Snow-melt function
        snm = min(Fsnm*(Tx365-Txsnm), sb_old)
        sbx = sb_old - snm 
        Salim = Sa*D
        if sbx >= etm:
            Sb365 = sbx-etm
            Eta365 = etm
            wb = wb_old+snm+Pcp365-etm

            if wb > Salim:
                wx = wb - Salim
                wb = Salim
            else:
                wx = 0
        else:
            Sb365 = 0.
            wb, wx, Eta = eta(wb_old+snm, etm, Sa, D, p, Pcp365)
            Eta365 = Eta

        if wb < 0:
            wb = 0
      
        Wb365 = wb
        Wx365 = wx
        kc365 = kc_list[1]

    elif Ta365 < 5. and Ta365 > 0.:
        # Biological activities before start of growing period
        etm = kc_list[2] * Eto365
        Etm365 = etm

        # In case there is still snow
        if sb_old > 0.:
            snm = min(Fsnm*(Tx365-Txsnm), sb_old)
        else:
            snm = 0.

        sbx = sb_old-snm

        wb, wx, Eta = eta(wb_old+snm, etm, Sa, D, p, Pcp365)

        if Eta > Etm365:
            Eta365 = etm
        else:
            Eta365=Eta

        Wx365 = wx
        Wb365 = wb
        Sb365 = sbx
        kc365 = kc_list[2]

    elif lgpt5_point < 365 and Ta365 >= 5.:
        if doy >= istart0 and doy <= istart1:
            # case 2 -- kc increases from 0.5 to 1.0 during first month of LGP
            # case 3 -- kc = 1 until daily Ta falls below 5C
            xx = min((doy-istart0)/30., 1.)
            kc = kc_list[3]*(1.-xx)+(kc_list[4]*xx)
        else:
            # case 1 -- kc=0.5 for days until start of growing period
            kc = kc_list[3]

        etm = kc * Eto365
        Etm365 = etm
        # In case there is still snow
        if sb_old > 0.:
            snm = min(Fsnm*(Tx365-Txsnm), sb_old)
        else:
            snm = 0.

        sbx = sb_old-snm

        wb, wx, Eta = eta(wb_old+snm, etm, Sa, D, p, Pcp365)

        if Eta > Etm365:
            Eta365 = etm
        else:
            Eta365 = Eta

        Wb365 = wb
        Wx365 = wx
        Sb365 = sbx
        kc365 = kc

    else:
        etm = kc_list[4] * Eto365
        Etm365 = etm
        # In case there is still snow
        if sb_old > 0.:
            snm = min(Fsnm*(Tx365-Txsnm), sb_old)
        else:
            snm = 0.

        wb, wx, Eta = eta(wb_old, etm, Sa, D, p, Pcp365)

        if Eta > Etm365:
            Eta365 = etm
        else:
            Eta365 = Eta
            
        Wb365 = wb
        Wx365 = wx
        Sb365 = sbx
        kc365 = kc_list[4]

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

