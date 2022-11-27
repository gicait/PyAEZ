
"""
PyAEZ: LGPCalc.py calculates the length of growing period (LGP)
2022: K.Boonma ref. GAEZv4
"""

import numba as nb
import numpy as np


@nb.jit(nopython=True)
def rainPeak(totalPrec_monthly,meanT_daily,lgpt5_point):
    """Scan the monthly precipitation for the month with peak rainfall over 3 months

    Args:
        totalPrec_monthly (float): value of monthly precipitation for a pixel
        meanT_daily (float): daily mean temperature for a pixel
        lgpt5_point (float): thermal length of growing period (T>5C) for a pixel

    Returns:
        int: month of the peak rainfall
        1D NumPy: the smoothened daily mean temperature curve
        int: the starting date of the growing period
        int: the ending date of the growing period
    """    
    #============================================
    # Find peak rain (over 3 months)
    mon = np.arange(0,12)
    rainmax = np.zeros((np.shape(totalPrec_monthly)))
    rainmax[0] = totalPrec_monthly[11]+totalPrec_monthly[0]+totalPrec_monthly[1]

    for i in range(1,11):
        rainmax[i] = totalPrec_monthly[i-1]+totalPrec_monthly[i]+totalPrec_monthly[i+1]

    rainmax[11] = totalPrec_monthly[10]+totalPrec_monthly[11]+totalPrec_monthly[0]
    mon_rainmax=mon[rainmax==rainmax.max()][0]
    #============================================

    days = np.arange(0,365)
    deg = 5
    mat = np.zeros((days.shape[0], deg + 1))
    mat[:, 0] = np.ones_like(days)
    for n in range(1, deg + 1):
        mat[:, n] = (days**n)
    y = meanT_daily
    pinv = np.linalg.pinv(mat)
    p = (pinv*y).sum(axis=-1)
    days_f = np.arange(0., 365.)
    meanT_daily_new = np.zeros_like(days_f)
    for coeff in p[::-1]:
        meanT_daily_new = days_f * meanT_daily_new + coeff
    lgpt5_veg = days_f[meanT_daily_new >= 5.0]
    
    #============================================
    if lgpt5_point < 365.:
        istart0 = lgpt5_veg[0]
        istart1 = istart0 + lgpt5_point-1
    else:
        istart0 = 0.
        istart1 = lgpt5_point-1

    return meanT_daily_new,istart0,istart1
#============================================
@nb.jit(nopython=True)
def isfromt0(meanT_daily_new,doy):
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

#============================================

@nb.jit(nopython=True)
def eta(wb_old,etm,Sa,D,p,rain,snm):
    """Calculate actual evapotranspiration (ETa)

    Args:
        wb_old (float): daily water balance left from the previous day
        etm (float): maximul evapotranspiration
        Sa (float): Available soil moisture holding capacity [mm/m]
        D (float): rooting depth [m]
        p (float): soil moisture depletion fraction (0-1)
        rain (float): amount of rainfall
        snm (float): amount of snow-melt

    Returns:
        float: a value for daily water balance
        float: a value for total available soil moisture
        float: the calculated actual evapotranspiration
    """    
    # wb = wb_old+rain
    wx=0.
    Salim = max(Sa*D,1.)
    wr = 100*(1.-p)
    # Rasw = min(wr,wx)

    if rain>= etm:
        eta = etm
    elif rain<etm and rain+wb_old+snm-wr > etm:
        eta = etm
    else:
        rho = (rain+wb_old+snm)/wr
        # eta = min(rain+rho*etm,etm)
        eta = rho*etm

    # wb=wb_old+rain-eta
    wb = min(wb_old+snm+rain-eta,Salim)
    # wb=wb_old+snm+rain-eta
    if wb>Salim:
        wx = wb-Salim
    else:
        wx = Salim
    
    if wb<0.: wb=0.

    return wb, wx, eta


@nb.jit(nopython=True)
def psh(ng,et0):
    """Calculate soil moisture depletion fraction (0-1)

    Args:
        ng (float): crop group
        et0 (float): potential evapotranspiration [mm/day]

    Returns:
        float: soil moisture depletion fraction
    """    
    #ng = crop group
    # eto = potential evapotranspiration [mm/day]
    if ng==0.:
        psh0 = 0.5
    else:
        psh0 = 0.3+(ng-1)*.05

    psh = psh0 + .04 * (5.-et0)

    if psh < 0.1:
        psh = 0.1
    elif psh > 0.8:
        psh = 0.8
    
    return psh


@nb.jit(nopython=True)
def EtaCalc(Tx365, Ta365, Pcp365, Txsnm, Fsnm, Eto365, wb_old,sb_old,doy,fromT0,istart0,istart1,Sa,D,p,kc_list,lgpt5_point):
    """Calculate actual evapotranspiration (ETa)
    """    
    # Period with Tmax <= Txsnm (precipitaton falls as snow as is added to snow bucket)
    if lgpt5_point<365. and Tx365 < 0. and Ta365 <=0.:
        kc = kc_list[0]
        etm = kc * Eto365
        Etm365 = etm
        snm=0.
        sbx = sb_old -snm+ Pcp365              
        wb, wx, Eta = eta(wb_old, etm, Sa, D, p, 0., snm)
        if sbx >= etm:
            sb = sbx - etm
            # Eta365 = etm
        else:
            sb = 0.

        Eta365 = Eta
        Wb365 = wb
        Wx365 = 0. 
        Sb365 = sb
        kc365 = kc

    # Snow-melt takes place; minor evapotranspiration
    elif lgpt5_point < 365. and Ta365 <= 0. and Tx365>=0.:
        kc = kc_list[1]
        etm = kc * Eto365
        Etm365 = etm

        # Snow-melt function
        snm = min(Fsnm*(Tx365-Txsnm),sb_old)
        sb=sb_old-snm
        # wb_snm=wb_old+snm
        sbx = sb     
        if sbx >= etm:
            Sb365 = sbx-etm
            Eta365 = etm
            if wb_old > Sa:
                wx = wb_old -Sa + Pcp365
                wb =Sa
            else:
                wx = Pcp365
        else:
            Sb365 = 0.
            wb, wx, Eta = eta(wb_old, etm, Sa, D, p, Pcp365, snm)
        Eta365 = Eta
        Wb365 = wb
        Wx365 = wx
        kc365 = kc

    elif lgpt5_point < 365. and Ta365 > 0. and Ta365 < 5.:
        # if fromT0 == 1:
            # Biological activities before start of growing period
        kc = kc_list[2]
        # else:
        #     # Reduced biological activities before dormancy
        #     kc = kc_list[6]

        etm = kc * Eto365
        Etm365 = etm

        #In case there is still snow
        if sb_old > 0.:
            snm = min(Fsnm*(Tx365-Txsnm), sb_old)
        else:
            snm = 0.

        # wb_snm = wb_old+snm
        sb = sb_old-snm
        wb, wx, Eta = eta(wb_old, etm, Sa, D, p, Pcp365, snm)
        
        Eta365 = Eta
        Wx365 = wx
        Wb365 = wb
        Sb365 = sb
        kc365 = kc

    elif lgpt5_point < 365. and Ta365 >= 5.:
        # if istart0 > 0. and istart1 < 365.:
        if doy >= istart0 and doy <= istart1: 
                #case 2 -- kc increases from 0.5 to 1.0 during first month of LGP
                #case 3 -- kc = 1 until daily Ta falls below 5C
            xx = min((doy-istart0)/30.,1.)
            kc = kc_list[3]*(1.-xx)+(kc_list[4]*xx)
            # elif fromT0 == 1: 
                #case 1 -- kc=0.5 for days until start of growing period
                # kc = kc_list[3]
        else:
            #     #case 1 -- kc=0.5 for days until start of growing period
            kc = kc_list[5]
        # elif istart0 > 0. and istart1 > 365.:
        # # This is the case for the areas with year-round temperature growing period
        #         kc = kc_list[5]
        # else:
        # This is the case for the areas with year-round temperature growing period kc=1
            # kc = kc_list[4]
            
        etm = kc* Eto365
        Etm365 = etm
        
        #In case there is still snow
        # if sb_old > 0.: 
        #     snm = min(Fsnm*(Tx365-Txsnm), sb_old)
        # else:
        #     snm = 0.
        snm=0.
        sb=sb_old-snm
        # wb_snm=wb_old+snm
        wb, wx, Eta = eta(wb_old, etm, Sa, D, p, Pcp365, snm)
        
        Eta365 = Eta
        Wb365 = wb
        Wx365 = wx
        Sb365 = sb
        kc365 = kc
    else:
        kc = kc_list[4]
        etm = kc * Eto365
        Etm365 = etm
        #In case there is still snow
        # if sb_old > 0.:
        #     snm = min(Fsnm*(Tx365-Txsnm), sb_old)
        # else:
        #     snm = 0.
        snm=0.
        # sb = sb_old-snm
        # wb_snm=wb_old+snm
        wb, wx, Eta = eta(wb_old, etm, Sa, D, p, Pcp365, snm)

        Eta365 = Eta
        Wb365 = wb
        Wx365 = wx
        Sb365 = sb
        kc365 = kc


    return Eta365, Etm365, Wb365,Wx365,Sb365,kc365




