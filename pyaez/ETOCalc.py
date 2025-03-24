"""
PyAEZ version 2.3 (Dec 2023)
ETOCalc.py calculates the reference evapotranspiration from the climatic data provided by the PyAEZ user.

2020: N. Lakmal Deshapriya, Thaileng Thol
2022/2023: Kittiphon Boonma  (Numba)
2023 (Dec): Swun Wunna Htet
2024 (Dec): Swun Wunna Htet

Modification
------------
1. Removed the object class type for ETO calculation.
"""
import numpy as np
import numba as nb

@nb.jit(nopython=True)
def calculateETONumba(cycle_begin, cycle_end, latitude, alt,  minT_daily, maxT_daily, windspeed_daily, shortRad_daily, rel_humidity, leap_year:bool = False):
    """Calculate the reference evapotranspiration with Penmann-Monteith Equation

    Arguments:
        cycle_begin (int): Julian day for the beginning of crop cycle
        cycle_end (int): Julian day for the ending of crop cycle
        latitude (float): a latitude value [Decimal Degrees]
        altitude (float): an altitude value [meters]
        min_temp (float): Minimum temperature [Celcius]
        max_temp (float): Maximum temperature [Celcius]
        wind_speed (float): Windspeed at 2m altitude [m/s]
        short_rad (float): Radiation [MJ/m2.day]
        rel_humidity (float): Relative humidity [decimal percentage]
        leap_year (bool): True for leap year, False for non-leap year.
    Returns:
        eto (1-D NumPy Array): pixel-based time-series reference evapotranspiration [mm/day]
    """        
    # constants
    tavg = 0.5*(maxT_daily+minT_daily)  # Averaged temperature
    lam = 2.501 - 0.002361 * tavg  # Latent heat of vaporization

    # Wind speed
    u2m = windspeed_daily.copy()
    # limit to no less than 0.5 m/s; FAO 56, p.63
    u2m[windspeed_daily < 0.5] = 0.5

    # Mean Saturation Vapor Pressure derived from air temperature
    es_tmin = 0.6108 * np.exp((17.27 * minT_daily) / (minT_daily + 237.3))
    es_tmax = 0.6108 * np.exp((17.27 * maxT_daily) / (maxT_daily + 237.3))
    ea = 0.5 * (es_tmin + es_tmax)
    ed = rel_humidity * ea

    # es = 0.5*(es_tmin + es_tmax)
    # ea = rel_humidity * es  # Actual Vapor Pressure derived from relative humidity

    # slope vapour pressure curve
    dlmx = 4098. * es_tmax / (maxT_daily + 237.3)**2
    dlmn = 4098. * es_tmin / (minT_daily + 237.3)**2
    dl = 0.5* (dlmx + dlmn)

    # Atmospheric pressure
    ap = 101.3*np.power(((293-(0.0065*alt))/293), 5.256)

    # Psychrometric constant
    gam = 0.0016286 * ap/lam

    hw = 200.
    ht = 190.
    hc = 12.

    # aerodynamic resistance (changed based on FORTRAN routine)
    rhoa = (np.log((hw-(0.667*hc))/(0.123*hc)) * np.log((ht-(0.667*hc))/(0.0123*hc)))/ (0.41 * 0.41)

    # crop canopy resistance
    Rl = 100  # daily stomata resistance of a single leaf (s/m)
    
    # Standard is xLAI = 24
    RLAI = 24 * 0.12
    rhoc = Rl/(0.5*RLAI)  # crop canopy resistance

    gamst = gam * (1. + (rhoc/rhoa * u2m))

    # net radiation Rn = Rns - Rnl
    # Julien days of middle day of months
    dayoyr = np.arange(cycle_begin, cycle_end+1)
    months = np.arange(1,13)

    if leap_year:
        dayoyr[:31] = int(30.42 * months[0] - 15.23)
        dayoyr[31:60]=  int(30.42 * months[1] - 15.23)
        dayoyr[60:91]=  int(30.42 * months[2] - 15.23)
        dayoyr[91:121]=  int(30.42 * months[3] - 15.23)
        dayoyr[121:152]=  int(30.42 * months[4] - 15.23)
        dayoyr[152:182]=  int(30.42 * months[5] - 15.23)
        dayoyr[182:213]=  int(30.42 * months[6] - 15.23)
        dayoyr[213:244]=  int(30.42 * months[7] - 15.23)
        dayoyr[244:274]=  int(30.42 * months[8] - 15.23)
        dayoyr[274:305]=  int(30.42 * months[9] - 15.23)
        dayoyr[305:335]=  int(30.42 * months[10] - 15.23)
        dayoyr[335:]=  int(30.42 * months[11] - 15.23)
    else:
        dayoyr[:31] =  int(30.42 * months[0] - 15.23)
        dayoyr[31:59]=  int(30.42 * months[1] - 15.23)
        dayoyr[59:90]=  int(30.42 * months[2] - 15.23)
        dayoyr[90:120]=  int(30.42 * months[3] - 15.23)
        dayoyr[120:151]=  int(30.42 * months[4] - 15.23)
        dayoyr[151:181]=  int(30.42 * months[5] - 15.23)
        dayoyr[181:212]=  int(30.42 * months[6] - 15.23)
        dayoyr[212:243]=  int(30.42 * months[7] - 15.23)
        dayoyr[243:273]=  int(30.42 * months[8] - 15.23)
        dayoyr[273:304]=  int(30.42 * months[9] - 15.23)
        dayoyr[304:334]=  int(30.42 * months[10] - 15.23)
        dayoyr[334:]=  int(30.42 * months[11] - 15.23)

    latr = latitude * np.pi/180.

    # (a) calculate extraterrestrial radiation
    # solar declination (rad)
    sdcl = 0.4093 * np.sin((0.017214206 * dayoyr) - 1.405)
    # relative distance earth to sun
    sdst = 1.0 + 0.033 * np.cos(0.017214206 * dayoyr)
    xx = np.sin(sdcl) * np.sin(latr)
    yy = np.cos(sdcl) * np.cos(latr)
    zz = xx/yy
    
    # calculate extraterrestrial radiation
    # 2*pi/365 = 0.017214206
	# sdcl ... solar declination [rad]
	# sdst ... relative distance earth - sun
	# omg  ... sunset hour angle [rad]

    dayhr = np.zeros(dayoyr.shape)
    omg = np.zeros(dayoyr.shape)

    for i in range(dayhr.shape[0]):
        
        if abs(zz[i]) >= 0.9999:
            if zz[i] >0:
                dayhr[i] = 23.999
                omg[i] = np.pi
            else:
                dayhr[i] = 0.001
                omg[i] = 0.
        else:
            omg[i] = np.arctan(zz[i]/ np.sqrt(1.- (zz[i] * zz[i]))) + 1.5708
            dayhr[i] = 24. * (omg[i]/np.pi)

    ra = 37.586 * sdst * ((omg*xx) + (np.sin(omg)*yy))

    # (b) solar radiation Rs (0.25, 0.50 Angstrom coefficients)
    # In FORTRAN, incoming radiation is calculated from sunshine hour data by this formula
    # rs = (0.25 + (0.50 * (sd/dayhr))) * ra

    # In PyAEZ, incoming shortwave radiation comes from input data
    rs = shortRad_daily
    rs0 = (0.75 + (2e-5 * alt)) * ra

    # (c) net shortwave radiation Rns = (1 - alpha) * Rs
    # (alpha for grass = 0.23)
    rns = 0.77 * rs

    # (d) net longwave radiation Rnl
    # Stefan-Boltzmann constant [MJ K-4 m-2 day-1]
    sub_cst = 4.903e-9

    # rnl = sub_cst * (0.1 + 0.9 * (sd / dayhr)) * (0.34 - 0.139 * np.sqrt(ea)) * \
    #     0.5 * ((maxT_daily + 273.16) **
    #            4 + (minT_daily + 273.16) ** 4)
    # Stefan-Boltzmann constant [MJ K-4 m-2 day-1]
    # rnl = (((273.16+maxT_daily)**4)+((273.16 + minT_daily)**4)) * \
    #     (0.34 - (0.14*(ea**0.5))) * \
    #     ((1.35*(rs/rs0))-0.35)*sub_cst/2
    
    TmK4 = (maxT_daily +273.16)**4
    TnK4 = (minT_daily + 273.16)**4
    err_fct = 0.34 - (0.139 * np.sqrt(ed))
    cloudiness_fct = (1.35 * rs/rs0) - 0.35

    rnl = sub_cst * ((TmK4 + TnK4)/2) * err_fct * cloudiness_fct

    # (e) net radiation Rn = Rns - Rnl
    rn = rns - rnl
    rn0 = rn

    # (f) soil heat flux [MJ/m2/day]
    ta_dublicate_last2 = np.append(tavg, np.array([tavg[-1]]))
    ta_dublicate_first2 = np.append(np.array([tavg[-1]]), tavg)
    G = 0.14 * (ta_dublicate_last2 - ta_dublicate_first2)
    G = G[0:G.size-1]
    # G = 0

    # (g) calculate aerodynamic and radiation terms of ET0

    et0ady = gam/(dl+gamst) * (900./(tavg + 273))* u2m * (ea-ed)
    et0rad = dl/(dl+gamst) * (rn - G)/lam
    
    et0 = et0ady + et0rad

    et0 = np.where(et0<=0., 0, et0)

    return et0

@nb.jit(nopython=True)
def calculateNetRadiationFlux(cycle_begin, cycle_end, latitude, alt,  minT_daily, maxT_daily, shortRad_daily, wind_sp, rel_humidity, leap_year:bool = False):
    """Calculate the net radiation flux based on Penmann-Monteith Equation

    Arguments:
        cycle_begin (int): Julian day for the beginning of crop cycle
        cycle_end (int): Julian day for the ending of crop cycle
        latitude (float): a latitude value [Decimal Degrees]
        altitude (float): an altitude value [meters]
        min_temp (float): Minimum temperature [Celcius]
        max_temp (float): Maximum temperature [Celcius]
        wind_speed (float): Windspeed at 2m altitude [m/s]
        short_rad (float): Radiation [MJ/m2.day]
        rel_humidity (float): Relative humidity [decimal percentage]
        wind_sp (decimal): wind speed measurement at 2 m altitude [m/s]
        leap_year (bool): True for leap year, False for non-leap year.
    Returns:
        Rn (1-D NumPy Array): pixel-based time-series accumulated net radiation [MJ/m2/day]
    """
    # Wind speed
    u2m = wind_sp.copy()
    # limit to no less than 0.5 m/s; FAO 56, p.63
    u2m[wind_sp < 0.5] = 0.5

    # Mean Saturation Vapor Pressure derived from air temperature
    es_tmin = 0.6108 * np.exp((17.27 * minT_daily) / (minT_daily + 237.3))
    es_tmax = 0.6108 * np.exp((17.27 * maxT_daily) / (maxT_daily + 237.3))
    ea = 0.5 * (es_tmin + es_tmax)
    ed = rel_humidity * ea


    # net radiation Rn = Rns - Rnl
    # Julien Days
    dayoyr = np.arange(cycle_begin, cycle_end+1)
    months = np.arange(1,13)

    if leap_year:
        dayoyr[:31] = int(30.42 * months[0] - 15.23)
        dayoyr[31:60]=  int(30.42 * months[1] - 15.23)
        dayoyr[60:91]=  int(30.42 * months[2] - 15.23)
        dayoyr[91:121]=  int(30.42 * months[3] - 15.23)
        dayoyr[121:152]=  int(30.42 * months[4] - 15.23)
        dayoyr[152:182]=  int(30.42 * months[5] - 15.23)
        dayoyr[182:213]=  int(30.42 * months[6] - 15.23)
        dayoyr[213:244]=  int(30.42 * months[7] - 15.23)
        dayoyr[244:274]=  int(30.42 * months[8] - 15.23)
        dayoyr[274:305]=  int(30.42 * months[9] - 15.23)
        dayoyr[305:335]=  int(30.42 * months[10] - 15.23)
        dayoyr[335:]=  int(30.42 * months[11] - 15.23)
    else:
        dayoyr[:31] =  int(30.42 * months[0] - 15.23)
        dayoyr[31:59]=  int(30.42 * months[1] - 15.23)
        dayoyr[59:90]=  int(30.42 * months[2] - 15.23)
        dayoyr[90:120]=  int(30.42 * months[3] - 15.23)
        dayoyr[120:151]=  int(30.42 * months[4] - 15.23)
        dayoyr[151:181]=  int(30.42 * months[5] - 15.23)
        dayoyr[181:212]=  int(30.42 * months[6] - 15.23)
        dayoyr[212:243]=  int(30.42 * months[7] - 15.23)
        dayoyr[243:273]=  int(30.42 * months[8] - 15.23)
        dayoyr[273:304]=  int(30.42 * months[9] - 15.23)
        dayoyr[304:334]=  int(30.42 * months[10] - 15.23)
        dayoyr[334:]=  int(30.42 * months[11] - 15.23)

    latr = latitude * np.pi/180.

    # (a) calculate extraterrestrial radiation
    # solar declination (rad)
    sdcl = 0.4093 * np.sin(0.017214206 * dayoyr - 1.405)
    # relative distance earth to sun
    sdst = 1.0 + 0.033 * np.cos(0.017214206 * dayoyr)
    xx = np.sin(sdcl) * np.sin(latr)
    yy = np.cos(sdcl) * np.cos(latr)
    zz = xx/yy
    
    # calculate extraterrestrial radiation
    # 2*pi/365 = 0.017214206
	# sdcl ... solar declination [rad]
	# sdst ... relative distance earth - sun
	# omg  ... sunset hour angle [rad]

    dayhr = np.zeros(dayoyr.shape)
    omg = np.zeros(dayoyr.shape)

    for i in range(dayhr.shape[0]):
        
        if abs(zz[i]) >= 0.9999:
            if zz[i] >0:
                dayhr[i] = 23.999
                omg[i] = np.pi
            else:
                dayhr[i] = 0.001
                omg[i] = 0.
        else:
            omg[i] = np.arctan(zz[i]/ np.sqrt(1.- (zz[i] * zz[i]))) + 1.5708
            dayhr[i] = 24. * (omg[i]/np.pi)


    ra = 37.586 * sdst * (omg*xx + (np.sin(omg)*yy))

    # (b) solar radiation Rs (0.25, 0.50 Angstrom coefficients)
    # In FORTRAN, incoming radiation is calculated from sunshine hour data by this formula
    # rs = (0.25 + (0.50 * (sd/dayhr))) * ra

    # In PyAEZ, incoming shortwave radiation comes from input data
    rs = shortRad_daily
    rs0 = (0.75 + (0.00002 * alt)) * ra

    # (c) net shortwave radiation Rns = (1 - alpha) * Rs
    # (alpha for grass = 0.23)
    rns = 0.77 * rs

    # (d) net longwave radiation Rnl
    # Stefan-Boltzmann constant [MJ K-4 m-2 day-1]
    sub_cst = 0.000000004903

    # rnl = sub_cst * (0.1 + 0.9 * (sd / dayhr)) * (0.34 - 0.139 * np.sqrt(ea)) * \
    #     0.5 * ((maxT_daily + 273.16) **
    #            4 + (minT_daily + 273.16) ** 4)
    # Stefan-Boltzmann constant [MJ K-4 m-2 day-1]
    # rnl = (((273.16+maxT_daily)**4)+((273.16 + minT_daily)**4)) * \
    #     (0.34 - (0.14*(ea**0.5))) * \
    #     ((1.35*(rs/rs0))-0.35)*sub_cst/2
    
    TmK4 = (maxT_daily +273.15)**4
    TnK4 = (minT_daily + 273.15)**4
    err_fct = 0.34 - (0.139 * np.sqrt(ed))
    cloudiness_fct = (1.35 * rs/rs0) - 0.35

    rnl = sub_cst * ((TmK4 + TnK4)/2) * err_fct * cloudiness_fct

    # (e) net radiation Rn = Rns - Rnl
    rn = rns - rnl

    return rn
# ---------------------------------------------- End of Code ------------------------------------------------------------------- #