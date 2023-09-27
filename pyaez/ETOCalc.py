"""
PyAEZ version 2.1.0 (June 2023)
ETOCalc.py calculates the reference evapotranspiration from 
the climatic data provided by the PyAEZ user.
2020: N. Lakmal Deshapriya, Thaileng Thol
2022/2023: Kittiphon Boonma  (Numba)

"""
import numpy as np
import numba as nb

class ETOCalc(object):
 
    def __init__(self, cycle_begin, cycle_end, latitude, altitude):
        """Initiate a ETOCalc Class instance

        Args:
            cycle_begin (int): Julian day for the beginning of crop cycle
            cycle_end (int): Julian day for the ending of crop cycle
            latitude (float): a latitude value
            altitude (float): an altitude value
        """        
        self.cycle_begin = cycle_begin
        self.cycle_end = cycle_end
        self.latitude = latitude
        self.alt = altitude

    def setClimateData(self, min_temp, max_temp, wind_speed, short_rad, rel_humidity):
        """Load the climatic (point) data into the Class

        Args:
            min_temp (float): Minimum temperature [Celcius]
            max_temp (float): Maximum temperature [Celcius]
            wind_speed (float): Windspeed at 2m altitude [m/s]
            short_rad (float): Radiation [MJ/m2.day]
            rel_humidity (float): Relative humidity [decimal percentage]
        """        

        self.minT_daily = min_temp # Celcius
        self.maxT_daily = max_temp # Celcius
        self.windspeed_daily = wind_speed # m/s at 2m
        self.shortRad_daily = short_rad # MJ/m2.day
        self.rel_humidity = rel_humidity # Fraction

    @staticmethod
    @nb.jit(nopython=True)
    def calculateETONumba(cycle_begin, cycle_end, latitude, alt,  minT_daily, maxT_daily, windspeed_daily, shortRad_daily, rel_humidity):
        """Calculate the reference evapotranspiration with Penmann-Monteith Equation

        Returns:
            float: ETo of a single pixel (function is called pixel-wise)
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
        es = 0.5*(es_tmin + es_tmax)
        ea = rel_humidity * es  # Actual Vapor Pressure derived from relative humidity

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

        # aerodynamic resistance
        rhoa = 208/u2m

        # crop canopy resistance
        Rl = 100  # daily stomata resistance of a single leaf (s/m)
        # Standard is xLAO = 24
        RLAI = 24 * 0.12
        rhoc = Rl/(0.5*RLAI)  # crop canopy resistance

        gamst = gam * (1. + rhoc/rhoa)

        # net radiation Rn = Rns - Rnl
        # Julien Days
        dayoyr = np.arange(cycle_begin, cycle_end+1)


        latr = latitude * np.pi/180.

        # (a) calculate extraterrestrial radiation
        # solar declination (rad)
        sdcl = 0.4093 * np.sin(0.017214206 * dayoyr - 1.405)
        # relative distance earth to sun
        sdst = 1.0 + 0.033 * np.cos(0.017214206 * dayoyr)
        xx = np.sin(sdcl) * np.sin(latr)
        yy = np.cos(sdcl) * np.cos(latr)
        zz = xx/yy
        # omg = np.arccos(-np.tan(latr)*np.tan(sdcl))  # Sunset hour angle (rad)

        omg = np.tan(zz / (1. - zz*zz)**0.5) + 1.5708
        dayhr = 24. * (omg/np.pi)

        omg[np.where(np.logical_and(np.abs(zz) >= 0.9999, zz > 0))] = np.pi
        dayhr[np.where(np.logical_and(np.abs(zz) >= 0.9999, zz > 0))] = 23.999

        omg[np.where(np.logical_and(np.abs(zz) >= 0.9999, zz <= 0))] = 0
        dayhr[np.where(np.logical_and(np.abs(zz) >= 0.9999, zz <= 0))] = 0.001

        ra = 37.586 * sdst * (omg*xx + np.sin(omg)*yy)

        # (b) solar radiation Rs (0.25, 0.50 Angstrom coefficients)
        # rs = (0.25 + (0.50 * (sd/dayhr))) * ra
        rs = shortRad_daily
        rs0 = (0.75 + 0.00002 * alt) * ra

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
        rnl = (((273.16+maxT_daily)**4)+((273.16 + minT_daily)**4)) * \
            (0.34 - (0.14*(ea**0.5))) * \
            ((1.35*(rs/rs0))-0.35)*sub_cst/2

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
        et0ady = gam/(dl+gamst) * 900./(tavg+273.) * u2m * (es-ea)
        et0rad = dl/(dl+gamst) * (rn-G) / lam

        et0 = et0ady + et0rad

        et0[et0 < 0] = 0

        return et0

    def calculateETO(self):
        return ETOCalc.calculateETONumba(self.cycle_begin, self.cycle_end, self.latitude, self.alt,  self.minT_daily, self.maxT_daily, self.windspeed_daily, self.shortRad_daily, self.rel_humidity)
