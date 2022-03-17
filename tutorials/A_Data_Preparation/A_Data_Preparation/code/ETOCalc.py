"""
PyAEZ
Written by Thaileng Thol
"""

import numpy as np

class ETOCalc(object):

    def __init__(self, cycle_begin, cycle_end, latitude, altitude):
        self.cycle_begin = cycle_begin
        self.cycle_end = cycle_end
        self.latitude = latitude
        self.alt = altitude

    def setClimateData(self, min_temp, max_temp, wind_speed, short_rad, rel_humidity):

        self.minT_daily = min_temp # Celcius
        self.maxT_daily = max_temp # Celcius
        self.windspeed_daily = wind_speed # m/s at 2m
        self.shortRad_daily = short_rad # MJ/m2.day
        self.rel_humidity = rel_humidity # Fraction

    def calculateETO(self):

        # Monthly Average Temperature
        ta = (self.minT_daily + self.maxT_daily)/2

         # Saturation Vapor Pressure
        es_tmin = 0.6108 * np.exp((17.27 * self.minT_daily) / (self.minT_daily + 237.3))
        es_tmax = 0.6108 * np.exp((17.27 * self.maxT_daily) / (self.maxT_daily + 237.3))
        es = (es_tmin + es_tmax) / 2.0

        # Actual Vapor Pressure
        ea = self.rel_humidity * es
        # when relative humidity data is not available, we can use following approximation based min temperature
        #ea = 0.611 * np.exp((17.27 * self.minT_daily)/(self.minT_daily + 237.3))

        # Latitude in Radian
        lat_rad = self.latitude * np.pi/180

        # Julien Days
        doy = np.arange(self.cycle_begin, self.cycle_end+1)
        sd = 0.4093 * np.sin((2*np.pi*doy/365)-1.39)

        # Sunset Hour angle
        sha = np.arccos(-np.tan(lat_rad)*np.tan(sd))

        # Relative distance Earth to Sun
        dr = 1 + 0.033*np.cos(2*np.pi*doy/365)

        # Extraterrestrial radiation (top of atmosphere radiation)
        Ra = 0.082*(24*60/np.pi)*dr*((sha*np.sin(lat_rad)*np.sin(sd))+(np.cos(lat_rad)*np.cos(sd)*np.sin(sha)))

        # Clear-sky solar radiation
        Rso= Ra*(0.75+(0.00002*self.alt))

        # Net incoming Short-wave radiation
        alpha = 0.23
        Rns = self.shortRad_daily * (1-alpha)

        # Net outgoing long wave-radiation
        sub_cst = 0.000000004903  # Stefan-Boltzmann constant [MJ K-4 m-2 day-1]
        Rnl = (((273.16+self.maxT_daily)**4)+((273.16 +self.minT_daily)**4))*(0.34 - (0.14*(ea**0.5)))*((1.35*(self.shortRad_daily/Rso))-0.35)*sub_cst/2

        # Net Radiation flux at crop surface
        Rn = Rns - Rnl

        # Slope of Vapour Presure
        D = 4098 * (0.6108*np.exp((17.27*ta) /(237.3+ta)))/(ta+237.3)**2

        # Atmospheric Pressure
        P = 101.3*np.power(((293-(0.0065*self.alt))/293),5.26)

        # Psychrometric Constant
        psy = 0.000665*P

        # Soil Heat Flux - G
        ta_dublicate_last2 = np.append(ta, [ta[-1], ta[-1]])
        ta_dublicate_first2 = np.append(np.array([ta[0], ta[0]]), ta)
        G = 0.14 * (ta_dublicate_last2 - ta_dublicate_first2)
        G = G[0:G.size-2]

        # ETo Penman Monteith
        ETo_a = 0.408*D*(Rn-G)
        ETo_b = psy*900*self.windspeed_daily*(es-ea)/(ta+273)
        ETo_c = D + (psy*(1+0.34*self.windspeed_daily))
        ETo = (ETo_a + ETo_b)/ETo_c

        ETo[ETo < 0]= 0

        return ETo
