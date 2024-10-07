"""
PyAEZ version 2.4 (Dec 2024)
Biomass Calculation

2020: N. Lakmal Deshapriya
2023: Swun Wunna Htet & Kittiphon Boonma
2023 (Dec): Swun Wunna Htet
2024 (Dec): Swun Wunna Htet

Modification:

1. Look-up tables of Ac, Bc, Bo and Pm are cross-checked with De Wit.(1965) publication.
2. Revised cycle start, end and cycle length determination.
3. Minor solar radiation conversion factor revision from 4.189 to 4.1868.
4. Day time temperature calculation corrected to respective day of year interval.
5. Numba incorporation are removed due to lack of support for scipy interpolation.
6. Ac, Bc, and Bo reference tables for Southern Hemisphere are updated.
7. Fixed an issue when day time temperature calculation during year transition is not properly calculated for leap year.
8. The LAI and growth rate multiplier (l) look-up tables are updated. Interpolated l value will be used for the biomass calculation.
9. Removed unnecessary input arguments for the Bn calculation.
10. Scipy interpolation is revoked for Numba enhancement.
11. Object-based programming is revoked. Instead, a single Numba-enhanced functions will be provided.
"""

import numpy as np
import numba as nb
from math import sqrt

# class BioMassCalc(object):

#     def __init__(self, cycle_begin, cycle_end, latitude, leap_year = False):

#         """
#         Biomasss Calculation object class creation.
        
#         Args:
#             cycle_begin (int): starting DOY of the crop cycle
#             cycle_end (int) : end DOY of the crop cycle
#             latitude (float): Decimal latitude of a pixel location
#             leap_year (Boolean): Yes/No for leap year.
#         """

#         self.cycle_begin = cycle_begin
#         self.cycle_end = cycle_end
#         self.cycle_len = cycle_end - cycle_begin

#         self.lat = latitude
#         self.lat_index1 = int(np.floor((latitude)/10));
#         self.lat_index2 = int(np.ceil((latitude)/10));
#         self.lat_t = self.lat_index1 * 10;
#         self.lat_b = self.lat_index2 * 10;
#         self.leap_year = leap_year


#     def setClimateData(self, min_temp, max_temp, short_rad):
#         """
#         Set the climate parameters.
        
#         Args:
#             min_temp = minimum temperature (Deg C, 1-D NumPy Array)
#             max_temp = maximum temperature (Deg C, 1-D NumPy Array)
#             short_rad = shortwave radiation (W/m2, 1-D NumPy Array)
#         """
        
#         self.minT_daily = min_temp
#         self.maxT_daily = max_temp
#         self.shortRad_daily = short_rad

#         # conversion shortwave radiation from W/m2 to cal/cm2/day
#         self.shortRad_daily = self.shortRad_daily * 2.06362854686156 # corrected (5/10/2023)

#         # calculation of mean temperature and day-time temperature
#         self.meanT_daily = (self.minT_daily + self.maxT_daily)/2

#         # day-time temperature corrected and calculated according to DOY
#         self.dT_daily = np.zeros(self.minT_daily.shape[0])

#         doy_lst = np.arange(self.cycle_begin, self.cycle_end+1)
        
#         # reformatting doy list if its out of 365
#         doy_lst = np.where(doy_lst>366, 366-doy_lst, doy_lst) if self.leap_year else np.where(doy_lst>365, 365-doy_lst, doy_lst)
        
#         for i1 in range(self.dT_daily.shape[0]):

#             doy = doy_lst[i1]

#             if self.leap_year:
#                 if doy<=31:# Jan
#                     self.dT_daily[i1] = 0.0278 + 0.3301*self.minT_daily[i1] + 0.6716*self.maxT_daily[i1];
#                 elif doy in range(32, 60+1): # Feb
#                     self.dT_daily[i1] = 0.078 + 0.3345*self.minT_daily[i1] + 0.6672*self.maxT_daily[i1];
#                 elif doy in range(61, 91+1): # Mar
#                     self.dT_daily[i1] = 0.0770 + 0.3392*self.minT_daily[i1] + 0.6642*self.maxT_daily[i1];
#                 elif doy in range(92, 121+1): # Apr
#                     self.dT_daily[i1] = 0.2276 + 0.3466*self.minT_daily[i1] + 0.6536*self.maxT_daily[i1];
#                 elif doy in range(122, 152+1): # May
#                     self.dT_daily[i1] = 0.2494 + 0.3399*self.minT_daily[i1] + 0.6576*self.maxT_daily[i1];
#                 elif doy in range(153, 182+1): # June
#                     self.dT_daily[i1] = 0.9955+ 0.3335*self.minT_daily[i1] + 0.6628*self.maxT_daily[i1];
#                 elif doy in range(183, 213+1): # July
#                     self.dT_daily[i1] = 0.2624+ 0.3351*self.minT_daily[i1] + 0.659*self.maxT_daily[i1];
#                 elif doy in range(214, 244+1): # Aug
#                     self.dT_daily[i1] = 0.2860+ 0.3297*self.minT_daily[i1] + 0.6612*self.maxT_daily[i1];
#                 elif doy in range(245, 274 +1): # Sep
#                     self.dT_daily[i1] = 0.3492+ 0.3410*self.minT_daily[i1] + 0.6515*self.maxT_daily[i1];
#                 elif doy in range(275, 305+1): # Oct
#                     self.dT_daily[i1] = 0.1598+ 0.3394*self.minT_daily[i1] + 0.6591*self.maxT_daily[i1];
#                 elif doy in range(306, 335+1): # Nov
#                     self.dT_daily[i1] = 0.1156+ 0.3476*self.minT_daily[i1] + 0.6571*self.maxT_daily[i1];
#                 else: # Dec
#                     self.dT_daily[i1] = -0.0617+ 0.3462*self.minT_daily[i1] + 0.6649*self.maxT_daily[i1];
#             else:
#                 if doy<=31:# Jan
#                     self.dT_daily[i1] = 0.0278 + 0.3301*self.minT_daily[i1] + 0.6716*self.maxT_daily[i1];
#                 elif doy in range(32, 59+1): # Feb
#                     self.dT_daily[i1] = 0.078 + 0.3345*self.minT_daily[i1] + 0.6672*self.maxT_daily[i1];
#                 elif doy in range(60, 90+1): # Mar
#                     self.dT_daily[i1] = 0.0770 + 0.3392*self.minT_daily[i1] + 0.6642*self.maxT_daily[i1];
#                 elif doy in range(91, 120+1): # Apr
#                     self.dT_daily[i1] = 0.2276 + 0.3466*self.minT_daily[i1] + 0.6536*self.maxT_daily[i1];
#                 elif doy in range(121, 151+1): # May
#                     self.dT_daily[i1] = 0.2494 + 0.3399*self.minT_daily[i1] + 0.6576*self.maxT_daily[i1];
#                 elif doy in range(152, 181+1): # June
#                     self.dT_daily[i1] = 0.9955+ 0.3335*self.minT_daily[i1] + 0.6628*self.maxT_daily[i1];
#                 elif doy in range(182, 212+1): # July
#                     self.dT_daily[i1] = 0.2624+ 0.3351*self.minT_daily[i1] + 0.659*self.maxT_daily[i1];
#                 elif doy in range(213, 243+1): # Aug
#                     self.dT_daily[i1] = 0.2860+ 0.3297*self.minT_daily[i1] + 0.6612*self.maxT_daily[i1];
#                 elif doy in range(244, 273 +1): # Sep
#                     self.dT_daily[i1] = 0.3492+ 0.3410*self.minT_daily[i1] + 0.6515*self.maxT_daily[i1];
#                 elif doy in range(274, 304+1): # Oct
#                     self.dT_daily[i1] = 0.1598+ 0.3394*self.minT_daily[i1] + 0.6591*self.maxT_daily[i1];
#                 elif doy in range(305, 334+1): # Nov
#                     self.dT_daily[i1] = 0.1156+ 0.3476*self.minT_daily[i1] + 0.6571*self.maxT_daily[i1];
#                 else: # Dec
#                     self.dT_daily[i1] = -0.0617+ 0.3462*self.minT_daily[i1] + 0.6649*self.maxT_daily[i1];

#     def setCropParameters(self, LAI, HI, legume, adaptability):
#         """
#         Set the crop-specific parameters for biomass estimation.
        
#         Args:
#             LAI = leaf area index (float)
#             HI = harvest index (float)
#             legume = legume crop or not (boolean, 1 = True, 0 = False)
#             adaptability = FAO crop adaptability class (int, [1,2,3,4])
#             """
#         self.LAi = LAI # leaf area index
#         self.HI = HI # harvest index
#         self.legume = legume # binary value
#         self.adaptability = adaptability-1
        
    
#     def calculateBioMass(self):
#         self.Bn =  BioMassCalc.calculateBiomassNumba(self.cycle_begin, self.cycle_end, self.cycle_len, self.lat, self.lat_t, self.lat_b, 
#                                                self.shortRad_daily, self.meanT_daily, self.dT_daily, self.LAi, self.legume, self.adaptability)


#     def calculateYield(self):
#         self.PYield = np.round(self.Bn * self.HI, 0).astype(int);
#         return self.PYield

 #--------------------------------------------------------- Main Functions Ends Here------------------------------------------------------------#
 # ----------------------------------------------------- Numba Enhanced Functions Starts Here --------------------------------------------------#

# This routine will be included in the session because it produces very close values as scipy cubic spline. And 
# with Numba enhancement, the execution is quite faster from 0.0229s to 0.00299s (10 times faster)

def calculateBiomassNumba(cycle_begin:int, cycle_end:int, cycle_len:int, latitude:float,
                         shortRad_daily, meanT_daily, minT_daily, maxT_daily,
                        LAi:float, legume:int, adaptability:int, leap_year:bool):
    
    lat = latitude
    lat_index1 = int(np.floor((latitude)/10));
    lat_index2 = int(np.ceil((latitude)/10));
    lat_t = lat_index1 * 10;
    lat_b = lat_index2 * 10;

    """Calculate the biomass from the settings of all input variables"""
    # creating the latitude array from
    lat_array = np.arange(-90,100,10)

    # select the index of upper row of 
    top_row_idx = np.argwhere(lat_array == lat_t)[0][0]
    bot_row_idx = np.argwhere(lat_array == lat_b)[0][0]

    """Max Radiation"""
    Ac = np.array([[397, 252, 40, 0, 0, 0, 0, 0, 0, 154, 339, 428], # -90 deg S
                    [393, 248, 81, 3, 0, 0, 0, 0, 28, 162, 334, 424], # -80 deg S
                    [380, 269, 142, 45, 2, 0, 0, 20, 89, 209, 331, 408], # -70 deg S
                    [389, 309, 201, 103, 37, 14, 22, 72, 149, 260, 356, 408], # -60 deg S
                    [405, 344, 254, 163, 92, 61, 73, 131, 207, 304, 380, 418], # -50 deg S
                    [413, 369, 298, 220, 151, 118, 131, 190, 260, 339, 396, 422], # -40 deg S
                    [411, 384, 333, 270, 210, 179, 191, 245, 303, 363, 400, 417], # -30 deg S
                    [399, 386, 357, 313, 264, 238, 249, 293, 337, 375, 394, 400], # -20 deg S
                    [375, 377, 369, 345, 311, 291, 299, 332, 359, 375, 377, 374], # -10 deg S
                    [343,360,369,364,349,337,342,357,368,365,349,337], # 0 deg N
                    [299,332,359,375,377,374,375,377,369,345,311,291],# 10 deg N
                    [249,293,337,375,394,400,399,386,357,313,264,238],# 20 deg N
                    [191,245,303,363,400,417,411,384,333,270,210,179],# 30 deg N
                    [131,190,260,339,396,422,413,369,298,220,151,118],# 40 deg N
                    [73,131,207,304,380,418,405,344,254,163, 92, 61],# 50 deg N
                    [22, 72,149,260,356,408,389,309,201,103, 37, 14], # 60 deg N
                    [0, 20, 89,209,331,408,380,269,142, 45,  2,  0],# 70 deg N
                    [0,  0, 28,162,334,424,393,248, 81,  3,  0,  0],# 80 deg N
                    [0,  0,  0,154,339,428,397,252, 40,  0,  0,  0]])# 90 deg N
    
    """Biomass in open day"""
    Bc = np.array([[302, 215, 35, 0, 0, 0, 0, 0, 0, 131, 269, 319],# -90 deg S
                    [632, 474, 195, 11, 0, 0, 0, 0, 94, 333, 571, 663],# -80 deg S
                    [575, 427, 262, 114, 7, 0, 0, 65, 185, 350, 506, 612],# -70 deg S
                    [523, 436, 316, 195, 94, 49, 66, 151, 254, 383, 487, 544],# -60 deg S
                    [509, 448, 358, 260, 173, 130, 147, 223, 310, 409, 484, 522],# -50 deg S
                    [496, 455, 390, 314, 241, 204, 218, 283, 353, 427, 480, 506],# -40 deg S
                    [483, 456, 412, 356, 299, 269, 281, 333, 385, 437, 471, 489],# -30 deg S
                    [465, 451, 425, 387, 348, 325, 334, 371, 407, 439, 460, 468],# -20 deg S
                    [440, 439, 431, 411, 385, 370, 376, 401, 422, 437, 440, 440],# -10 deg S
                    [413, 424, 429, 426, 417, 410, 413, 422, 429, 427, 418, 410], # 0 deg N
                    [376,401,422,437,440,440,440,439,431,411,385,370], # 10 deg N
                    [334,371,407,439,460,468,465,451,425,387,348,325],# 20 deg N
                    [281,333,385,437,471,489,483,456,412,356,299,269],# 30 deg N
                    [218,283,353,427,480,506,497,455,390,314,241,204], # 40 deg N
                    [147,223,310,409,484,522,509,448,358,260,173,130],# 50 deg N
                    [66,151,254,383,487,544,523,436,316,195, 94, 49], # 60 deg N
                    [0, 65,185,350,506,612,575,427,262,114,  7,  0], # 70 deg N
                    [0,  0, 94,333,571,663,632,474,195, 11,  0,  0], # 80 deg N
                    [0,  0,  0,371,588, 677,646,497,167,  0,  0,  0]]) # 90 deg N
        
    """Biomass in cloudy day"""
    Bo = np.array([[302, 215, 35, 0, 0, 0, 0, 0, 0, 131, 269, 319],# -90 deg S
                    [297, 196, 69, 2, 0, 0, 0, 0, 24, 133, 257, 318],# -80 deg S
                    [273, 200, 112, 38, 1, 0, 0, 16, 74, 158, 241, 291],# -70 deg S
                    [265, 216, 148, 82, 31, 11, 19, 60, 114, 187, 245, 276],# -60 deg S
                    [265, 230, 178, 121, 73, 51, 60, 100, 150, 207, 251, 273],# -50 deg S
                    [263, 239, 200, 155, 112, 91, 99, 137, 178, 223, 253, 268],# -40 deg S
                    [258, 243, 216, 182, 148, 130, 137, 168, 200, 232, 251, 261],# -30 deg S
                    [249, 242, 226, 203, 178, 164, 170, 193, 215, 235, 246, 250],# -20 deg S
                    [236, 235, 230, 218, 203, 193, 197, 212, 225, 234, 236, 235],# -10 deg S
                    [219,226,230,228,221,216,218,225,230,228,222,216],# 0 deg N
                    [197,212,225,234,236,235,236,235,230,218,203,193],# 10 deg N
                    [170,193,215,235,246,250,249,242,226,203,178,164],# 20 deg N
                    [137,168,200,232,251,261,258,243,216,182,148,130],# 30 deg N
                    [99,137,178,223,253,268,263,239,200,155,112, 91],# 40 deg N
                    [60,100,150,207,251,273,265,230,178,121, 73, 51],# 50 deg N
                    [19, 60,114,187,245,276,265,216,148, 82, 31, 11],# 60 deg N
                    [0, 16, 74,158,241,291,273,200,112, 38,  1,  0],# 70 deg N
                    [0,  0, 24,133,257,318,297,196, 69,  2,  0,  0],# 80 deg N
                    [0,  0,  0,131,269,319,302,215, 35,  0,  0,  0]])# 90 deg N

    """Pm values """
    PmIndexExtDtTemp = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]);

    PmIndexExt = np.array([[0.0, 15.0, 20.0, 20.0, 15.0, 5.0, 0.0],
                            [0.0, 0.0, 15.0, 32.5, 35.0, 35.0, 35.0],
                            [0.0, 0.0, 5.0, 45.0, 65.0, 65.0, 65.0],
                            [0.0, 5.0, 45.0, 65.0, 65.0, 65.0, 65.0]]);
    
    """Growth rate multiplier table"""
    LAI_table = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
    l_table = np.array([0.35, 0.6, 0.8, 0.92, 1., 1.05, 1.08, 1.1])

    Ac_interp1 = np.zeros(12);
    for i in range(Ac_interp1.shape[0]):
        Ac_interp1[i] = np.interp(lat, [lat_t, lat_b], [Ac[top_row_idx,:][i], Ac[bot_row_idx,:][i]]);

    Bc_interp1 = np.zeros(12);
    for i in range(Bc_interp1.shape[0]):
        Bc_interp1[i] = np.interp(lat, [lat_t, lat_b], [Bc[top_row_idx,:][i], Bc[bot_row_idx,:][i]]);
    
    Bo_interp1 = np.zeros(12);
    for i in range(Bo_interp1.shape[0]):
        Bo_interp1[i] = np.interp(lat, [lat_t, lat_b], [Bo[top_row_idx,:][i], Bo[bot_row_idx,:][i]]);
    
    # New Middle DOY, 1 additional addded at DOY 15's front and two added after DOY 345 
    new_doy = np.arange(-15, 410, 30);

    # new concatenated interpolated Ac, Bc and Bo
    new_Ac_interp1 = np.concatenate(([Ac_interp1[-1]],  Ac_interp1, [Ac_interp1[0]], [Ac_interp1[1]]));
    new_Bc_interp1 = np.concatenate(([Bc_interp1[-1]],  Bc_interp1, [Bc_interp1[0]], [Bc_interp1[1]]));
    new_Bo_interp1 = np.concatenate(([Bo_interp1[-1]],  Bo_interp1, [Bo_interp1[0]], [Bo_interp1[1]]));

    # Applying Numba-Enhanced Cubic Spline Interpolation to Ac, Bc and Bo
    Ac_interp  = cubic_interp1d(np.arange(cycle_begin, cycle_end), new_doy, new_Ac_interp1)
    Bc_interp  = cubic_interp1d(np.arange(cycle_begin, cycle_end), new_doy, new_Bc_interp1)
    Bo_interp  = cubic_interp1d(np.arange(cycle_begin, cycle_end), new_doy, new_Bo_interp1)


    Ac_mean = np.mean( Ac_interp );
    bc_mean = np.mean( Bc_interp );
    bo_mean = np.mean( Bo_interp );

    meanT_mean = np.mean( meanT_daily );

    # calculation of Day-time temperature
    dT_daily = DayTimeTemperature(minT_daily, maxT_daily, cycle_begin, cycle_end, leap_year)
    dT_mean = np.mean( dT_daily );

    # conversion shortwave radiation from W/m2 to cal/cm2/day
    shortRad_daily_adj = shortRad_daily * 2.06362854686156 # corrected (5/10/2023)
    Rg = np.mean( shortRad_daily_adj);

    '''the Fraction of the Daytime the Sky is Clouded'''
    f_day_clouded = (Ac_mean - (0.5 * Rg))/(0.8 * (Ac_mean));

    '''Maximum net Rate of CO 2 Exchange of Leaves'''

    PmIndexExt_1Row = PmIndexExt[adaptability-1,:];
    iPm = np.interp(dT_mean,PmIndexExtDtTemp,PmIndexExt_1Row);

    '''Adjust for Temperature and LAI'''
    # Calculate Ct (Correct)
    if legume == 1:
        c = 0.0283;
    else:
        c = 0.0108;

    # Fortran => Ct c * (0.044 + 0.0019 * meanT_mean + 0.0010 * np.power(meanT_mean,2)) (Changed)
    # With Gunther's confirmation, coefficients for the squared term will be changed from 0.001 t0 0.00104289
    Ct = c*(0.0044 + (0.0019*meanT_mean) + (0.00104289*np.power(meanT_mean,2)));

    # growth rate multiplier (linear interpolation)
    l = np.interp(LAi, LAI_table, l_table)

    '''Maximum Rate of Gross Biomass Production''' # Minor change
    bgm = 0.
    if iPm > 20:
        bgm = (f_day_clouded * (.8 + (0.01 * iPm)) * bo_mean) + ((1-f_day_clouded) * (.5 + (.025 * iPm)) * bc_mean);
    elif iPm < 20:
        bgm = (f_day_clouded * (.5 + (.025 * iPm)) * bo_mean) + ((1-f_day_clouded) * (.05 * iPm) * bc_mean);
    # 
    elif iPm == 20:
        bgm = (f_day_clouded*bo_mean) + ((1 - f_day_clouded)*bc_mean);

    '''net biomass production '''
    Bn = (0.36 * bgm * l)/((1/cycle_len)+0.25*Ct)

    return Bn

@nb.jit(nopython = True)
def DayTimeTemperature(minT_daily, maxT_daily, cycle_begin:int, cycle_end:int, leap_year:bool):

    # day-time temperature corrected and calculated according to DOY
    dT_daily = np.zeros(minT_daily.shape[0])

    doy_lst = np.arange(cycle_begin, cycle_end+1)
    
    # reformatting doy list if its out of 365
    doy_lst = np.where(doy_lst>366, 366-doy_lst, doy_lst) if leap_year else np.where(doy_lst>365, 365-doy_lst, doy_lst)
    
    for i1 in range(dT_daily.shape[0]):

        doy = doy_lst[i1]

        if leap_year:
            if doy<=31:# Jan
                dT_daily[i1] = 0.0278 + 0.3301*minT_daily[i1] + 0.6716*maxT_daily[i1];
            elif doy in range(32, 60+1): # Feb
                dT_daily[i1] = 0.078 + 0.3345*minT_daily[i1] + 0.6672*maxT_daily[i1];
            elif doy in range(61, 91+1): # Mar
                dT_daily[i1] = 0.0770 + 0.3392*minT_daily[i1] + 0.6642*maxT_daily[i1];
            elif doy in range(92, 121+1): # Apr
                dT_daily[i1] = 0.2276 + 0.3466*minT_daily[i1] + 0.6536*maxT_daily[i1];
            elif doy in range(122, 152+1): # May
                dT_daily[i1] = 0.2494 + 0.3399*minT_daily[i1] + 0.6576*maxT_daily[i1];
            elif doy in range(153, 182+1): # June
                dT_daily[i1] = 0.9955+ 0.3335*minT_daily[i1] + 0.6628*maxT_daily[i1];
            elif doy in range(183, 213+1): # July
                dT_daily[i1] = 0.2624+ 0.3351*minT_daily[i1] + 0.659*maxT_daily[i1];
            elif doy in range(214, 244+1): # Aug
                dT_daily[i1] = 0.2860+ 0.3297*minT_daily[i1] + 0.6612*maxT_daily[i1];
            elif doy in range(245, 274 +1): # Sep
                dT_daily[i1] = 0.3492+ 0.3410*minT_daily[i1] + 0.6515*maxT_daily[i1];
            elif doy in range(275, 305+1): # Oct
                dT_daily[i1] = 0.1598+ 0.3394*minT_daily[i1] + 0.6591*maxT_daily[i1];
            elif doy in range(306, 335+1): # Nov
                dT_daily[i1] = 0.1156+ 0.3476*minT_daily[i1] + 0.6571*maxT_daily[i1];
            else: # Dec
                dT_daily[i1] = -0.0617+ 0.3462*minT_daily[i1] + 0.6649*maxT_daily[i1];
        else:
            if doy<=31:# Jan
                dT_daily[i1] = 0.0278 + 0.3301*minT_daily[i1] + 0.6716*maxT_daily[i1];
            elif doy in range(32, 59+1): # Feb
                dT_daily[i1] = 0.078 + 0.3345*minT_daily[i1] + 0.6672*maxT_daily[i1];
            elif doy in range(60, 90+1): # Mar
                dT_daily[i1] = 0.0770 + 0.3392*minT_daily[i1] + 0.6642*maxT_daily[i1];
            elif doy in range(91, 120+1): # Apr
                dT_daily[i1] = 0.2276 + 0.3466*minT_daily[i1] + 0.6536*maxT_daily[i1];
            elif doy in range(121, 151+1): # May
                dT_daily[i1] = 0.2494 + 0.3399*minT_daily[i1] + 0.6576*maxT_daily[i1];
            elif doy in range(152, 181+1): # June
                dT_daily[i1] = 0.9955+ 0.3335*minT_daily[i1] + 0.6628*maxT_daily[i1];
            elif doy in range(182, 212+1): # July
                dT_daily[i1] = 0.2624+ 0.3351*minT_daily[i1] + 0.659*maxT_daily[i1];
            elif doy in range(213, 243+1): # Aug
                dT_daily[i1] = 0.2860+ 0.3297*minT_daily[i1] + 0.6612*maxT_daily[i1];
            elif doy in range(244, 273 +1): # Sep
                dT_daily[i1] = 0.3492+ 0.3410*minT_daily[i1] + 0.6515*maxT_daily[i1];
            elif doy in range(274, 304+1): # Oct
                dT_daily[i1] = 0.1598+ 0.3394*minT_daily[i1] + 0.6591*maxT_daily[i1];
            elif doy in range(305, 334+1): # Nov
                dT_daily[i1] = 0.1156+ 0.3476*minT_daily[i1] + 0.6571*maxT_daily[i1];
            else: # Dec
                dT_daily[i1] = -0.0617+ 0.3462*minT_daily[i1] + 0.6649*maxT_daily[i1];
    return dT_daily

@nb.jit(nopython = True)        
def cubic_interp1d(x0, x, y):
    """
    Interpolate a 1-D function using cubic splines.
      x0 : a float or an 1d-array
      x : (N,) array_like
          A 1-D array of real/complex values.
      y : (N,) array_like
          A 1-D array of real values. The length of y along the
          interpolation axis must be equal to the length of x.

    Implement a trick to generate at first step the cholesky matrice L of
    the tridiagonal matrice A (thus L is a bidiagonal matrice that
    can be solved in two distinct loops).

    additional ref: www.math.uh.edu/~jingqiu/math4364/spline.pdf
    Alternative for scipy cubic spline interpolation cited to https://stackoverflow.com/questions/31543775/how-to-perform-cubic-spline-interpolation-in-python.
    """
    x = np.asfarray(x)
    y = np.asfarray(y)

    # remove non finite values
    # indexes = np.isfinite(x)
    # x = x[indexes]
    # y = y[indexes]

    # check if sorted
    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]

    size = len(x)

    xdiff = np.diff(x)
    ydiff = np.diff(y)

    # allocate buffer matrices
    Li = np.empty(size)
    Li_1 = np.empty(size-1)
    z = np.empty(size)

    # fill diagonals Li and Li-1 and solve [L][y] = [B]
    Li[0] = sqrt(2*xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0 # natural boundary
    z[0] = B0 / Li[0]

    for i in range(1, size-1, 1):
        Li_1[i] = xdiff[i-1] / Li[i-1]
        Li[i] = sqrt(2*(xdiff[i-1]+xdiff[i]) - Li_1[i-1] * Li_1[i-1])
        Bi = 6*(ydiff[i]/xdiff[i] - ydiff[i-1]/xdiff[i-1])
        z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    i = size - 1
    Li_1[i-1] = xdiff[-1] / Li[i-1]
    Li[i] = sqrt(2*xdiff[-1] - Li_1[i-1] * Li_1[i-1])
    Bi = 0.0 # natural boundary
    z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    # solve [L.T][x] = [y]
    i = size-1
    z[i] = z[i] / Li[i]
    for i in range(size-2, -1, -1):
        z[i] = (z[i] - Li_1[i-1]*z[i+1])/Li[i]

    # find index
    index = np.searchsorted(x, x0)
    np.clip(index, 1, size-1, index)

    xi1, xi0 = x[index], x[index-1]
    yi1, yi0 = y[index], y[index-1]
    zi1, zi0 = z[index], z[index-1]
    hi1 = xi1 - xi0

    # calculate cubic
    f0 = zi0/(6*hi1)*(xi1-x0)**3 + \
         zi1/(6*hi1)*(x0-xi0)**3 + \
         (yi1/hi1 - zi1*hi1/6)*(x0-xi0) + \
         (yi0/hi1 - zi0*hi1/6)*(xi1-x0)
    return f0

 # ---------------------------------------------------------- Developer's Codes ---------------------------------------------------------------------#
 # ------This section is specific to techincal deep-dive investigation of the module (not meant for Module 2 integration) ---------------------------#

    # """Developer Note: This below code section is to investigate for intermediate variables used in internal biomass calcualtion. Do not remove this code snippet below"""
    # @staticmethod
    # @nb.jit(nopython=True)
    # def calculateBioMassinterNumba(self, cycle_begin, cycle_end, cycle_len, lat, lat_t, lat_b, shortRad_daily, meanT_daily, dT_daily, LAi, legume, adaptability):

    #     """Calculate the biomass from the settings of all input variables
    #     """
    #     # creating the latitude array from
    #     lat_array = np.arange(-90,100,10)

    #     # select the index of upper row of 
    #     top_row_idx = np.argwhere(lat_array == lat_t)[0][0]
    #     bot_row_idx = np.argwhere(lat_array == lat_b)[0][0]

    #     """Max Radiation"""
    #     Ac = np.array([[397, 252, 40, 0, 0, 0, 0, 0, 0, 154, 339, 428], # -90 deg S
    #                    [393, 248, 81, 3, 0, 0, 0, 0, 28, 162, 334, 424], # -80 deg S
    #                    [380, 269, 142, 45, 2, 0, 0, 20, 89, 209, 331, 408], # -70 deg S
    #                    [389, 309, 201, 103, 37, 14, 22, 72, 149, 260, 356, 408], # -60 deg S
    #                    [405, 344, 254, 163, 92, 61, 73, 131, 207, 304, 380, 418], # -50 deg S
    #                    [413, 369, 298, 220, 151, 118, 131, 190, 260, 339, 396, 422], # -40 deg S
    #                    [411, 384, 333, 270, 210, 179, 191, 245, 303, 363, 400, 417], # -30 deg S
    #                    [399, 386, 357, 313, 264, 238, 249, 293, 337, 375, 394, 400], # -20 deg S
    #                    [375, 377, 369, 345, 311, 291, 299, 332, 359, 375, 377, 374], # -10 deg S
    #                    [343,360,369,364,349,337,342,357,368,365,349,337], # 0 deg N
    #                    [299,332,359,375,377,374,375,377,369,345,311,291],# 10 deg N
    #                    [249,293,337,375,394,400,399,386,357,313,264,238],# 20 deg N
    #                    [191,245,303,363,400,417,411,384,333,270,210,179],# 30 deg N
    #                    [131,190,260,339,396,422,413,369,298,220,151,118],# 40 deg N
    #                    [73,131,207,304,380,418,405,344,254,163, 92, 61],# 50 deg N
    #                    [22, 72,149,260,356,408,389,309,201,103, 37, 14], # 60 deg N
    #                    [0, 20, 89,209,331,408,380,269,142, 45,  2,  0],# 70 deg N
    #                    [0,  0, 28,162,334,424,393,248, 81,  3,  0,  0],# 80 deg N
    #                    [0,  0,  0,154,339,428,397,252, 40,  0,  0,  0]])# 90 deg N
        
    #     """Biomass in open day"""
    #     Bc = np.array([[302, 215, 35, 0, 0, 0, 0, 0, 0, 131, 269, 319],# -90 deg S
    #                    [632, 474, 195, 11, 0, 0, 0, 0, 94, 333, 571, 663],# -80 deg S
    #                    [575, 427, 262, 114, 7, 0, 0, 65, 185, 350, 506, 612],# -70 deg S
    #                    [523, 436, 316, 195, 94, 49, 66, 151, 254, 383, 487, 544],# -60 deg S
    #                    [509, 448, 358, 260, 173, 130, 147, 223, 310, 409, 484, 522],# -50 deg S
    #                    [496, 455, 390, 314, 241, 204, 218, 283, 353, 427, 480, 506],# -40 deg S
    #                    [483, 456, 412, 356, 299, 269, 281, 333, 385, 437, 471, 489],# -30 deg S
    #                    [465, 451, 425, 387, 348, 325, 334, 371, 407, 439, 460, 468],# -20 deg S
    #                    [440, 439, 431, 411, 385, 370, 376, 401, 422, 437, 440, 440],# -10 deg S
    #                    [413, 424, 429, 426, 417, 410, 413, 422, 429, 427, 418, 410], # 0 deg N
    #                    [376,401,422,437,440,440,440,439,431,411,385,370], # 10 deg N
    #                    [334,371,407,439,460,468,465,451,425,387,348,325],# 20 deg N
    #                    [281,333,385,437,471,489,483,456,412,356,299,269],# 30 deg N
    #                    [218,283,353,427,480,506,497,455,390,314,241,204], # 40 deg N
    #                    [147,223,310,409,484,522,509,448,358,260,173,130],# 50 deg N
    #                    [66,151,254,383,487,544,523,436,316,195, 94, 49], # 60 deg N
    #                    [0, 65,185,350,506,612,575,427,262,114,  7,  0], # 70 deg N
    #                    [0,  0, 94,333,571,663,632,474,195, 11,  0,  0], # 80 deg N
    #                    [0,  0,  0,371,588, 677,646,497,167,  0,  0,  0]]) # 90 deg N
            
    #     """Biomass in cloudy day"""
    #     Bo = np.array([[302, 215, 35, 0, 0, 0, 0, 0, 0, 131, 269, 319],# -90 deg S
    #                    [297, 196, 69, 2, 0, 0, 0, 0, 24, 133, 257, 318],# -80 deg S
    #                    [273, 200, 112, 38, 1, 0, 0, 16, 74, 158, 241, 291],# -70 deg S
    #                    [265, 216, 148, 82, 31, 11, 19, 60, 114, 187, 245, 276],# -60 deg S
    #                    [265, 230, 178, 121, 73, 51, 60, 100, 150, 207, 251, 273],# -50 deg S
    #                    [263, 239, 200, 155, 112, 91, 99, 137, 178, 223, 253, 268],# -40 deg S
    #                    [258, 243, 216, 182, 148, 130, 137, 168, 200, 232, 251, 261],# -30 deg S
    #                    [249, 242, 226, 203, 178, 164, 170, 193, 215, 235, 246, 250],# -20 deg S
    #                    [236, 235, 230, 218, 203, 193, 197, 212, 225, 234, 236, 235],# -10 deg S
    #                    [219,226,230,228,221,216,218,225,230,228,222,216],# 0 deg N
    #                    [197,212,225,234,236,235,236,235,230,218,203,193],# 10 deg N
    #                    [170,193,215,235,246,250,249,242,226,203,178,164],# 20 deg N
    #                    [137,168,200,232,251,261,258,243,216,182,148,130],# 30 deg N
    #                    [99,137,178,223,253,268,263,239,200,155,112, 91],# 40 deg N
    #                    [60,100,150,207,251,273,265,230,178,121, 73, 51],# 50 deg N
    #                    [19, 60,114,187,245,276,265,216,148, 82, 31, 11],# 60 deg N
    #                    [0, 16, 74,158,241,291,273,200,112, 38,  1,  0],# 70 deg N
    #                    [0,  0, 24,133,257,318,297,196, 69,  2,  0,  0],# 80 deg N
    #                    [0,  0,  0,131,269,319,302,215, 35,  0,  0,  0]])# 90 deg N

    #     """Pm values """
    #     PmIndexExtDtTemp = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]);

    #     PmIndexExt = np.array([[0.0, 15.0, 20.0, 20.0, 15.0, 5.0, 0.0],
    #                             [0.0, 0.0, 15.0, 32.5, 35.0, 35.0, 35.0],
    #                             [0.0, 0.0, 5.0, 45.0, 65.0, 65.0, 65.0],
    #                             [0.0, 5.0, 45.0, 65.0, 65.0, 65.0, 65.0]]);
        
    #     """Growth rate multiplier table"""
    #     LAI_table = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
    #     l_table = np.array([0.35, 0.6, 0.8, 0.92, 1., 1.05, 1.08, 1.1])

    #     Ac_interp1 = np.zeros(12);
    #     for i in range(Ac_interp1.shape[0]):
    #         Ac_interp1[i] = np.interp(lat, [lat_t, lat_b], [Ac[top_row_idx,:][i], Ac[bot_row_idx,:][i]]);

    #     Bc_interp1 = np.zeros(12);
    #     for i in range(Bc_interp1.shape[0]):
    #         Bc_interp1[i] = np.interp(lat, [lat_t, lat_b], [Bc[top_row_idx,:][i], Bc[bot_row_idx,:][i]]);
        
    #     Bo_interp1 = np.zeros(12);
    #     for i in range(Bo_interp1.shape[0]):
    #         Bo_interp1[i] = np.interp(lat, [lat_t, lat_b], [Bo[top_row_idx,:][i], Bo[bot_row_idx,:][i]]);
        
    #     # New Middle DOY, 1 additional addded at DOY 15's front and two added after DOY 345 
    #     new_doy = np.arange(-15, 410, 30);

    #     # new concatenated interpolated Ac, Bc and Bo
    #     new_Ac_interp1 = np.concatenate(([Ac_interp1[-1]],  Ac_interp1, [Ac_interp1[0]], [Ac_interp1[1]]));
    #     new_Bc_interp1 = np.concatenate(([Bc_interp1[-1]],  Bc_interp1, [Bc_interp1[0]], [Bc_interp1[1]]));
    #     new_Bo_interp1 = np.concatenate(([Bo_interp1[-1]],  Bo_interp1, [Bo_interp1[0]], [Bo_interp1[1]]));

    #     # Cubin Spline interpolation classes created for Ac, Bc and Bo
    #     cbl_ac = CubicSpline(new_doy, new_Ac_interp1, extrapolate= True);
    #     cbl_bc = CubicSpline(new_doy, new_Bc_interp1, extrapolate= True);
    #     cbl_bo = CubicSpline(new_doy, new_Bo_interp1, extrapolate= True);

    #     # Cubic Spline interpolation for individual DOY within the cycle length for Ac, Bc and Bo

    #     Ac_interp = cbl_ac(np.arange(cycle_begin, cycle_end), extrapolate= True);
    #     Bc_interp = cbl_bc(np.arange(cycle_begin, cycle_end), extrapolate= True);
    #     Bo_interp = cbl_bo(np.arange(cycle_begin, cycle_end), extrapolate= True);


    #     Ac_mean = np.mean( Ac_interp );
    #     bc_mean = np.mean( Bc_interp );
    #     bo_mean = np.mean( Bo_interp );

    #     meanT_mean = np.mean( meanT_daily );
    #     dT_mean = np.mean( dT_daily );
    #     Rg = np.mean( shortRad_daily );

    #     '''the Fraction of the Daytime the Sky is Clouded'''
    #     f_day_clouded = (Ac_mean - (0.5 * Rg))/(0.8 * (Ac_mean));


    #     '''Maximum net Rate of CO 2 Exchange of Leaves'''

    #     PmIndexExt_1Row = PmIndexExt[adaptability,:];
    #     iPm = np.interp(dT_mean,PmIndexExtDtTemp,PmIndexExt_1Row);

    #     '''Adjust for Temperature and LAI'''
    #     # Calculate Ct (Correct)
    #     if legume == 1:
    #         c = 0.0283;
    #     else:
    #         c = 0.0108;

    #     # Fortran => Ct c * (0.044 + 0.0019 * meanT_mean + 0.0010 * np.power(meanT_mean,2)) (Changed)
    #     # With Gunther's confirmation, coefficients for the squared term will be changed from 0.001 t0 0.00104289
    #     Ct = c*(0.0044 + (0.0019*meanT_mean) + (0.00104289*np.power(meanT_mean,2)));

    #     # growth rate multiplier (linear interpolation)
    #     l = np.interp(LAi, LAI_table, l_table)

    #     '''Maximum Rate of Gross Biomass Production''' # Minor change
    #     bgm = 0.
    #     if iPm > 20:
    #         bgm = (f_day_clouded * (.8 + (0.01 * iPm)) * bo_mean) + ((1-f_day_clouded) * (.5 + (.025 * iPm)) * bc_mean);
    #     elif iPm < 20:
    #         bgm = (f_day_clouded * (.5 + (.025 * iPm)) * bo_mean) + ((1-f_day_clouded) * (.05 * iPm) * bc_mean);
    #     # 
    #     elif iPm == 20:
    #         bgm = (f_day_clouded*bo_mean) + ((1 - f_day_clouded)*bc_mean);

    #     '''net biomass production '''
    #     Bn = (0.36 * bgm * l)/((1/cycle_len)+0.25*Ct)
        
    #     return (np.array([Ac_mean, bc_mean, bo_mean, meanT_mean, dT_mean, Rg, f_day_clouded, iPm, c, Ct, l, bgm, Bn]), [Ac_interp, Bc_interp, Bo_interp], [Ac_interp1, Bc_interp1, Bo_interp1])
    
    # def biomassinter(self):
    #     return self.calculateBioMassinterNumba(self.cycle_begin, self.cycle_end, self.cycle_len, self.lat, self.lat_t, self.lat_b, 
    #                                            self.shortRad_daily, self.meanT_daily, self.dT_daily, self.LAi, self.legume, self.adaptability)

