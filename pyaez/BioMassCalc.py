"""
PyAEZ version 2.3 (Dec 2023)
Biomass Calculation

2020: N. Lakmal Deshapriya
2023: Swun Wunna Htet & Kittiphon Boonma
2023 (Dec): Swun Wunna Htet

Modification:

1. Look-up tables of Ac, Bc, Bo and Pm are cross-checked with De Wit.(1965) publication.
2. Revised cycle start, end and cycle length determination.
3. Minor solar radiation conversion factor revision from 4.189 to 4.1868.
4. Day time temperature calculation corrected to respective day of year interval.
5. Numba incorporation are removed due to lack of support for scipy interpolation.
6. Ac, Bc, and Bo reference tables for Southern Hemisphere are updated.
"""

import numpy as np
from scipy.interpolate import CubicSpline

class BioMassCalc(object):

    def __init__(self, cycle_begin, cycle_end, latitude):

        self.cycle_begin = cycle_begin
        self.cycle_end = cycle_end
        self.cycle_len = cycle_end - cycle_begin +1

        self.lat = latitude
        self.lat_index1 = int(np.floor((latitude)/10));
        self.lat_index2 = int(np.ceil((latitude)/10));
        self.lat_t = self.lat_index1 * 10;
        self.lat_b = self.lat_index2 * 10;


    def setClimateData(self, min_temp, max_temp, short_rad):
        """
        Set the climate parameters.
        
        Args:
            min_temp = minimum temperature (Deg C, 1-D NumPy Array)
            max_temp = maximum temperature (Deg C, 1-D NumPy Array)
            short_rad = shortwave radiation (W/m2, 1-D NumPy Array)
            """
        
        self.minT_daily = min_temp
        self.maxT_daily = max_temp
        self.shortRad_daily = short_rad

        # conversion shortwave radiation from W/m2 to cal/cm2/day
        self.shortRad_daily = self.shortRad_daily * 2.06362854686156 # corrected (5/10/2023)

        # calculation of mean temperature and day-time temperature
        self.meanT_daily = (self.minT_daily + self.maxT_daily)/2

        # day-time temperature corrected and calculated according to DOY
        self.dT_daily = np.zeros(self.minT_daily.shape[0])

        for i1 in range(self.cycle_begin, self.cycle_end):

            if i1<=31:# Jan
                self.dT_daily[i1 - self.cycle_begin] = 0.0278 + 0.3301*self.minT_daily[i1 - self.cycle_begin] + 0.6716*self.maxT_daily[i1 - self.cycle_begin];
            elif i1 in range(32, 59+1): # Feb
                self.dT_daily[i1 - self.cycle_begin] = 0.078 + 0.3345*self.minT_daily[i1 - self.cycle_begin] + 0.6672*self.maxT_daily[i1 - self.cycle_begin];
            elif i1 in range(60, 90+1): # Mar
                self.dT_daily[i1 - self.cycle_begin] = 0.0770 + 0.3392*self.minT_daily[i1 - self.cycle_begin] + 0.6642*self.maxT_daily[i1 - self.cycle_begin];
            elif i1 in range(91, 120+1): # Apr
                self.dT_daily[i1 - self.cycle_begin] = 0.2276 + 0.3466*self.minT_daily[i1 - self.cycle_begin] + 0.6536*self.maxT_daily[i1 - self.cycle_begin];
            elif i1 in range(121, 151+1): # May
                self.dT_daily[i1 - self.cycle_begin] = 0.2494 + 0.3399*self.minT_daily[i1 - self.cycle_begin] + 0.6576*self.maxT_daily[i1 - self.cycle_begin];
            elif i1 in range(152, 181+1): # June
                self.dT_daily[i1 - self.cycle_begin] = 0.9955+ 0.3335*self.minT_daily[i1 - self.cycle_begin] + 0.6628*self.maxT_daily[i1 - self.cycle_begin];
            elif i1 in range(182, 212+1): # July
                self.dT_daily[i1 - self.cycle_begin] = 0.2624+ 0.3351*self.minT_daily[i1 - self.cycle_begin] + 0.659*self.maxT_daily[i1 - self.cycle_begin];
            elif i1 in range(213, 243+1): # Aug
                self.dT_daily[i1 - self.cycle_begin] = 0.2860+ 0.3297*self.minT_daily[i1 - self.cycle_begin] + 0.6612*self.maxT_daily[i1 - self.cycle_begin];
            elif i1 in range(244, 273 +1): # Sep
                self.dT_daily[i1 - self.cycle_begin] = 0.3492+ 0.3410*self.minT_daily[i1 - self.cycle_begin] + 0.6515*self.maxT_daily[i1 - self.cycle_begin];
            elif i1 in range(274, 304+1): # Oct
                self.dT_daily[i1 - self.cycle_begin] = 0.1598+ 0.3394*self.minT_daily[i1 - self.cycle_begin] + 0.6591*self.maxT_daily[i1 - self.cycle_begin];
            elif i1 in range(305, 334+1): # Nov
                self.dT_daily[i1 - self.cycle_begin] = 0.1156+ 0.3476*self.minT_daily[i1 - self.cycle_begin] + 0.6571*self.maxT_daily[i1 - self.cycle_begin];
            else: # Dec
                self.dT_daily[i1 - self.cycle_begin] = -0.0617+ 0.3462*self.minT_daily[i1 - self.cycle_begin] + 0.6649*self.maxT_daily[i1 - self.cycle_begin];

    def setCropParameters(self, LAI, HI, legume, adaptability):
        """
        Set the crop-specific parameters for biomass estimation.
        
        Args:
            LAI = leaf area index (float)
            HI = harvest index (float)
            legume = legume crop or not (boolean, 1 = True, 0 = False)
            adaptability = FAO crop adaptability class (int, [1,2,3,4])
            """
        self.LAi = LAI # leaf area index
        self.HI = HI # harvest index
        self.legume = legume # binary value
        self.adaptability = adaptability-1
        
    def calculateBiomassNumba(cycle_begin, cycle_end, cycle_len, lat, lat_index1, lat_index2,
                         lat_t, lat_b, minT_daily, maxT_daily, shortRad_daily, meanT_daily,
                         dT_daily, LAi, HI, legume, adaptability):

        """Calculate the biomass from the settings of all input variables
        """
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

        # Cubin Spline interpolation classes created for Ac, Bc and Bo
        cbl_ac = CubicSpline(new_doy, new_Ac_interp1, extrapolate= True);
        cbl_bc = CubicSpline(new_doy, new_Bc_interp1, extrapolate= True);
        cbl_bo = CubicSpline(new_doy, new_Bo_interp1, extrapolate= True);

        # Cubic Spline interpolation for individual DOY within the cycle length for Ac, Bc and Bo

        Ac_interp = cbl_ac(np.arange(cycle_begin, cycle_end+1), extrapolate= True);
        Bc_interp = cbl_bc(np.arange(cycle_begin, cycle_end+1), extrapolate= True);
        Bo_interp = cbl_bo(np.arange(cycle_begin, cycle_end+1), extrapolate= True);


        Ac_mean = np.mean( Ac_interp );
        bc_mean = np.mean( Bc_interp );
        bo_mean = np.mean( Bo_interp );

        meanT_mean = np.mean( meanT_daily );
        dT_mean = np.mean( dT_daily );
        Rg = np.mean( shortRad_daily );

        '''the Fraction of the Daytime the Sky is Clouded'''
        f_day_clouded = (Ac_mean - (0.5 * Rg))/(0.8 * (Ac_mean));


        '''Maximum net Rate of CO 2 Exchange of Leaves'''

        PmIndexExt_1Row = PmIndexExt[adaptability,:];
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

        # growth rate multiplier
        if(1 <= LAi and LAi < 2):
            l = 0.4;
        elif(LAi < 3):
            l = 0.6;
        elif(LAi < 4):
            l = 0.8;
        elif(LAi < 5):
            l = 0.9;
        elif(5 <= LAi):
            l = 1;
        elif(LAi < 1):
            print("LAI too Low")
            return

        '''Maximum Rate of Gross Biomass Production''' # Minor change

        if iPm > 20:
            bgm = (f_day_clouded * (.8 + (0.01 * iPm)) * bo_mean) + ((1-f_day_clouded) * (.5 + (.025 * iPm)) * bc_mean);
        elif iPm < 20:
            bgm = (f_day_clouded * (.5 + (.025 * iPm)) * bo_mean) + ((1-f_day_clouded) * .05 * iPm * bc_mean);
        # 
        elif iPm == 20:
            bgm = (f_day_clouded*bo_mean) + ((1 - f_day_clouded)*bc_mean);

        '''net biomass production '''
        Bn = (0.36 * bgm * l)/((1/cycle_len)+0.25*Ct)

        return Bn
    
    def calculateBioMass(self):
        self.Bn =  BioMassCalc.calculateBiomassNumba(self.cycle_begin, self.cycle_end, self.cycle_len, self.lat, self.lat_index1, self.lat_index2, self.lat_t, self.lat_b,  self.minT_daily, self.maxT_daily, 
                                               self.shortRad_daily, self.meanT_daily, self.dT_daily, self.LAi, self.HI, self.legume, self.adaptability)

    # 
    def calculateYield(self):
        self.PYield = np.round(self.Bn * self.HI, 0).astype(int);
        return self.PYield
    
    """Developer Note: This below code section is to investigate for intermediate variables used in internal biomass calcualtion. Do not remove this code snippet below"""
    # @staticmethod
    # @nb.jit(nopython=True)
    def calculateBioMassinterNumba(self, cycle_begin, cycle_end, cycle_len, lat, lat_index1, lat_index2, lat_t, lat_b,  minT_daily, maxT_daily, shortRad_daily, meanT_daily, dT_daily, LAi, HI, legume, adaptability):

        """Calculate the biomass from the settings of all input variables
        """
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

        # Cubin Spline interpolation classes created for Ac, Bc and Bo
        cbl_ac = CubicSpline(new_doy, new_Ac_interp1, extrapolate= True);
        cbl_bc = CubicSpline(new_doy, new_Bc_interp1, extrapolate= True);
        cbl_bo = CubicSpline(new_doy, new_Bo_interp1, extrapolate= True);

        # Cubic Spline interpolation for individual DOY within the cycle length for Ac, Bc and Bo

        Ac_interp = cbl_ac(np.arange(cycle_begin, cycle_end+1), extrapolate= True);
        Bc_interp = cbl_bc(np.arange(cycle_begin, cycle_end+1), extrapolate= True);
        Bo_interp = cbl_bo(np.arange(cycle_begin, cycle_end+1), extrapolate= True);


        Ac_mean = np.mean( Ac_interp );
        bc_mean = np.mean( Bc_interp );
        bo_mean = np.mean( Bo_interp );

        meanT_mean = np.mean( meanT_daily );
        dT_mean = np.mean( dT_daily );
        Rg = np.mean( shortRad_daily );

        '''the Fraction of the Daytime the Sky is Clouded'''
        f_day_clouded = (Ac_mean - (0.5 * Rg))/(0.8 * (Ac_mean));


        '''Maximum net Rate of CO 2 Exchange of Leaves'''

        PmIndexExt_1Row = PmIndexExt[adaptability,:];
        iPm = np.interp(dT_mean,PmIndexExtDtTemp,PmIndexExt_1Row);

        '''Adjust for Temperature and LAI'''
        # Calculate Ct (Correct)
        if legume == 1:
            c = 0.0283;
        else:
            c = 0.0108;

        # Fortran => Ct c * (0.044 + 0.0019 * meanT_mean + 0.0010 * np.power(meanT_mean,2)) (Changed)
        Ct = c*(0.0044 + (0.0019*meanT_mean) + (0.0010*np.power(meanT_mean,2)));

        if(1 <= LAi and LAi < 2):
            l = 0.4;
        elif(LAi < 3):
            l = 0.6;
        elif(LAi < 4):
            l = 0.8;
        elif(LAi < 5):
            l = 0.9;
        elif(5 <= LAi):
            l = 1;
        elif(LAi < 1):
            print("LAI too Low")
            return

        '''Maximum Rate of Gross Biomass Production''' # Minor change

        if iPm > 20:
            bgm = (f_day_clouded * (.8 + (0.01 * iPm)) * bo_mean) + ((1-f_day_clouded) * (.5 + (.025 * iPm)) * bc_mean);
        elif iPm < 20:
            bgm = (f_day_clouded * (.5 + (.025 * iPm)) * bo_mean) + ((1-f_day_clouded) * (.05 * iPm) * bc_mean);
        # 
        elif iPm == 20:
            bgm = (f_day_clouded*bo_mean) + ((1 - f_day_clouded)*bc_mean);

        '''net biomass production '''
        Bn = (0.36 * bgm * l)/((1/cycle_len)+0.25*Ct)
        
        return np.array([Ac_mean, bc_mean, bo_mean, meanT_mean, dT_mean, Rg, f_day_clouded, iPm, c, Ct, l, bgm, Bn])
    
    def biomassinter(self):
        return self.calculateBioMassinterNumba(self.cycle_begin, self.cycle_end, self.cycle_len, self.lat, self.lat_index1, self.lat_index2, self.lat_t, self.lat_b,  self.minT_daily, self.maxT_daily, 
                                               self.shortRad_daily, self.meanT_daily, self.dT_daily, self.LAi, self.HI, self.legume, self.adaptability)

        



