"""
PyAEZ
Written by N. Lakmal Deshapriya
"""

import numpy as np
import numba as nb

class BioMassCalc(object):

    def __init__(self, cycle_begin, cycle_end, latitude):
        self.cycle_begin = cycle_begin
        self.cycle_end = cycle_end
        self.cycle_len = cycle_end - cycle_begin + 1

        self.lat = np.abs(latitude)
        self.lat_index1 = np.floor(np.abs(latitude)/10);
        self.lat_index2 = np.ceil(np.abs(latitude)/10);

    def setClimateData(self, min_temp, max_temp, short_rad):

        self.minT_daily = min_temp
        self.maxT_daily = max_temp
        self.shortRad_daily = short_rad

        ## convert radiation values from W/m^2 to cal/cm2/day1
        self.shortRad_daily = ((self.shortRad_daily / 4.19) * (60*60*24)) / (100*100);

        '''Calculate Mean T and dT'''

        ## Calculate Mean Temperature Values
        self.meanT_daily = (self.minT_daily + self.maxT_daily)/2;

        ## Calculate Mean dT Values (day time temperature)
        self.dT_daily = np.zeros(self.minT_daily.shape);

        i_count = 0
        for i1 in range(self.cycle_begin, self.cycle_end+1):

            if i1<30:
                self.dT_daily[i_count] = 0.0278 + 0.3301*self.minT_daily[i_count] + 0.6716*self.maxT_daily[i_count];
            elif i1<30*2:
                self.dT_daily[i_count] = 0.078 + 0.3345*self.minT_daily[i_count] + 0.6672*self.maxT_daily[i_count];
            elif i1<30*3:
                self.dT_daily[i_count] = 0.0770 + 0.3392*self.minT_daily[i_count] + 0.6642*self.maxT_daily[i_count];
            elif i1<30*4:
                self.dT_daily[i_count] = 0.2276 + 0.3466*self.minT_daily[i_count] + 0.6536*self.maxT_daily[i_count];
            elif i1<30*5:
                self.dT_daily[i_count] = 0.2494 + 0.3399*self.minT_daily[i_count] + 0.6576*self.maxT_daily[i_count];
            elif i1<30*6:
                self.dT_daily[i_count] = 0.9955+ 0.3335*self.minT_daily[i_count] + 0.6628*self.maxT_daily[i_count];
            elif i1<30*7:
                self.dT_daily[i_count] = 0.2624+ 0.3351*self.minT_daily[i_count] + 0.659*self.maxT_daily[i_count];
            elif i1<30*8:
                self.dT_daily[i_count] = 0.2860+ 0.3297*self.minT_daily[i_count] + 0.6612*self.maxT_daily[i_count];
            elif i1<30*9:
                self.dT_daily[i_count] = 0.3492+ 0.3410*self.minT_daily[i_count] + 0.6515*self.maxT_daily[i_count];
            elif i1<30*10:
                self.dT_daily[i_count] = 0.1598+ 0.3394*self.minT_daily[i_count] + 0.6591*self.maxT_daily[i_count];
            elif i1<30*11:
                self.dT_daily[i_count] = 0.1156+ 0.3476*self.minT_daily[i_count] + 0.6571*self.maxT_daily[i_count];
            else:
                self.dT_daily[i_count] = -0.0617+ 0.3462*self.minT_daily[i_count] + 0.6649*self.maxT_daily[i_count];

            i_count = i_count + 1

    def setCropParameters(self, LAI, HI, legume, adaptability):
        self.LAi = LAI # leaf area index
        self.HI = HI # harvest index
        self.legume = legume # binary value
        self.adaptability = adaptability-1  #one of [1,2,3,4] classes

    @staticmethod
    @nb.jit(nopython=True)
    def calculateBioMassNumba(cycle_begin, cycle_end, cycle_len, lat, lat_index1, lat_index2, minT_daily, maxT_daily, shortRad_daily, meanT_daily, dT_daily, LAi, HI, legume, adaptability):

        '''Max Radiation'''
        Ac = np.array([[343,360,369,364,349,337,342,357,368,365,349,337],
            [299,332,359,375,377,374,375,377,369,345,311,291],
            [249,293,337,375,394,400,399,386,357,313,264,238],
            [191,245,303,363,400,417,411,384,333,270,210,179],
            [131,190,260,339,396,422,413,369,298,220,151,118],
             [73,131,207,304,380,418,405,344,254,163, 92, 61],
             [22, 72,149,260,356,408,389,309,201,103, 37, 14],
              [0, 20, 89,209,331,408,380,269,142, 45,  2,  0],
              [0,  0, 28,162,334,424,393,248, 81,  3,  0,  0],
              [0,  0,  0,154,339,428,397,252, 40,  0,  0,  0]]);

        '''Biomass in open day'''
        bc = np.array([[413,424,429,426,417,410,413,422,429,427,418,410],
            [376,401,422,437,440,440,440,439,431,411,385,370],
            [334,371,407,439,460,468,465,451,425,387,348,325],
            [281,333,385,437,471,489,483,456,412,356,299,269],
            [218,283,353,427,480,506,497,455,390,314,241,204],
            [147,223,310,409,484,522,509,488,358,260,173,130],
             [66,151,254,383,487,544,523,436,316,195, 94, 49],
              [0, 65,185,350,506,612,575,427,262,114,  7,  0],
              [0,  0, 94,333,571,663,632,474,195, 11,  0,  0],
              [0,  0,  0,371,588, 67,646,497,167,  0,  0,  0]]);

        ''' Biomass in cloudy day '''
        bo = np.array([[219,226,230,228,221,216,218,225,230,228,222,216],
            [197,212,225,234,236,235,236,235,230,218,203,193],
            [170,193,215,235,246,250,249,242,226,203,178,164],
            [137,168,200,232,251,261,258,243,216,182,148,130],
             [99,137,178,223,253,268,263,239,200,155,112, 91],
             [60,100,150,207,251,273,265,230,178,121, 73, 51],
             [19, 60,114,187,245,276,265,216,148, 82, 31, 11],
              [0, 16, 74,158,241,291,273,200,112, 38,  1,  0],
              [0,  0, 24,133,257,318,297,196, 69,  2,  0,  0],
              [0,  0,  0,131,269,319,302,215, 35,  0,  0,  0]]);

        '''Pm values table'''
        PmIndexExtDtTemp = np.array([5.0,10.0,15.0,20.0,25.0,30.0,35.0]);
        PmIndexExt = np.array([[0.0,15.0,20.0,20.0,  15.0, 5.0, 0.0],
            [0.0, 0.0,15.0,32.5,35.0,35.0,35.0],
            [0.0, 0.0, 5.0,45.0,65.0,65.0,65.0],
            [0.0, 5.0,45.0,65.0, 65.0,65.0,65.0]]);

        ''' Calculate average values of Ac, bc, bo, meanT, dT  in the season '''

        doy_middle_of_month = np.arange(0,12)*30 + 15 # Calculate doy of middle of month

        Ac_interp1 = np.interp(np.arange(cycle_begin, cycle_end+1), doy_middle_of_month, Ac[int(lat_index1),:])
        bc_interp1 = np.interp(np.arange(cycle_begin, cycle_end+1), doy_middle_of_month, bc[int(lat_index1),:])
        bo_interp1 = np.interp(np.arange(cycle_begin, cycle_end+1), doy_middle_of_month, bo[int(lat_index1),:])
        Ac_interp2 = np.interp(np.arange(cycle_begin, cycle_end+1), doy_middle_of_month, Ac[int(lat_index2),:])
        bc_interp2 = np.interp(np.arange(cycle_begin, cycle_end+1), doy_middle_of_month, bc[int(lat_index2),:])
        bo_interp2 = np.interp(np.arange(cycle_begin, cycle_end+1), doy_middle_of_month, bo[int(lat_index2),:])

        Ac_interp = Ac_interp1 + (int(lat)-int(lat_index1)*10)*((Ac_interp2-Ac_interp1)/10)
        bc_interp = bc_interp1 + (int(lat)-int(lat_index1)*10)*((bc_interp2-bc_interp1)/10)
        bo_interp = bo_interp1 + (int(lat)-int(lat_index1)*10)*((bo_interp2-bo_interp1)/10)

        Ac_mean = np.mean( Ac_interp );
        bc_mean = np.mean( bc_interp );
        bo_mean = np.mean( bo_interp );

        meanT_mean = np.mean( meanT_daily );
        dT_mean = np.mean( dT_daily );
        shortRad_mean = np.mean( shortRad_daily );

        '''the Fraction of the Daytime the Sky is Clouded'''

        Rg = shortRad_mean;
        f_day_clouded = (Ac_mean - (0.5 * Rg))/(0.8 * (Ac_mean));

        '''Maximum net Rate of CO 2 Exchange of Leaves'''

        PmIndexExt_1Row = PmIndexExt[adaptability,:];
        iPm = np.interp(dT_mean,PmIndexExtDtTemp,PmIndexExt_1Row);

        '''Adjust for Temperature and LAI'''
        # Calculate Ct
        if legume == 1:
            c = 0.0283;
        else:
            c = 0.0108;

        Ct = c*(0.0044 + 0.0019*meanT_mean + 0.0010*np.power(meanT_mean,2));

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

        '''Maximum Rate of Gross Biomass Production'''

        if iPm > 20:
            bgm = f_day_clouded * (.8 + 0.01 * iPm) * bo_mean + (1-f_day_clouded) * (.5 + .025 * iPm) * bc_mean;
        elif iPm < 20:
            bgm = f_day_clouded * (.5 + .025 * iPm) * bo_mean + (1-f_day_clouded) * (.05 * iPm) * bc_mean;
        else:
            bgm = f_day_clouded*bo_mean + (1 - f_day_clouded)*bc_mean;

        '''net biomass production '''

        Bn = (0.36 * bgm * l) / ( (1/cycle_len) + 0.25*Ct );

        return Bn


    def calculateBioMass(self):
        self.Bn = BioMassCalc.calculateBioMassNumba(self.cycle_begin, self.cycle_end, self.cycle_len, self.lat, self.lat_index1, self.lat_index2, self.minT_daily, self.maxT_daily, self.shortRad_daily, self.meanT_daily, self.dT_daily, self.LAi, self.HI, self.legume, self.adaptability)

    def calculateYield(self):
        self.PYield = self.Bn * self.HI;
        return self.PYield
