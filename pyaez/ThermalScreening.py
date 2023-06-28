"""
PyAEZ version 2.1.0 (June 2023)
ETOCalc.py calculates the reference evapotranspiration from
2020: N. Lakmal Deshapriya
2022/2023: Swun Wunna Htet
"""

import numpy as np


class ThermalScreening(object):

    def __init__(self):
        # self.set_tclimate_screening = False
        self.set_lgpt_screening = False
        self.set_Tsum_screening = False
        self.set_Tprofile_screening = False
        self.set_parameter_adjusted = False  
        self.setTypeBConstraint = False 

    def setparameter(self, cycle_len, start_day):
        self.cycle_len = cycle_len
        self.start_day = start_day


    def setparameteradjusted(self, cycle_len_rain, cycle_len_irri, Start_day):
        self.cycle_len_r = cycle_len_rain
        self.cycle_len_i = cycle_len_irri
        self.start_day = Start_day
        self.set_parameter_adjusted = True


    def setClimateData(self, minT_daily, maxT_daily):
        # self.cycle_len=cycle_len
        self.meanT_daily = (minT_daily + maxT_daily) / 2
        self.lgp0 = self.getThermalLGP0()
        self.lgp5 = self.getThermalLGP5()
        self.lgp10 = self.getThermalLGP10()

        if self.set_parameter_adjusted:
            self.tprofile_rain = self.getTemperatureProfile(
                start_day=self.start_day, cycle_len=self.cycle_len_r)
            self.tprofile_irr = self.getTemperatureProfile(
                start_day=self.start_day, cycle_len=self.cycle_len_i)
        else:
            self.tprofile = self.getTemperatureProfile(
                start_day=self.start_day, cycle_len=self.cycle_len)
        # self.tsum0 = self.getTemperatureSum0(self.cycle_len)
        if self.set_parameter_adjusted:  # if the perennial crop is adjusted we must check for both condition
            self.tsum0_r = self.getTemperatureSum0(self.cycle_len_r)
            self.tsum0_i = self.getTemperatureSum0(self.cycle_len_i)
        else:
            self.tsum0 = self.getTemperatureSum0(self.cycle_len)

    """ Calculation of Module I indicators"""
    def getThermalLGP0(self):
        return np.sum(self.meanT_daily > 0)

    def getThermalLGP5(self):
        return np.sum(self.meanT_daily > 5)

    def getThermalLGP10(self):
        return np.sum(self.meanT_daily > 10)

    def getTemperatureSum0(self, cycle_len):
        tempT = self.meanT_daily[self.start_day-1: self.start_day-1+cycle_len]
        tempT[tempT <= 0] = 0
        return np.round(np.sum(tempT), decimals=0)

    def getTemperatureSum5(self):
        tempT = self.meanT_daily
        tempT[tempT <= 5] = 0
        return np.round(np.sum(tempT), decimals=0)

    def getTemperatureSum10(self):
        tempT = self.meanT_daily
        tempT[tempT <= 10] = 0
        return np.round(np.sum(tempT), decimals=0)


    def getTemperatureProfile(self, start_day, cycle_len):
        # Calculation of Temp Profile for 1-D numpy array of climate data input
        temp1D = self.meanT_daily.copy()
        temp1D = temp1D[start_day-1: start_day-1 + cycle_len]

        interp1D = np.zeros(temp1D.shape)

        # start 5th order polynomical interpolation
        days = np.arange(temp1D.shape[0])

        quadspl = np.poly1d(np.polyfit(days, temp1D, 5))

        interp1D = quadspl(days)

        # adjustment for differences between front day and back day
        meanT_daily_add1day = np.concatenate((interp1D, interp1D[0:1]))
        meanT_first = meanT_daily_add1day[:-1]
        meanT_diff = meanT_daily_add1day[1:] - meanT_daily_add1day[:-1]

        A9 = np.sum(np.logical_and(meanT_diff > 0, meanT_first < -5))
        A8 = np.sum(np.logical_and(meanT_diff > 0, np.logical_and(
            meanT_first >= -5, meanT_first < 0)))
        A7 = np.sum(np.logical_and(meanT_diff > 0, np.logical_and(
            meanT_first >= 0, meanT_first < 5)))
        A6 = np.sum(np.logical_and(meanT_diff > 0, np.logical_and(
            meanT_first >= 5, meanT_first < 10)))
        A5 = np.sum(np.logical_and(meanT_diff > 0, np.logical_and(
            meanT_first >= 10, meanT_first < 15)))
        A4 = np.sum(np.logical_and(meanT_diff > 0, np.logical_and(
            meanT_first >= 15, meanT_first < 20)))
        A3 = np.sum(np.logical_and(meanT_diff > 0, np.logical_and(
            meanT_first >= 20, meanT_first < 25)))
        A2 = np.sum(np.logical_and(meanT_diff > 0, np.logical_and(
            meanT_first >= 25, meanT_first < 30)))
        A1 = np.sum(np.logical_and(meanT_diff > 0, meanT_first >= 30))

        B9 = np.sum(np.logical_and(meanT_diff < 0, meanT_first < -5))
        B8 = np.sum(np.logical_and(meanT_diff < 0, np.logical_and(
            meanT_first >= -5, meanT_first < 0)))
        B7 = np.sum(np.logical_and(meanT_diff < 0, np.logical_and(
            meanT_first >= 0, meanT_first < 5)))
        B6 = np.sum(np.logical_and(meanT_diff < 0, np.logical_and(
            meanT_first >= 5, meanT_first < 10)))
        B5 = np.sum(np.logical_and(meanT_diff < 0, np.logical_and(
            meanT_first >= 10, meanT_first < 15)))
        B4 = np.sum(np.logical_and(meanT_diff < 0, np.logical_and(
            meanT_first >= 15, meanT_first < 20)))
        B3 = np.sum(np.logical_and(meanT_diff < 0, np.logical_and(
            meanT_first >= 20, meanT_first < 25)))
        B2 = np.sum(np.logical_and(meanT_diff < 0, np.logical_and(
            meanT_first >= 25, meanT_first < 30)))
        B1 = np.sum(np.logical_and(meanT_diff < 0, meanT_first >= 30))

        # releasing memory
        del (temp1D, interp1D, days, quadspl,
             meanT_daily_add1day, meanT_first, meanT_diff)

        return [A1, A2, A3, A4, A5, A6, A7, A8, A9, B1, B2, B3, B4, B5, B6, B7, B8, B9]

    """Thermal Screening Flags"""
    # def setThermalClimateScreening(self, t_climate, no_t_climate):
    #     self.t_climate = t_climate
    #     self.no_t_climate = no_t_climate # list of unsuitable thermal climate
    #     self.set_tclimate_screening = True

    def setLGPTScreening(self, no_lgpt, optm_lgpt):
        self.no_lgpt = no_lgpt
        self.optm_lgpt = optm_lgpt

        self.set_lgpt_screening = True

    # 2. Modification (SWH)
    def setTSumScreening(self, LnS, LsO, LO, HnS, HsO, HO):
        self.LnS = LnS
        self.LsO = LsO
        self.LO = LO
        self.HnS = HnS
        self.HsO = HsO
        self.HO = HO
        self.set_Tsum_screening = True

    def setTProfileScreening(self, no_Tprofile, optm_Tprofile):
        self.no_Tprofile = no_Tprofile
        self.optm_Tprofile = optm_Tprofile
        self.set_Tprofile_screening = True

    # 4 Modification
    def applyTypeBConstraint(self, data, input_temp_profile, perennial_flag=False):

        self.rule = data['Constraint'].to_numpy()
        self.constr_type = data['Type'].to_numpy()
        self.optimal = data['Optimal'].to_numpy()
        self.sub_optimal = data['Sub-Optimal'].to_numpy()
        self.not_suitable = data['Not-Suitable'].to_numpy()

        temp_profile = input_temp_profile
        """For Perennials"""
        if perennial_flag:
            N1a = temp_profile[0]
            N2a = temp_profile[1]
            N3a = temp_profile[2]
            N4a = temp_profile[3]
            N5a = temp_profile[4]
            N6a = temp_profile[5]
            N7a = temp_profile[6]
            N8a = temp_profile[7]
            N9a = temp_profile[8]
            N1b = temp_profile[9]
            N2b = temp_profile[10]
            N3b = temp_profile[11]
            N4b = temp_profile[12]
            N5b = temp_profile[13]
            N6b = temp_profile[14]
            N7b = temp_profile[15]
            N8b = temp_profile[16]
            N9b = temp_profile[17]

            N1 = N1a + N1b
            N2 = N2a + N2b
            N3 = N3a + N3b
            N4 = N4a + N4b
            N5 = N5a + N5b
            N6 = N6a + N6b
            N7 = N7a + N7b
            N8 = N8a + N8b
            N9 = N9a + N9b

        else:
            """For non-perennials"""
            L1a = temp_profile[0]
            L2a = temp_profile[1]
            L3a = temp_profile[2]
            L4a = temp_profile[3]
            L5a = temp_profile[4]
            L6a = temp_profile[5]
            L7a = temp_profile[6]
            L8a = temp_profile[7]
            L9a = temp_profile[8]
            L1b = temp_profile[9]
            L2b = temp_profile[10]
            L3b = temp_profile[11]
            L4b = temp_profile[12]
            L5b = temp_profile[13]
            L6b = temp_profile[14]
            L7b = temp_profile[15]
            L8b = temp_profile[16]
            L9b = temp_profile[17]

            L1 = L1a + L1b
            L2 = L2a + L2b
            L3 = L3a + L3b
            L4 = L4a + L4b
            L5 = L5a + L5b
            L6 = L6a + L6b
            L7 = L7a + L7b
            L8 = L8a + L8b
            L9 = L9a + L9b

        self.calc_value = []
        for i in range(len(self.rule)):
            self.calc_value.append(eval(self.rule[i]))

        self.setTypeBConstraint = True

        'Releasing the memory'
        if perennial_flag:
            del (temp_profile, N1, N2, N3, N4, N5, N6, N7, N8, N9, N1a, N2a, N3a, N4a,
                 N5a, N6a, N7a, N8a, N9a, N1b, N2b, N3b, N4b, N5b, N6b, N7b, N8b, N9b)
        else:
            del (temp_profile, L1, L2, L3, L4, L5, L6, L7, L8, L9, L1a, L2a, L3a, L4a,
                 L5a, L6a, L7a, L8a, L9a, L1b, L2b, L3b, L4b, L5b, L6b, L7b, L8b, L9b)

    """Getting Suitability and Reduction factors"""

    def getSuitability(self):

        # Thermal Climate Screening
        # if self.set_tclimate_screening:
        #     if self.t_climate in self.no_t_climate:
        #         return False
        # output = True
        # result = []

        # LGPT screening
        if self.set_lgpt_screening:
            if self.lgp0 <= self.no_lgpt[0] or self.lgp5 <= self.no_lgpt[1] or self.lgp10 <= self.no_lgpt[2]:
                return False

        # TSUM Screening
        if self.set_Tsum_screening:

            if self.set_parameter_adjusted:
                if np.logical_or([self.tsum0_r > self.HnS or self.tsum0_r < self.LnS], [self.tsum0_i > self.HnS or self.tsum0_i < self.LnS]):
                    return False

            else:
                if (self.tsum0 > self.HnS or self.tsum0 < self.LnS):
                    return False

        # Temperature Profile Screening
        if self.set_Tprofile_screening:
            for i1 in range(len(self.tprofile)):
                if self.tprofile[i1] <= self.no_Tprofile[i1]:
                    return False

        # 4 (Modification) Type B Constraint Screening
        if self.setTypeBConstraint:
            for i in range(len(self.calc_value)):

                if self.constr_type[i] == '=':
                    if self.calc_value[i] != self.optimal[i]:
                        return False

                if self.constr_type[i] == '<=':
                    if self.calc_value[i] > self.optimal[i] or self.calc_value[i] > self.not_suitable[i]:
                        return False

                if self.constr_type[i] == '>=':
                    if self.calc_value[i] < self.optimal[i] or self.calc_value[i] < self.not_suitable[i]:
                        return False

        return True

    # 3 Modification
    # def setRHandDT(self, rel_humidity, min_temp_daily, max_temp_daily):

    #     self.RH_daily = rel_humidity
    #     self.min_temp = min_temp_daily
    #     self.max_temp = max_temp_daily
    #     self.RH_avg = np.mean(self.RH)
    #     self.DTRavg = np.mean(self.max_temp) - np.mean(self.min_temp) # Diurnal temperature range

    # Old function

    def getReductionFactor(self, rain_or_irr):

        thermal_screening_f = 1

        # Thermal Climate Screening
        # if self.set_tclimate_screening:
        #     if self.t_climate in self.no_t_climate:
        #         thermal_screening_f = 0

        # LGPT Screening logics
        if self.set_lgpt_screening:

            if self.lgp0 < self.optm_lgpt[0]:
                f1 = (
                    (self.lgp0-self.no_lgpt[0])/(self.optm_lgpt[0]-self.no_lgpt[0])) * 0.75 + 0.25
                thermal_screening_f = np.min([f1, thermal_screening_f])

            if self.lgp5 < self.optm_lgpt[1]:
                f1 = (
                    (self.lgp5-self.no_lgpt[1])/(self.optm_lgpt[1]-self.no_lgpt[1])) * 0.75 + 0.25
                thermal_screening_f = np.min([f1, thermal_screening_f])

            if self.lgp10 < self.optm_lgpt[2]:
                f1 = (
                    (self.lgp10-self.no_lgpt[2])/(self.optm_lgpt[2]-self.no_lgpt[2])) * 0.75 + 0.25
                thermal_screening_f = np.min([f1, thermal_screening_f])

        # 2. Modification (SWH)
        # TSUM Screening
        if self.set_Tsum_screening:

            """ If it's perennial, and parameters are adjusted, TSUM will be different for both irrigated and rainfed conditions
            because the efficient cycle lengths are different. Thus, for TSUM0 calculation, however this has to be clarified."""

            if self.set_parameter_adjusted:

                if rain_or_irr == 'rain':
                    tsum0 = self.tsum0_r

                elif rain_or_irr == 'irr':
                    tsum0 = self.tsum_r

                # Start TSUM screening
                if tsum0 in range(self.LO, self.HO):
                    f1 = 1
                    thermal_screening_f = np.min([f1, thermal_screening_f])

                # Within Sub-optimal range (Part 1) (25% reduction factor)
                elif tsum0 in range(self.LsO, self.LO):
                    f1 = ((tsum0-self.LsO)/(self.LO-self.LsO)) * 0.25 + 0.75
                    thermal_screening_f = np.min([f1, thermal_screening_f])

                # Within Sub-optimal range (Part 2) (25% reduction factor)
                elif tsum0 in range(self.HO, self.HsO):
                    f1 = ((self.HsO-tsum0)/(self.HsO-self.HO)) * 0.25 + 0.75
                    thermal_screening_f = np.min([f1, thermal_screening_f])

                # Within Marginal range (Part 1) (75% reduction factor)
                elif tsum0 in range(self.LnS, self.LsO):
                    f1 = ((tsum0-self.LnS)/(self.LsO-self.LnS)) * 0.75
                    thermal_screening_f = np.min([f1, thermal_screening_f])

                # Within Marginal range (Part 2) (75% reduction factor)
                elif tsum0 in range(self.HsO, self.HnS):
                    f1 = ((self.HnS-tsum0)/(self.HnS-self.HsO)) * 0.75
                    thermal_screening_f = np.min([f1, thermal_screening_f])

                # Within Not suitable range (100% reduction factor)
                elif tsum0 <= self.LnS or tsum0 >= self.HnS:
                    f1 = 0
                    thermal_screening_f = np.min([f1, thermal_screening_f])

            else:
                # Within Optimal range (no reduction in factor)
                if self.tsum0 in range(self.LO, self.HO):
                     f1 = 1
                     thermal_screening_f = np.min([f1, thermal_screening_f])

                # Within Sub-optimal range (Part 1) (25% reduction factor)
                elif self.tsum0 in range(self.LsO, self.LO):
                     f1 = ((self.tsum0-self.LsO)/ \
                           (self.LO-self.LsO)) * 0.25 + 0.75
                     thermal_screening_f = np.min([f1, thermal_screening_f])

                # Within Sub-optimal range (Part 2) (25% reduction factor)
                elif self.tsum0 in range(self.HO, self.HsO):
                     f1 = ((self.HsO-self.tsum0)/ \
                           (self.HsO-self.HO)) * 0.25 + 0.75
                     thermal_screening_f = np.min([f1, thermal_screening_f])

                # Within Marginal range (Part 1) (75% reduction factor)
                elif self.tsum0 in range(self.LnS, self.LsO):
                     f1 = ((self.tsum0-self.LnS)/(self.LsO-self.LnS)) * 0.75
                     thermal_screening_f = np.min([f1, thermal_screening_f])

                # Within Marginal range (Part 2) (75% reduction factor)
                elif self.tsum0 in range(self.HsO, self.HnS):
                     f1 = ((self.HnS-self.tsum0)/(self.HnS-self.HsO)) * 0.75
                     thermal_screening_f = np.min([f1, thermal_screening_f])

                # Within Not suitable range (100% reduction factor)
                elif self.tsum0 <= self.LnS or self.tsum0 >= self.HnS:
                     f1 = 0
                     thermal_screening_f = np.min([f1, thermal_screening_f])   

        if self.setTypeBConstraint:

            """Loop for each user-specified rule"""
            for i in range(len(self.calc_value)):

                """If calculated rule is greater than optimal threshold, fc is set to 1"""

                if self.constr_type[i] == '<=':

                    """If """
                    if self.optimal[i] != self.sub_optimal[i] != self.not_suitable[i]:

                        if self.calc_value[i] >= self.optimal[i] and self.calc_value[i] <= self.sub_optimal[i]:
                            f1 = ((self.calc_value[i] - self.optimal[i])/(
                                self.sub_optimal[i] - self.optimal[i]) * 0.25) + 0.75

                        elif self.calc_value[i] >= self.sub_optimal[i] and self.calc_value[i] <= self.not_suitable[i]:
                            f1 = ((self.calc_value[i] - self.sub_optimal[i])/(self.not_suitable[i] - self.sub_optimal[i]) * 0.25 ) + 0.75

                    elif self.optimal[i] != self.sub_optimal[i] and self.sub_optimal[i] == self.not_suitable[i]:
                        f1 = ((self.calc_value[i] - self.optimal[i])/ \
                              (self.sub_optimal[i] - self.optimal[i]) * 0.25) + 0.75

                elif self.constr_type[i] == '>=':

                    if self.optimal[i] != self.sub_optimal[i] != self.not_suitable[i]:

                        if self.calc_value[i] <= self.optimal[i] and self.calc_value[i] >= self.sub_optimal[i]:
                            f1 = ((self.calc_value[i] - self.sub_optimal[i])/(
                                self.optimal[i] - self.sub_optimal[i]) * 0.25) + 0.75

                        elif self.calc_value[i] <= self.sub_optimal[i] and self.calc_value[i] >= self.not_suitable[i]:
                            f1 = ((self.calc_value[i] - self.not_suitable[i])/(
                                self.sub_optimal[i] - self.not_suitable[i]) * 0.25) + 0.75

                    elif self.optimal[i] != self.sub_optimal[i] and self.sub_optimal[i] == self.not_suitable[i]:
                        f1 = ((self.calc_value[i] - self.optimal[i])/ \
                              (self.sub_optimal[i] - self.optimal[i]) * 0.25) + 0.75

                thermal_screening_f = np.min([f1, thermal_screening_f])
                # print('Thermal Screening factor = ', thermal_screening_f)

        return thermal_screening_f


    def getReductionFactor2(self, rain_or_irr):

        thermal_screening_f = 1

        # LGPT Screening logics
        if self.set_lgpt_screening:

            if self.lgp0 < self.optm_lgpt[0]:
                f1 = (
                    (self.lgp0-self.no_lgpt[0])/(self.optm_lgpt[0]-self.no_lgpt[0])) * 0.75 + 0.25
                thermal_screening_f = np.min([f1, thermal_screening_f])

            if self.lgp5 < self.optm_lgpt[1]:
                f1 = (
                    (self.lgp5-self.no_lgpt[1])/(self.optm_lgpt[1]-self.no_lgpt[1])) * 0.75 + 0.25
                thermal_screening_f = np.min([f1, thermal_screening_f])

            if self.lgp10 < self.optm_lgpt[2]:
                f1 = (
                    (self.lgp10-self.no_lgpt[2])/(self.optm_lgpt[2]-self.no_lgpt[2])) * 0.75 + 0.25
                thermal_screening_f = np.min([f1, thermal_screening_f])

        # TSUM Screening
        if self.set_Tsum_screening:

            """ If it's perennial, and parameters are adjusted, TSUM will be different for both irrigated and rainfed conditions 
            because the efficient cycle lengths are different. Thus, for TSUM0 calculation, however this has to be clarified."""

            # TSUM Screening
            if self.set_Tsum_screening:

                """ If it's perennial, and parameters are adjusted, TSUM will be different for both irrigated and rainfed conditions 
                because the efficient cycle lengths are different. Thus, for TSUM0 calculation, however this has to be clarified."""

                if self.set_parameter_adjusted:

                    if rain_or_irr == 'rain':
                        tsum0 = self.tsum0_r

                    elif rain_or_irr == 'irr':
                        tsum0 = self.tsum_r

                    # Start TSUM screening
                    if tsum0 in range(self.LO, self.HO):
                        f1 = 1
                        thermal_screening_f = np.min([f1, thermal_screening_f])

                    # Within Sub-optimal range (Part 1) (25% reduction factor)
                    elif tsum0 in range(self.LsO, self.LO):
                        f1 = ((tsum0-self.LsO)/(self.LO-self.LsO)) * 0.25 + 0.75
                        thermal_screening_f = np.min([f1, thermal_screening_f])

                    # Within Sub-optimal range (Part 2) (25% reduction factor)
                    elif tsum0 in range(self.HO, self.HsO):
                        f1 = ((self.HsO-tsum0)/(self.HsO-self.HO)) * 0.25 + 0.75
                        thermal_screening_f = np.min([f1, thermal_screening_f])

                    # Within Marginal range (Part 1) (75% reduction factor)
                    elif tsum0 in range(self.LnS, self.LsO):
                        f1 = ((tsum0-self.LnS)/(self.LsO-self.LnS)) * 0.75
                        thermal_screening_f = np.min([f1, thermal_screening_f])

                    # Within Marginal range (Part 2) (75% reduction factor)
                    elif tsum0 in range(self.HsO, self.HnS):
                        f1 = ((self.HnS-tsum0)/(self.HnS-self.HsO)) * 0.75
                        thermal_screening_f = np.min([f1, thermal_screening_f])

                    # Within Not suitable range (100% reduction factor)
                    elif tsum0 <= self.LnS or tsum0 >= self.HnS:
                        f1 = 0
                        thermal_screening_f = np.min([f1, thermal_screening_f])    

                else:
                    # Within Optimal range (no reduction in factor)
                    if self.tsum0 in range(self.LO, self.HO):
                         f1 = 1
                         thermal_screening_f = np.min([f1, thermal_screening_f])

                    # Within Sub-optimal range (Part 1) (25% reduction factor)
                    elif self.tsum0 in range(self.LsO, self.LO):
                         f1 = ((self.tsum0-self.LsO)/ \
                               (self.LO-self.LsO)) * 0.25 + 0.75
                         thermal_screening_f = np.min([f1, thermal_screening_f])

                    # Within Sub-optimal range (Part 2) (25% reduction factor)
                    elif self.tsum0 in range(self.HO, self.HsO):
                        f1 = ((self.HsO-self.tsum0)/ \
                            (self.HsO-self.HO)) * 0.25 + 0.75
                        thermal_screening_f = np.min([f1, thermal_screening_f])

                    # Within Marginal range (Part 1) (75% reduction factor)
                    elif self.tsum0 in range(self.LnS, self.LsO):
                        f1 = ((self.tsum0-self.LnS)/ \
                            (self.LsO-self.LnS)) * 0.75
                        thermal_screening_f = np.min([f1, thermal_screening_f])

                    # Within Marginal range (Part 2) (75% reduction factor)
                    elif self.tsum0 in range(self.HsO, self.HnS):
                        f1 = ((self.HnS-self.tsum0)/(self.HnS-self.HsO)) * 0.75
                        thermal_screening_f = np.min([f1, thermal_screening_f])

                    # Within Not suitable range (100% reduction factor)
                    elif self.tsum0 <= self.LnS or self.tsum0 >= self.HnS:
                        f1 = 0
                        thermal_screening_f = np.min([f1, thermal_screening_f])   


        if self.setTypeBConstraint:

            # """Loop for each user-specified rule"""
            for i in range(len(self.calc_value)):

                # "Constraint Rule calculation for greater than"
                if self.constr_type[i] == '<=':

                    # """Check if all threshold values are the same"""
                    if self.optimal[i] == self.sub_optimal[i] == self.not_suitable[i]:

                        # """Calculated value will be compared with optimum threshold"""

                        if self.calc_value[i] <= self.optimal[i]:
                            f1 = 1
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])

                        else:
                            f1 = 0
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])

                    elif self.optimal[i] != self.sub_optimal[i] == self.not_suitable[i]:

                        if self.calc_value[i] <= self.optimal[i]:
                            f1 = 1
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])

                    # """If calculated value within range between optimal and sub-optimum/not-suitable"""
                        elif self.calc_value[i] > self.optimal[i] and self.calc_value[i] <= self.sub_optimal[i]:
                            f1 = ((self.calc_value[i] - self.optimal[i])/(
                                self.sub_optimal[i] - self.optimal[i]) * 0.25) + 0.75
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])

                    # """For calculated value beyond sub-optimum/not-suitable, use previous linear interpolation (But not sure)"""
                        elif self.calc_value > self.sub_optimal[i]:
                            f1 = 0
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])
                        # Wrong, must be zero

                    # """If all thresholds are different, go linear interpolation to each threshold interval"""

                    elif self.optimal[i] != self.sub_optimal[i] != self.not_suitable[i]:

                        if self.calc_value[i] <= self.optimal[i]:
                            f1 = 1
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])

                    # """If calculated value within range between optimal and sub-optimum/not-suitable"""
                        elif self.calc_value[i] > self.optimal[i] and self.calc_value[i] <= self.sub_optimal[i]:
                            f1 = ((self.calc_value[i] - self.optimal[i])/(
                                self.sub_optimal[i] - self.optimal[i]) * 0.25) + 0.75
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])

                    # """For calculated value beyond sub-optimum/not-suitable, use previous linear interpolation (But not sure)"""
                        elif self.calc_value[i] > self.sub_optimal[i] and self.calc_value[i] <= self.not_suitable[i]:
                            f1 = ((self.calc_value[i] - self.not_suitable[i])/(
                                self.sub_optimal[i] - self.not_suitable[i]) * 0.25) + 0.75
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])

                    # """For calculated values beyond not-suitable threshold (not sure)"""
                        elif self.calc_value[i] > self.not_suitable[i]:
                            f1 = 0
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])


                elif self.constr_type[i] == '>=':

                    # """Check if all threshold values are the same"""
                    if self.optimal[i] == self.sub_optimal[i] == self.not_suitable[i]:

                        # """Calcualted value will be compared with optimum threshold"""
                        if self.calc_value[i] >= self.optimal[i]:
                            f1 = 1
                        else:
                            f1 = 0  # (Not sure)
                        thermal_screening_f = np.min([f1, thermal_screening_f])

                    # """If different next checking sub-optimal and not suitable are the same"""
                    elif self.optimal[i] != self.sub_optimal[i] == self.not_suitable[i]:

                        if self.calc_value[i] >= self.optimal[i]:
                            f1 = 1
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])

                    # """If calculated value within range between optimal and sub-optimum/not-suitable"""
                        elif self.calc_value[i] < self.optimal[i] and self.calc_value[i] >= self.sub_optimal[i]:
                            f1 = ((self.calc_value[i] - self.optimal[i])/(
                                self.sub_optimal[i] - self.optimal[i]) * 0.25) + 0.75
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])

                    # """If calculated value beyond sub-optimum/not-suitable (Not sure)"""
                        elif self.calc_value[i] < self.sub_optimal[i]:
                            f1 = 0
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])

                    # """If all thresholds are different, go linear interpolation to each threshold interval"""
                    elif self.optimal[i] != self.sub_optimal[i] != self.not_suitable[i]:

                        # """Calculated value will be compared with optimum threshold"""
                        if self.calc_value[i] >= self.optimal[i]:
                            f1 = 1
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])

                    # """If calculated value is between optimum and sub-optimum threshold"""
                        elif self.calc_value[i] < self.optimal[i] and self.calc_value[i] >= self.sub_optimal[i]:
                            f1 = ((self.calc_value[i] - self.optimal[i])/(
                                self.sub_optimal[i] - self.optimal[i]) * 0.25) + 0.75
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])

                    # """If calculated value between sub-optimum and not-suitable threshold"""
                        elif self.calc_value[i] < self.sub_optimal[i] and self.calc_value >= self.not_suitable[i]:
                            f1 = ((self.calc_value[i] - self.not_suitable[i])/(
                                self.sub_optimal[i] - self.not_suitable[i]) * 0.25) + 0.75
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])

                    # """If calculated value beyond not-suitable threshold (Not sure)"""
                        elif self.calc_value[i] <= self.not_suitable[i]:
                            f1 = 0
                            thermal_screening_f = np.min(
                                [f1, thermal_screening_f])

        return thermal_screening_f
