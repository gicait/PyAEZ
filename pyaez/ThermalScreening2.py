"""
PyAEZ version 2.4 (Dec 2024)
Thermal Screening
2020: N. Lakmal Deshapriya
2022/2023: Swun Wunna Htet
2023 (Dec): Swun Wunna Htet
2024 (Dec): Swun Wunna Htet

Modification:
1. Removing time slicing with start date and cycle length.
2. Removing getSuitability function.
3. TSUM threhold values are rounded as integers for overcoming inconsistent data types.
4. The two different temperature data is used (each with different cycle length slicing).
5. The climate data will only use mean temperature instead of minimum and maximum temperature to reduce
   extra calculation.
6. LGPT screening is now removed because of the new consideration with LGPT and LGP in Module 2.
7. Smoothening is applied to TSUM Screening based on GAEZ routine. However, the investigation with
   residual handling must be asserted.
8. Major code flow revision. Object classes will be used for input data provision. The calculation 
   procedures will be omitted out from object class for possible Numba enhancement (but not every single functions).
"""

import numpy as np
import numba as nb
from numba.typed import List

class ThermalScreening(object):

    def __init__(self):
        self.set_Tsum_screening = False
        self.set_CropSpecificRule = False

    
    def setClimateData(self, meanT_daily_tsum,  meanT_daily_tp, perennial_flag):
        """
        Setting up the Climate Data. Input climate data will be used for calculation of 
        TSUM values and temperature profile calculation.
        
        Parameters
        ----------
        meanT_daily_tsum (1D NumPy Array): crop-cycle specific mean temperature used for
                                           TSUM calculation [Deg Celsius]
        meanT_daily_tp (1D NumPy Array): crop-cycle specific mean temperature used for
                                         temperature profile calculation [Deg Celsius]
        
        Returns
        ------
        None.
        """

        self.tprofile = getTemperatureProfile(meanT_daily_tsum)
        self.tsum0 = getTemperatureSum0(meanT_daily_tp)
        self.perennial_flag = perennial_flag

    def setTSumScreening(self, LnS, LsO, LO, HnS, HsO, HO):
        self.LnS = round(LnS)
        self.LsO = round(LsO)
        self.LO = round(LO)
        self.HnS = round(HnS)
        self.HsO = round(HsO)
        self.HO = round(HO)
        self.set_Tsum_screening = True

    def setCropSpecificRule(self, data, input_temp_profile ,perennial_flag):

        self.calc_CropSpecific_Rule = calculateTemperatureProfileClasses(data, input_temp_profile, perennial_flag)
        self.set_CropSpecificRule = True
    
    def getReductionFactor(self):
        return getReductionFactorNumba(self.set_Tsum_screening, self.LnS, self.LsO, self.LO, self.HnS, self.HsO, self.HO, self.tsum0,
                                       self.set_CropSpecificRule, self.calc_CropSpecific_Rule, self.perennial_flag)


#----------------------------------------------Major Functions End Here----------------------------------------------------------------
#-----------------------------------------Numba Enhanced Functions Starts Here---------------------------------------------------------
# This numba enhancement decorator is omitted out because the function runs faster without nb.jit() (nb.jit takes 2.3 seconds, no decorator produce faster 0.5 seconds)
@nb.jit(nopython = True)
def getReductionFactorNumba(set_Tsum_screening:bool, LnS, LsO, LO, HnS, HsO, HO, tsum0,
                            set_CropSpecificRule:bool, crop_specific_rule_data, perennial_flag):
    
    fc1_final = 1.

    # TSUM screening
    if set_Tsum_screening:
        #Start TSUM screening
        if tsum0 in range(LO, HO):
            f1 = 1.
            fc1_final = min(f1, fc1_final)

        # Within Sub-optimal range (Part 1) (25% reduction factor)
        elif tsum0 in range(LsO, LO):
            f1 = ((tsum0-LsO)/(LO-LsO)) * 0.25 + 0.75
            fc1_final = min(f1, fc1_final)

        # Within Sub-optimal range (Part 2) (25% reduction factor)
        elif tsum0 in range(HO, HsO):
            f1 = ((HsO-tsum0)/(HsO-HO)) * 0.25 + 0.75
            fc1_final = min(f1, fc1_final)

        # Within Marginal range (Part 1) (75% reduction factor)
        elif tsum0 in range(LnS, LsO):
            f1 = ((tsum0-LnS)/(LsO-LnS)) * 0.75
            fc1_final = min(f1, fc1_final)

        # Within Marginal range (Part 2) (75% reduction factor)
        elif tsum0 in range(HsO, HnS):
            f1 = ((HnS-tsum0)/(HnS-HsO)) * 0.75
            fc1_final = min(f1, fc1_final)

        # Within Not suitable range (100% reduction factor)
        elif tsum0 <= LnS or tsum0 >= HnS:
            f1 = 0
            fc1_final = min(f1, fc1_final)


    # Crop Specific Rule Screening (Temperature Profile Screening)
    if set_CropSpecificRule:
        
        specific_data = crop_specific_rule_data
        calc_value = specific_data[0]
        constr_type =  specific_data[1]
        optimal =  specific_data[2]
        sub_optimal =  specific_data[3]
        not_suitable =  specific_data[4]

        # calc_value = specific_data[0]
        # constr_type = specific_data[1]
        # optimal = specific_data[2]
        # sub_optimal  = specific_data[3]
        # not_suitable = specific_data[4]

        # """Loop for each user-specified rule"""
        for i in range(len(calc_value)):

            # "Constraint Rule calculation for greater than"
            if constr_type[i] == '<=' or constr_type[i] == '≤':

                # """Check if all threshold values are the same"""
                if optimal[i] == sub_optimal[i] == not_suitable[i]:

                    # """Calculated value will be compared with optimum threshold"""
                    if calc_value[i] <= optimal[i]:
                        f1 = 1
                        fc1_final = min(f1, fc1_final)

                    else:
                        f1 = 0
                        fc1_final = min(f1, fc1_final)

                elif optimal[i] != sub_optimal[i] == not_suitable[i]:

                    if calc_value[i] <= optimal[i]:
                        f1 = 1
                        fc1_final = min(f1, fc1_final)

                    # """If calculated value within range between optimal and sub-optimum/not-suitable"""
                    elif calc_value[i] > optimal[i] and calc_value[i] <= sub_optimal[i]:
                        f1 = ((calc_value[i] - optimal[i])/(sub_optimal[i] - optimal[i]) * 0.25) + 0.75
                        fc1_final = min(f1, fc1_final)

                    # """For calculated value beyond sub-optimum/not-suitable, use previous linear interpolation (But not sure)"""
                        # elif self.calc_value > self.sub_optimal[i]:
                    else:
                        f1 = 0
                        fc1_final = min(f1, fc1_final)

                # """If all thresholds are different, go linear interpolation to each threshold interval"""
                elif optimal[i] != sub_optimal[i] != not_suitable[i]:

                    if calc_value[i] <= optimal[i]:
                        f1 = 1
                        fc1_final = min(f1, fc1_final)


                    # """If calculated value within range between optimal and sub-optimum/not-suitable"""
                    elif calc_value[i] > optimal[i] and calc_value[i] <= sub_optimal[i]:
                        f1 = ((calc_value[i] - optimal[i])/(sub_optimal[i] - optimal[i]) * 0.25) + 0.75
                        fc1_final = min(f1, fc1_final)


                    # """For calculated value beyond sub-optimum/not-suitable, use previous linear interpolation (But not sure)"""
                    elif calc_value[i] > sub_optimal[i] and calc_value[i] <= not_suitable[i]:
                        f1 = ((calc_value[i] - not_suitable[i])/(sub_optimal[i] - not_suitable[i]) * 0.25) + 0.75
                        fc1_final = min(f1, fc1_final)


                    # """For calculated values beyond not-suitable threshold (not sure)"""
                    elif calc_value[i] > not_suitable[i]:
                        f1 = 0
                        fc1_final = min(f1, fc1_final)



            elif constr_type[i] == '>=' or constr_type[i] == '≥':

                # """Check if all threshold values are the same"""
                if optimal[i] == sub_optimal[i] == not_suitable[i]:

                    # """Calcualted value will be compared with optimum threshold"""
                    if calc_value[i] >= optimal[i]:
                        f1 = 1
                    else:
                        f1 = 0  # (Not sure)
                    fc1_final = min(f1, fc1_final)

                # """If different next checking sub-optimal and not suitable are the same"""
                elif optimal[i] != sub_optimal[i] == not_suitable[i]:

                    if calc_value[i] >= optimal[i]:
                        f1 = 1
                        fc1_final = min(f1, fc1_final)

                    # """If calculated value within range between optimal and sub-optimum/not-suitable"""
                    elif calc_value[i] < optimal[i] and calc_value[i] >= sub_optimal[i]:
                        f1 = ((calc_value[i] - optimal[i])/(sub_optimal[i] - optimal[i]) * 0.25) + 0.75
                        fc1_final = min(f1, fc1_final)

                    # """If calculated value beyond sub-optimum/not-suitable (Not sure)"""
                    elif calc_value[i] < sub_optimal[i]:
                        f1 = 0
                        fc1_final = min(f1, fc1_final)

                # """If all thresholds are different, go linear interpolation to each threshold interval"""
                elif optimal[i] != sub_optimal[i] != not_suitable[i]:

                    # """Calculated value will be compared with optimum threshold"""
                    if calc_value[i] >= optimal[i]:
                        f1 = 1
                        fc1_final = min(f1, fc1_final)

                    # """If calculated value is between optimum and sub-optimum threshold"""
                    elif calc_value[i] < optimal[i] and calc_value[i] >= sub_optimal[i]:
                        f1 = ((calc_value[i] - optimal[i])/(sub_optimal[i] - optimal[i]) * 0.25) + 0.75
                        fc1_final = min(f1, fc1_final)

                    # """If calculated value between sub-optimum and not-suitable threshold"""
                    elif calc_value[i] < sub_optimal[i] and calc_value[i] >= not_suitable[i]:
                        f1 = ((calc_value[i] - not_suitable[i])/(sub_optimal[i] - not_suitable[i]) * 0.25) + 0.75
                        fc1_final = min(f1, fc1_final)

                    # """If calculated value beyond not-suitable threshold (Not sure)"""
                    elif calc_value[i] <= not_suitable[i]:
                        f1 = 0
                        fc1_final = min(f1, fc1_final)

    return fc1_final

# ---------------------------------------- Numba Enhanced Functions End Here---------------------------------- #
# ------------------Intermediate Functions (Not available for Numba enhancement) Starts Here -------------------#

def getTemperatureSum0(temp1D):
    """
    Calculation of temperature summation at zero degree Celsius threshold.
    
    Parameters
    ----------
    temp1D (1-D NumPy Array): Input mean temperature (Deg C)
    
    Returns
    -------
    None.
    """
    interp1D = np.zeros(temp1D.shape)

    # start 5th order polynomical interpolation
    days = np.arange(temp1D.shape[0])
    quadspl = np.poly1d(np.polyfit(days, temp1D, 5))
    interp1D = quadspl(days)

    interp1D[interp1D <= 0] = 0
    return np.round(np.sum(interp1D), decimals=0)


def getTemperatureProfile(temp1D):
    """
    Calculation of temperature profile. The length of temperature data differs depend on 
    crop type (annuals or perennials).
    
    Parameters
    ----------
    temp1 (1-D NumPy Array): Input mean temperature (Deg C)
    
    Returns
    -------
    None.
    """
    # Calculation of Temp Profile for 1-D numpy array of climate data input
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


def insertCropSpecificRuleParameters(data):

    rule = data['Constraint'].to_numpy()
    constr_type = data['Type'].to_numpy()
    optimal = list(data['Optimal'].to_numpy())
    sub_optimal = list(data['Sub-Optimal'].to_numpy())
    not_suitable = list(data['Not-Suitable'].to_numpy())

    return rule, constr_type, optimal, sub_optimal, not_suitable


# 4 Modification
def calculateTemperatureProfileClasses(data, input_temp_profile, perennial_flag):

    Rule_data = insertCropSpecificRuleParameters(data)

    rule = Rule_data[0]

    constr_type = List()
    optimal= List()
    sub_optimal= List()
    not_suitable= List()

    for i in range(len(Rule_data[1])):
        constr_type.append(Rule_data[1][i])
    
    for i in range(len(Rule_data[2])):
        optimal.append(Rule_data[2][i])
    
    for i in range(len(Rule_data[3])):
        sub_optimal.append(Rule_data[3][i])
    
    for i in range(len(Rule_data[4])):
        not_suitable.append(Rule_data[4][i])


    # constr_type = Rule_data[1]
    # optimal = Rule_data[2]
    # sub_optimal = Rule_data[3]
    # not_suitable = Rule_data[4]
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

    calc_value = List()
    for i in range(len(rule)):
        calc_value.append(eval(rule[i]))

    # 'Releasing the memory'
    # if perennial_flag:
    #     del (temp_profile, N1, N2, N3, N4, N5, N6, N7, N8, N9, N1a, N2a, N3a, N4a,
    #             N5a, N6a, N7a, N8a, N9a, N1b, N2b, N3b, N4b, N5b, N6b, N7b, N8b, N9b)
    # else:
    #     del (temp_profile, L1, L2, L3, L4, L5, L6, L7, L8, L9, L1a, L2a, L3a, L4a,
    #             L5a, L6a, L7a, L8a, L9a, L1b, L2b, L3b, L4b, L5b, L6b, L7b, L8b, L9b)
    
    return calc_value, constr_type, optimal, sub_optimal, not_suitable

# ------------------Intermediate Functions (Not available for Numba enhancement) Ends Here -------------------#
