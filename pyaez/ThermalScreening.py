"""
PyAEZ version 2.4 (Dec 2024)
Thermal Screening
2020: N. Lakmal Deshapriya
2022/2023: Swun Wunna Htet
2023 (Dec): Swun Wunna Htet
2024 (Dec): Swun Wunna Htet

Modification:
1. LGPT screening is now removed because of the new consideration with LGPT and LGP in Module 2.
2. Smoothening is applied to TSUM Screening based on GAEZ routine. However, the investigation with
   residual handling must be asserted.
3. Object classes will be used for input data provision. The calculation 
   procedures will be omitted out from object class for possible Numba enhancement (but not every single functions).
"""

import numpy as np
import numba as nb
from numba.typed import List
from scipy.interpolate import interp1d


#----------------------------------------------Major Functions Starts Here----------------------------------------------------------------
# This numba enhancement decorator is omitted out because the function runs faster without nb.jit() (nb.jit takes 2.3 seconds, no decorator produce faster 0.5 seconds)
# @nb.jit(nopython = True)
# perennial flag in function removed as it is not applied.
def getReductionFactorNumba(set_Tsum_screening:bool, LnS, LsO, LO, HnS, HsO, HO, tsum0,
                            set_CropSpecificRule:bool, crop_specific_rule_data, hibernating_flag, vern_factor):
    """Calculation of LUT specific thermal suitability factor (fc1)
    
    Args:
        set_Tsum_screening (Bool): TSUM screening activation
        LnS (int): Lower boundary of not-suitable accumulated heat unit range.
        LsO (int): Lower boundary of sub-optimal accumulated heat unit range.
        LO (int): Lower boundary of optimal accumulated heat unit range.
        HnS (int):Upper boundary of not-suitable accumulated heat range.
        HsO (int): Upper boundary of sub-optimal accumulated heat range.
        HO (int): Upper boundary of not-suitable accumulated heat range.
        tsum0 (float): Temperature summation at 0 Deg Threshold
        set_CropSpecificRule (bool): Crop-specific Rule activation
        crop_specific_rule_data (list): calculated crop-specific rule constraints
        hibernating_flag (bool): Hibernation activation
        vern_factor (float): calculated vernalization factor.
        
    Return:
        fc1 (float): therml suitability factor, value between 0 (not suitable) and 1  (suitable)
    """
    
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

    # Vernalization Screening
    if hibernating_flag:
        fc1_final = min(vern_factor, fc1_final)

    # Crop Specific Rule Screening (Temperature Profile Screening)
    if set_CropSpecificRule:
        
        specific_data = crop_specific_rule_data
        calc_value = specific_data[0]
        constr_type =  specific_data[1]
        optimal =  specific_data[2]
        sub_optimal =  specific_data[3]
        not_suitable =  specific_data[4]

        for i in range(len(calc_value)):
            # """Check if all threshold values are the same"""
            if constr_type[i] == '<=' or constr_type[i] == '≤':
                # """Calculated value will be compared with optimum threshold"""
                if optimal[i] == sub_optimal[i] == not_suitable[i]:
                    if calc_value[i] <= optimal[i]:
                        f1 = vern_factor if hibernating_flag else 1
                    else:
                        f1 = 0
                    fc1_final = min(f1, fc1_final)
                
                elif optimal[i] != sub_optimal[i] == not_suitable[i]:
                    if calc_value[i] <= optimal[i]:
                        f1 = vern_factor if hibernating_flag else 1
                    # """If calculated value within range between optimal and sub-optimum/not-suitable"""
                    elif calc_value[i] > optimal[i] and calc_value[i] <= sub_optimal[i]:
                        f1 = vern_factor if hibernating_flag else ((calc_value[i] - optimal[i])/(sub_optimal[i] - optimal[i]) * 0.25) + 0.75
                    else:
                        f1 = 0
                    fc1_final = min(f1, fc1_final)

                 # """If all thresholds are different, go linear interpolation to each threshold interval"""
                elif optimal[i] != sub_optimal[i] != not_suitable[i]:
                    if calc_value[i] <= optimal[i]:
                        f1 = vern_factor if hibernating_flag else 1
                    # """If calculated value within range between optimal and sub-optimum/not-suitable"""
                    elif calc_value[i] > optimal[i] and calc_value[i] <= sub_optimal[i]:
                        f1 = ((calc_value[i] - optimal[i])/(sub_optimal[i] - optimal[i]) * 0.25) + 0.75
                    # """For calculated value beyond sub-optimum/not-suitable, use previous linear interpolation (But not sure)"""
                    elif calc_value[i] > sub_optimal[i] and calc_value[i] <= not_suitable[i]:
                        f1 = ((calc_value[i] - not_suitable[i])/(sub_optimal[i] - not_suitable[i]) * 0.25) + 0.75
                    else:
                        f1 = 0
                    fc1_final = min(f1, fc1_final)
            
            # """Check if all threshold values are the same"""
            elif constr_type[i] == '>=' or constr_type[i] == '≥':
                if optimal[i] == sub_optimal[i] == not_suitable[i]:
                    f1 = 1 if calc_value[i] >= optimal[i] else 0
                    fc1_final = min(f1, fc1_final)
                elif optimal[i] != sub_optimal[i] == not_suitable[i]:
                    if calc_value[i] >= optimal[i]:
                        f1 = vern_factor if hibernating_flag else 1
                    elif calc_value[i] < optimal[i] and calc_value[i] >= sub_optimal[i]:
                        f1 = vern_factor if hibernating_flag else ((calc_value[i] - optimal[i])/(sub_optimal[i] - optimal[i]) * 0.25) + 0.75
                    else:
                        f1 = 0
                    fc1_final = min(f1, fc1_final)
                
                elif optimal[i] != sub_optimal[i] != not_suitable[i]:
                    if calc_value[i] >= optimal[i]:
                        f1 = vern_factor if hibernating_flag else 1
                    elif calc_value[i] < optimal[i] and calc_value[i] >= sub_optimal[i]:
                        f1 = ((calc_value[i] - optimal[i])/(sub_optimal[i] - optimal[i]) * 0.25) + 0.75
                    elif calc_value[i] < sub_optimal[i] and calc_value[i] >= not_suitable[i]:
                        f1 = ((calc_value[i] - not_suitable[i])/(sub_optimal[i] - not_suitable[i]) * 0.25) + 0.75
                    else:
                        f1 = 0
                    fc1_final = min(f1, fc1_final)

    return fc1_final


# ---------------------------------------- Numba Enhanced Functions End Here---------------------------------- #
# ------------------Intermediate Functions (Not available for Numba enhancement) Starts Here -------------------#
def getInterpolatedTempData(temp1D):
    """
    Get smoothened temperature data for 365 days.
    
    Parameter
    ---------
    temp1D (1-D NumPy Array): Input mean temperature (Deg C)
    
    Returns
    -------
    interp_temp1D (1-D NumPy Array): Smoothened mean temperature.
    """
    interp1D = np.zeros(temp1D.shape)

    tmp = np.array([temp1D[k-1] for k in range(15,temp1D.shape[0]+1,30)])
    mid_doy = np.arange(15,temp1D.shape[0]+1,30)

    # Quadratic spline interpolation for 330 days
    int_mdl = interp1d(mid_doy, tmp, kind='quadratic', fill_value='extrapolate')

    interp1D = int_mdl(np.arange(1,temp1D.shape[0]+1))

    return interp1D

def findbeginningLGPT(mean_temp, threshold):
    """
    Find the beginning date of LGPt with specific threshold
    
    Parameter
    ---------
    mean_temp (1-D NumPy Array): smoothed mean temperature [Deg C]
    threshold (float/int)

    Return
    ------
    lgbt (int): beginning date of the threshold threshold
    """
    for i in range(mean_temp.shape[0]):
        if mean_temp[i]>threshold:
            return i
        else:
            continue
    return i

def getTemperatureGrowingPeriod(temp1D, threshold):
    """
    Calculation of thermal growing period at defined threshold.
    
    Parameters
    ----------
    temp1D (1-D NumPy Array): Input mean temperature (Deg C)
    
    Returns
    -------
    LGPt (int): thermal growing period.
    """
    # Calculation of Temp Profile for 1-D numpy array of climate data input
    tmp = np.array([temp1D[k-1] for k in range(15,temp1D.shape[0]+1,30)])
    mid_doy = np.arange(15,temp1D.shape[0]+1,30)

    # Quadratic spline interpolation for 330 days
    int_mdl = interp1d(mid_doy, tmp, kind='quadratic', fill_value='extrapolate')

    interp1D = int_mdl(np.arange(1,temp1D.shape[0]+1))

    lgpt = interp1D>=threshold

    return np.nansum(lgpt)

def getTemperatureSum(temp1D, threshold):
    """
    Calculation of temperature summation at defined threshold.
    
    Parameters
    ----------
    temp1D (1-D NumPy Array): Input mean temperature (Deg C)
    
    Returns
    -------
    TSUM (int): temperature summation.
    """
    # Calculation of Temp Profile for 1-D numpy array of climate data input
    tmp = np.array([temp1D[k-1] for k in range(15,temp1D.shape[0]+1,30)])
    mid_doy = np.arange(15,temp1D.shape[0]+1,30)

    # Quadratic spline interpolation for 330 days
    int_mdl = interp1d(mid_doy, tmp, kind='quadratic', fill_value='extrapolate')

    interp1D = int_mdl(np.arange(1,temp1D.shape[0]+1))

    interp1D[interp1D <= threshold] = 0

    return np.round(np.sum(interp1D), decimals=0)

def getSmoothTemp(temp1D):

    """Get a smoothened temperature curve done by quadratic spline.
    For 365/366 days.
    
    Parameters
    ----------
    temp1D (1-D NumPy Array): Input mean temperature (Deg C)
    
    Return
    ------
    smootheTemp (1-D NumPy Array): Quadratic spline smoothened temperature (Deg C)
    """
    # Calculation of Temp Profile for 1-D numpy array of climate data input
    tmp = np.array([temp1D[k-1] for k in range(15,temp1D.shape[0]+1,30)])
    mid_doy = np.arange(15,temp1D.shape[0]+1,30)

    # Quadratic spline interpolation for 330 days
    int_mdl = interp1d(mid_doy, tmp, kind='quadratic', fill_value='extrapolate')

    interp1D = int_mdl(np.arange(1,temp1D.shape[0]+1))

    return interp1D


def getTempTrend(temp1D):
    """
    Create a smoothened temperature curve of 365/366 days.
    
    Parameters
    ----------
    temp1 (1-D NumPy Array): Input mean temperature (Deg C)
    
    Returns
    -------
    Temperature Trend (1-D NumPy Array): Upward (+1)/ Downward (-1) trend data.
    """
    # Calculation of Temp Profile for 1-D numpy array of climate data input
    tmp = np.array([temp1D[k-1] for k in range(15,temp1D.shape[0]+1,30)])
    mid_doy = np.arange(15,temp1D.shape[0]+1,30)

    # Quadratic spline interpolation for 330 days
    int_mdl = interp1d(mid_doy, tmp, kind='quadratic', fill_value='extrapolate')

    interp1D = int_mdl(np.arange(1,temp1D.shape[0]+1))

    # Detect the warmest and coldest day of year.
    tmaxidx = np.argmax(interp1D)
    tminidx= np.argmin(interp1D)

    # Any days between the interval of warmest and coldest DOY will be decreasing trend
    meanT_diff = np.ones(interp1D.shape)
    meanT_diff[tmaxidx:tminidx] = -1

    return meanT_diff

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
    tmp = np.array([temp1D[k-1] for k in range(15,temp1D.shape[0]+1,30)])
    mid_doy = np.arange(15,temp1D.shape[0]+1,30)

    # Quadratic spline interpolation for 330 days
    int_mdl = interp1d(mid_doy, tmp, kind='quadratic', fill_value='extrapolate')

    interp1D = int_mdl(np.arange(1,temp1D.shape[0]+1))

    # Detect the warmest and coldest day of year.
    tmaxidx = np.argmax(interp1D)
    tminidx= np.argmin(interp1D)

    # Any days between the interval of warmest and coldest DOY will be decreasing trend
    meanT_diff = np.ones(interp1D.shape)
    meanT_diff[tmaxidx:tminidx] = -1

    # Allocating the temperature to the corresponding classes. New clases introduced (A0, B0)
    A9 = np.sum(np.logical_and(meanT_diff > 0, interp1D < -5))
    A8 = np.sum(np.logical_and(meanT_diff > 0, np.logical_and(
        interp1D >= -5, interp1D < 0)))
    A7 = np.sum(np.logical_and(meanT_diff > 0, np.logical_and(
        interp1D >= 0, interp1D < 5)))
    A6 = np.sum(np.logical_and(meanT_diff > 0, np.logical_and(
        interp1D >= 5, interp1D < 10)))
    A5 = np.sum(np.logical_and(meanT_diff > 0, np.logical_and(
        interp1D >= 10, interp1D < 15)))
    A4 = np.sum(np.logical_and(meanT_diff > 0, np.logical_and(
        interp1D >= 15, interp1D < 20)))
    A3 = np.sum(np.logical_and(meanT_diff > 0, np.logical_and(
        interp1D >= 20, interp1D < 25)))
    A2 = np.sum(np.logical_and(meanT_diff > 0, np.logical_and(
        interp1D >= 25, interp1D < 30)))
    A1 = np.sum(np.logical_and(meanT_diff > 0, interp1D >= 30, interp1D < 35))

    A0 = np.sum(np.logical_and(meanT_diff >0, interp1D >=35))

    B9 = np.sum(np.logical_and(meanT_diff < 0, interp1D < -5))
    B8 = np.sum(np.logical_and(meanT_diff < 0, np.logical_and(
        interp1D >= -5, interp1D < 0)))
    B7 = np.sum(np.logical_and(meanT_diff < 0, np.logical_and(
        interp1D >= 0, interp1D < 5)))
    B6 = np.sum(np.logical_and(meanT_diff < 0, np.logical_and(
        interp1D >= 5, interp1D < 10)))
    B5 = np.sum(np.logical_and(meanT_diff < 0, np.logical_and(
        interp1D >= 10, interp1D < 15)))
    B4 = np.sum(np.logical_and(meanT_diff < 0, np.logical_and(
        interp1D >= 15, interp1D < 20)))
    B3 = np.sum(np.logical_and(meanT_diff < 0, np.logical_and(
        interp1D >= 20, interp1D < 25)))
    B2 = np.sum(np.logical_and(meanT_diff < 0, np.logical_and(
        interp1D >= 25, interp1D < 30)))
    B1 = np.sum(np.logical_and(meanT_diff < 0, interp1D >= 30, interp1D < 35))
    B0 = np.sum(np.logical_and(meanT_diff <0, interp1D >=35))

    # releasing memory
    del (temp1D, interp1D, meanT_diff)

    return [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, B0, B1, B2, B3, B4, B5, B6, B7, B8, B9]


def insertCropSpecificRuleParameters(data):

    rule = data['Constraint'].to_numpy()
    constr_type = data['Type'].to_numpy()
    optimal = list(data['Optimal'].to_numpy())
    sub_optimal = list(data['Sub-Optimal'].to_numpy())
    not_suitable = list(data['Not-Suitable'].to_numpy())

    return rule, constr_type, optimal, sub_optimal, not_suitable


# 4 Modification
def calculateTemperatureProfileClasses(data, input_temp, cycle_len):

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


    tpro365 = getTemperatureProfile(input_temp)
    trpocycle = getTemperatureProfile(input_temp[:cycle_len])

    """For 365 days"""
    N0a = tpro365[0]
    N1a = tpro365[1]
    N2a = tpro365[2]
    N3a = tpro365[3]
    N4a = tpro365[4]
    N5a = tpro365[5]
    N6a = tpro365[6]
    N7a = tpro365[7]
    N8a = tpro365[8]
    N9a = tpro365[9]
    
    N0b = tpro365[10]
    N1b = tpro365[11]
    N2b = tpro365[12]
    N3b = tpro365[13]
    N4b = tpro365[14]
    N5b = tpro365[15]
    N6b = tpro365[16]
    N7b = tpro365[17]
    N8b = tpro365[18]
    N9b = tpro365[19]

    N0 = N0a + N0b
    N1 = N1a + N1b
    N2 = N2a + N2b
    N3 = N3a + N3b
    N4 = N4a + N4b
    N5 = N5a + N5b
    N6 = N6a + N6b
    N7 = N7a + N7b
    N8 = N8a + N8b
    N9 = N9a + N9b


    """For cycle-length specific"""
    L0a = trpocycle[0]
    L1a = trpocycle[1]
    L2a = trpocycle[2]
    L3a = trpocycle[3]
    L4a = trpocycle[4]
    L5a = trpocycle[5]
    L6a = trpocycle[6]
    L7a = trpocycle[7]
    L8a = trpocycle[8]
    L9a = trpocycle[9]

    L0b = trpocycle[10]
    L1b = trpocycle[11]
    L2b = trpocycle[12]
    L3b = trpocycle[13]
    L4b = trpocycle[14]
    L5b = trpocycle[15]
    L6b = trpocycle[16]
    L7b = trpocycle[17]
    L8b = trpocycle[18]
    L9b = trpocycle[19]

    L0 = L0a + L0b
    L1 = L1a + L1b
    L2 = L2a + L2b
    L3 = L3a + L3b
    L4 = L4a + L4b
    L5 = L5a + L5b
    L6 = L6a + L6b
    L7 = L7a + L7b
    L8 = L8a + L8b
    L9 = L9a + L9b

    # Evaluate each constraint equations and append
    calc_value = List()
    for i in range(len(rule)):
        calc_value.append(eval(rule[i]))
    
    return calc_value, constr_type, optimal, sub_optimal, not_suitable

#----------------------------------------------Major Functions Ends Here----------------------------------------------------------------
