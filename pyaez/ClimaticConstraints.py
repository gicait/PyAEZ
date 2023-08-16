""""
PyAEZ version 2.1.0 (June 2023)
2023: Swun Wunna Htet, Kittiphon Boonma

1.  Based of GAEZ appendices, the two different lookup tables of reduction factors
    depending on the annual mean temperatures of >= 20 and < 10 deg C are added.
    With this new tables, new fc3 factors are calculated separately for rainfe and
    irrigated conditions.
    The algorithm will check the annual mean temperature and assess the value from 
    the respective look-up fc3 factor to apply to yield.
2. Added missing logic of linear interpolation for pixels with annual mean temperature between
    10 and 20 deg Celsius to extract fc3 constraint factor.
3. Fixing long code to short and consistent.
    
"""
import numpy as np

from pyaez import ALL_REDUCTION_FACTORS_IRR as crop_P_IRR
from pyaez import ALL_REDUCTION_FACTORS_RAIN as crop_P_RAIN

class ClimaticConstraints(object):
    
    def __init__(self):
        
        """Extracting class intervals of LGPagc"""
        self.wetness_days_class_irr = np.array(crop_P_IRR.lgp_eq_class)
        self.wetness_days_class_rain = np.array(crop_P_RAIN.lgp_eq_class)
        
        """Extracting class intervals of LGPT10"""
        self.lgpt10_class_irr = np.array(crop_P_IRR.lgpt10_class)
        self.lgpt10_class_rain = np.array(crop_P_RAIN.lgpt10_class)
        
        """Extracting reduction factors for each LGPagc class"""
        """For irrigated"""
        self.fc3_reduction_fct_irr_gte_20 = 1 - (np.array(crop_P_IRR.lgp_eq_red_fr_gt_20deg)/100)
        self.fc3_reduction_fct_irr_lt_10 = 1 - (np.array(crop_P_IRR.lgp_eq_red_fr_lt_10deg)/100)
        
        self.lgpt10_irr = 1 - (np.array(crop_P_IRR.lgpt10)/100)
        
        
        """For rainfed"""
        self.fc3_reduction_fct_rain_gte_20 = 1 - (np.array(crop_P_RAIN.lgp_eq_red_fr_gt_20deg)/100)
        self.fc3_reduction_fct_rain_lt_10 = 1 - (np.array(crop_P_RAIN.lgp_eq_red_fr_lt_10deg)/100)
        
        self.lgpt10_rain = 1 - (np.array(crop_P_RAIN.lgpt10)/100)
        
        
        
        """Calculation of reduction factor fc3 for all class intervals"""
        
        # Irrigated (for b, c, d constraints), annual_mean_temp >= 20
        self.fc3_irr_gte20 = np.prod(self.fc3_reduction_fct_irr_gte_20, axis = 0)
        
        # Irrigated (for b, c, d constraints), annual_mean_temp < 10
        self.fc3_irr_lt10 = np.prod(self.fc3_reduction_fct_irr_lt_10, axis = 0)
        
        # Rainfed (for b, c, d constraints), annual_mean_temp >= 20
        self.fc3_rain_gte20 = np.prod(self.fc3_reduction_fct_rain_gte_20, axis = 0)
        
        # # Rainfed (for b, c, d constraints), annual_mean_temp < 10
        self.fc3_rain_lt10 = np.prod(self.fc3_reduction_fct_rain_lt_10, axis = 0)



    # Updated Routine
    def applyClimaticConstraints(self, min_temp, max_temp, lgp, lgpt10,  lgp_equv, yield_input, irr_or_rain, tclimate, no_tclimate):
        """
        Parameters
        ----------
        min_temp : 3-D numpy array.
            Minimum temperature in Degree Celsius.
        max_temp : 3-D numpy array.
            Maximum temperature in Degree Celsius.
        lgp : 2-D numpy array.
            Length of Growing Period.
        lgpt10 : 2-D numpy array.
            Frost-free periods (temperature growing period at 10 deg threshold).
        lgp_equv : 2-D numpy array.
            Equivalent Length of Growing Period.
        yield_input : 2-D numpy array.
            Yield map to apply agro-climatic constraint factor. It must corresponds to irr_or_rain setting.
        irr_or_rain : String.
            Choose one of two conditions: irr for irrigated or rain for rainfed conditions.
        tclimate : 2-D numpy array.
            Thermal climate.
        no_tclimate : list.
            A list of thermal climate classes not suitable for crop growth.

        Returns
        -------
        None.

        """
        
        """A wetness indicator is used to interpolate damage ratings from python look-up table"""
        self.ann_meanTdaily = np.round(np.mean((min_temp + max_temp)/2, axis = 2), 2)
        
        original_yield = np.copy(yield_input)
        
        self.LGP_agc = np.zeros(lgp.shape, dtype = int)
        self.clim_adj_yield = np.zeros(lgp.shape)
        self.fc3 = np.zeros(lgp.shape,np.float64)
        self.lgpt10_map = np.copy(lgpt10)
        
        for i_row in range(self.LGP_agc.shape[0]):
            for i_col in range(self.LGP_agc.shape[1]):

                #"""If not, starts LGPagc logic consideration"""
                
                if lgp[i_row, i_col] <= 120:
                    self.LGP_agc[i_row, i_col] = min(120, max(lgp[i_row, i_col], lgp_equv[i_row, i_col]))
                
                elif lgp[i_row, i_col] in range(121,210+1):
                    self.LGP_agc[i_row, i_col] = lgp[i_row, i_col]
                    
                elif lgp[i_row, i_col] > 210:
                    self.LGP_agc[i_row, i_col] = max(210, min(lgp[i_row, i_col], lgp_equv[i_row, i_col]))
        
        
        
        """Agro-climatic reduction factor calculaton from combination of  b, c, d and e constraints"""
        
        for i_row in range(yield_input.shape[0]):
            for i_col in range(yield_input.shape[1]):
                
                """Determine which condition your yield is considered"""
                # 1. For each condition, check lgpt10 in which class interval and use its reduction factor at the designated pixel
                # 2. Next, check the annual mean temperature at the location, and use its related look-up table.
                # 3. After this, go check on LGPagc of that location, and located its corresponding interval's reduction factor
                # 4. Compare "e" constraint and combination of 'b,c,d' constraint to get the most limiting reduction factor, and use it for final fc3 map.
                # 5. Use the final fc3 factor and yield value at the designated to adjust yield.
                
                """Thermal climate screening"""
                if tclimate[i_row, i_col] in no_tclimate:
                    self.fc3[i_row, i_col]= 0.
                    self.clim_adj_yield[i_row, i_col] = 0.
                    
                
                else:
                    # Annual mean temperature of a location

                    ann_mean = self.ann_meanTdaily[i_row, i_col].copy()

                    # Input for Irrigated condition
                    if irr_or_rain == 'irr':

                        lgp_class = self.wetness_days_class_irr.copy() # 13 classes
                        lgpt10_class = self.lgpt10_class_irr.copy() # 13 classes
                        lgpt10 = self.lgpt10_irr.copy() # 13 classes
                        fc3_all10 = self.fc3_reduction_fct_irr_lt_10.copy()
                        fc3_all20 = self.fc3_reduction_fct_irr_gte_20.copy()
                 
                    # Input for Rainfed conditions
                    elif irr_or_rain == 'rain':

                        lgp_class = self.wetness_days_class_rain.copy()
                        lgpt10_class = self.lgpt10_class_rain.copy()
                        lgpt10 = self.lgpt10_rain.copy()
                        fc3_all10 = self.fc3_reduction_fct_rain_lt_10.copy()
                        fc3_all20 = self.fc3_reduction_fct_rain_gte_20.copy()
                    
                    else:
                        raise Exception('Input Error: Provide either one of option (rain or irr)')
                    
                    # finding day-interval to extract corresponding fc3 for 'e' constraint 

                    for val in range(lgpt10_class.shape[0]):
                        startf, endf = lgpt10_class[val][0], lgpt10_class[val][1]
                        fc3_e = 1.
                        
                        if self.lgpt10_map[i_row, i_col] in range(startf, endf+1):
                            fc3_e = lgpt10[val]
                            break
                    
                    # Check for annual mean temperature if it is on selected 20 or 10 deg.
                    # For ann_mean >= 20 OR < 10, fc3 will be extracted from look-up table directly.
                    # For ann_mean between 10 and 20, linear interpolation is done to day-interval specific fc3 from two look up tables.
                    for index in range(lgp_class.shape[0]):
                        fc3_bcd = np.zeros(3) # empty array for b,c,d constraints

                        if self.LGP_agc[i_row, i_col] in range(lgp_class[index][0], lgp_class[index][1]+1):

                            if ann_mean >= 20:
                                fc3_bcd = fc3_all20[:,index]
                                break


                            elif ann_mean <10:
                                fc3_bcd = fc3_all10[:,index]
                                break

                            else:
                                # linear interpolation is done between 10 deg and 20 deg reduction factors depending on annual mean temperature
                                for k in range(3):
                                    fc3_bcd[k] = np.interp(ann_mean, [10,20], [fc3_all10[k, index], fc3_all20[k, index]])
                                break
                    
                    # Choose the most limiting fc3 from four constraints' contribution and apply to the yield
                    min_fc3_bcd = np.min(fc3_bcd)

                    self.fc3[i_row, i_col] = min(min_fc3_bcd, fc3_e)

                    self.clim_adj_yield[i_row, i_col] = original_yield[i_row, i_col] * self.fc3[i_row, i_col]

    def getClimateAdjustedYield(self):
        """
        Generate yield map adjusted with agro-climatic constraints.

        Returns
        -------
        TYPE: 2-D numpy array.
            Agro-climatic constraint applied yield.

        """
        return np.round(self.clim_adj_yield,decimals = 0)
    
    def getClimateReductionFactor(self):
        """
        Generates agro-climatic constraint map (fc3) applied to unconstrainted 
        yield.

        Returns
        -------
        TYPE : 2-D numpy array.
            Agro-climatic constraint map (fc3).

        """
        return np.round(self.fc3, decimals = 2)

#----------------- End of file -------------------------#
