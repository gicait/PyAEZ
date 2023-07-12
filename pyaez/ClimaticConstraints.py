""""
PyAEZ version 2.1.0 (June 2023)
2023: Swun Wunna Htet, Kittiphon Boonma

1.  Based of GAEZ appendices, the two different lookup tables of reduction factors
    depending on the annual mean temperatures of >= 20 and < 10 deg C are added.
    With this new tables, new fc3 factors are calculated separately for rainfe and
    irrigated conditions.
    The algorithm will check the annual mean temperature and assess the value from 
    the respective look-up fc3 factor to apply to yield.
    
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


    
    # New Routine
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
        self.lgpt10_map = lgpt10
        
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
                    
                    if irr_or_rain == 'irr':
                            
                        
                        for interval in range(self.lgpt10_class_irr.shape[0]):
                            
                            if self.lgpt10_map[i_row, i_col] in range(self.lgpt10_class_irr[interval][0], self.lgpt10_class_irr[interval][1]):
                                
                                fc1_lgpt10 = self.lgpt10_irr[interval]
                        
                        if self.ann_meanTdaily[i_row, i_col] >= 20:
                            
                            for class_index in range(self.wetness_days_class_irr.shape[0]):
                                
                                if self.LGP_agc[i_row, i_col] in range(self.wetness_days_class_irr[class_index][0], self.wetness_days_class_irr[class_index][1]):
                                    
                                    fc3_irr_wtout_e = self.fc3_irr_gte20[class_index]
                                    
                                    self.fc3[i_row, i_col] = min(fc3_irr_wtout_e, fc1_lgpt10)
                                
                        elif self.ann_meanTdaily[i_row, i_col] < 10:
                            
                            for class_index in range(self.wetness_days_class_irr.shape[0]):
                                
                                if self.LGP_agc[i_row, i_col] in range(self.wetness_days_class_irr[class_index][0], self.wetness_days_class_irr[class_index][1]):
                                    
                                    fc3_irr_wtout_e = self.fc3_irr_lt10[class_index]
                                    
                                    self.fc3[i_row, i_col] =min(fc3_irr_wtout_e, fc1_lgpt10)
                    
                    elif irr_or_rain == 'rain':
                        
                        for interval in range(self.lgpt10_class_rain.shape[0]):
                            
                            if self.lgpt10_map[i_row, i_col] in range(self.lgpt10_class_rain[interval][0], self.lgpt10_class_rain[interval][1]+1):
                                
                                fc1_lgpt10 = self.lgpt10_rain[interval]
                                
                        if np.isnan(self.lgpt10_map[i_row, i_col]):
                            fc1_lgpt10 = 0.
                        if self.ann_meanTdaily[i_row, i_col] >= 20:
                            
                            for class_index in range(self.wetness_days_class_rain.shape[0]):
                                
                                if self.LGP_agc[i_row, i_col] in range(self.wetness_days_class_rain[class_index][0], self.wetness_days_class_rain[class_index][1]+1):
                                        
                                    fc3_irr_wtout_e = self.fc3_rain_gte20[class_index]   
                                    self.fc3[i_row, i_col] = np.minimum(fc3_irr_wtout_e, fc1_lgpt10)
                                        
                                
                        elif self.ann_meanTdaily[i_row, i_col] < 10:
                            
                            for class_index in range(self.wetness_days_class_rain.shape[0]):
                                
                                if self.LGP_agc[i_row, i_col] in range(self.wetness_days_class_rain[class_index][0], self.wetness_days_class_rain[class_index][1]+1):
                                        
                                    fc3_irr_wtout_e = self.fc3_rain_lt10[class_index]   
                                    self.fc3[i_row, i_col] = np.minimum(fc3_irr_wtout_e, fc1_lgpt10)
                        
                        else:
                            self.fc3[i_row, i_col]=1                
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