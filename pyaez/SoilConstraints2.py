"""
PyAEZ version 2.3 (Dec 2024)
Module 4: Soil Constraints
2016: N. Lakmal Deshapriya
2023: Swun Wunna Htet
2024 (Dec) : Swun Wunna Htet

Modifications
1.  All reduction factors will be externally imported from excel sheets instead of providing
    python scripts.
2.  All reduction factors from excel sheets are recorded as python dictionaries. Algorithm will be the same as
    previous version. But the access of variables will be heavily depending on pandas incorporation and dictionaries.
3.  New soil evaluation method from GAEZ v5 is implemented.
4.  Soil property ratings with zero soil attributes will be set as 100 by default.

"""
import numpy as np
import pandas as pd
import math

class SoilConstraints(object):

    def __init__(self, rooting_depth):
        """ Initialization of Soil Constraints Class"""

        self.root_depth = rooting_depth # Crop-specific rooting depth in centimeters
        
        # Soil Attribute Ratings by Soil Profile Depth Layers (%) of new soil evaluation
        # Adjustment ratings for seven soil depth classes (D1 to D7)
        self.attr_ratings  = {
            'GRC': np.array([100., 100., 100., 100., 100., 100., 95.5]),
            'TXT': np.array([100., 100., 100., 100., 100., 100., 100.]),
            'OC':  np.array([100., 100.,  95., 83.3, 66., 66., 62.]),
            'pH': np.array([66., 70., 70., 76.7, 76.7, 83.3, 90.]),
            'CECsoil': np.array([100., 100., 100., 100., 100., 100., 100.]),
            'CECclay': np.array([100., 100., 100., 100., 100., 100., 100.]),
            'TEB': np.array([100., 100., 100., 83.3, 100., 100., 100.]),
            'BS': np.array([91.3, 92., 93.3, 92., 94.7, 96.7, 94.]),
            'ESP': np.array([100., 100., 100., 100., 100., 100., 100.]),
            'CCB':  np.array([100., 100., 100., 100., 100., 100., 100.]),
            'GYP':  np.array([100., 100., 100., 100., 100., 100., 100.]),
            'EC':  np.array([100., 100., 100., 100., 100., 100., 100.])}
        
        # Root mass distribution (%) by HWSD soil depth layer (D1 - D7) and Rooting Depth of Crop (cm)
        # >100 cm : For deep soils of rootable depth
        # 50 - 100 cm : For moderate deep soils of rootable depth
        # 10 - 50 cm: For shallow soils of rootable depth
        # <10 cm : For very shallow of rootable depth

        # This soil quality adjustment is done to SQ2.
        self.RMSDRD_table = {
            '>100':np.array([[62., 30., 8., 0., 0., 0., 0.],
                              [58., 28., 10., 3., 1., 0., 0.],
                              [54., 25., 12., 5., 3., 1., 0.],
                              [49., 24., 14., 6., 4., 3., 0.],
                              [42., 26., 14., 8., 4., 6., 0.]]),
            '50-100':np.array([[62., 30., 8., 0., 0., 0., 0.],
                               [58., 28., 10., 3., 1., 0., 0.],
                               [54., 25., 12., 5., 4., 0., 0.],
                               [48., 25., 15., 7., 5., 0., 0.],
                               [42., 27., 15., 10., 6., 0., 0.]]),
            '10-50': np.array([[62., 30., 8., 0., 0., 0., 0.],
                               [58., 30., 12., 0., 0., 0., 0.],
                               [57., 28., 15., 0., 0., 0., 0.],
                               [52., 29., 19., 0., 0., 0., 0.],
                               [47., 32., 21., 0., 0., 0., 0.]]),
            '<10':np.array([[100., 0., 0., 0., 0., 0., 0.],
                            [100., 0., 0., 0., 0., 0., 0.], 
                            [100., 0., 0., 0., 0., 0., 0.], 
                            [100., 0., 0., 0., 0., 0., 0.],
                            [100., 0., 0., 0., 0., 0., 0.]])                   
        }
        
        
    def calculateSoilQualities(self, irr_or_rain, soil_characteristics_path):
        """
        Intermediate function.
        
        Applying HWSD v2.1 soil evaluation method (7-layer approach).
        
        Args:
            irr_or_rain (String): Irrigated (I) or Rainfed (R)
            soil_characteristics_path (String): File path of SMU's soil characteristics
        """
        # reading soil properties from excel sheet
        main_df = pd.read_excel(soil_characteristics_path, sheet_name= None)

        # check if input excel have seven sheets, else, raise error.
        if len(list(main_df.keys())) != 7:
            raise Exception(r'Input excel must contain seven sheet. Please revise the input.')
        
        # check if all sheets have same number of SMUs, if not raise error.
        if main_df['D1'].shape != main_df['D2'].shape != main_df['D3'].shape != main_df['D4'].shape != main_df['D5'].shape != main_df['D6'].shape != main_df['D7'].shape:
            raise Exception(r'Unequal dimensions of soil characteristics sheets detected. Please revise the input again.')

        subsoil_class = ['D1','D2', 'D3', 'D4', 'D5', 'D6', 'D7']

        topsoil_df = main_df['D1'].copy()
        subsoil_df = main_df.copy()
        del(subsoil_df['D1'])


        self.SMU = topsoil_df['CODE']
        # zero array of all individual soil qualities for each self.SMU
        
        # 1st =  7 for 7 soil layers to evaluate, 2nd = self.SMUs, 3rd = Soil Qualities
        SQ_array = np.zeros((7, self.SMU.shape[0] ,7))


        for i in range(SQ_array.shape[0]):

            if i==0:
                sub_df = topsoil_df.copy()
                top_sub = 'top'
            else:
                df = subsoil_df.copy()
                top_sub = 'sub'
                sub_df = df[subsoil_class[i]]
            
            for j in range(self.SMU.shape[0]):

        
                t_code = sub_df.loc[sub_df['CODE'] == self.SMU[j]]

                # SQ1 calculation
                SQ_array[i,j,0] = self.soil_qty_1(TXT_val= t_code['TXT'].iloc[0], OC_val = t_code['OC'].iloc[0], pH_val=t_code['pH'].iloc[0],
                                        TEB_val= t_code['TEB'].iloc[0], condition= irr_or_rain, top_sub= top_sub)
            
                # SQ2 calculation
                SQ_array[i,j,1] = self.soil_qty_2(RSD_val = t_code['RSD'].iloc[0],TXT_val= t_code['TXT'].iloc[0], BS_val=t_code['BS'].iloc[0], CECclay_val=t_code['CEC_clay'].iloc[0], 
                                        CECsoil_val= t_code['CEC_soil'].iloc[0], pH_val=t_code['pH'].iloc[0], condition = irr_or_rain, top_sub= top_sub, soil_depth_class= i)
            
                # SQ3 calculation
                SQ_array[i,j,2] = self.soil_qty_3(RSD_val=t_code['RSD'].iloc[0], TXT_val = t_code['TXT'].iloc[0], VSP_val=t_code['VSP'].iloc[0], GSP_val= t_code['GSP'].iloc[0],  
                                                  SPH_val=t_code['SPH'].iloc[0], OTR_val=t_code['OTR'].iloc[0], ISL_val=t_code['ISL'].iloc[0], condition= irr_or_rain, soil_depth_class= i)

                # SQ4 calculation
                SQ_array[i,j,3] = self.soil_qty_4(DRG_val= t_code['DRG'].iloc[0], SPH_val=t_code['SPH'].iloc[0], condition= irr_or_rain)

                # SQ5 calculation
                SQ_array[i,j,4] = self.soil_qty_5(ESP_val=t_code['ESP'].iloc[0], EC_val=t_code['EC'].iloc[0], SPH_val=t_code['SPH'].iloc[0], condition = irr_or_rain)

                # SQ6 calculation
                SQ_array[i,j,5] = self.soil_qty_6(CCB_val=t_code['CCB'].iloc[0], GYP_val=t_code['GYP'].iloc[0], SPH_val=t_code['SPH'].iloc[0], condition=irr_or_rain)
                
                # SQ7 calculation
                SQ_array[i,j,6] = self.soil_qty_7(RSD_val=t_code['RSD'].iloc[0], GRC_val=t_code['GRC'].iloc[0], SPH_val=t_code['SPH'].iloc[0], TXT_val=t_code['TXT'].iloc[0], 
                                        VSP_val=t_code['VSP'].iloc[0], condition = irr_or_rain, soil_depth_class=i)

        # For each soil quality type, average them out.
        SQ_top = None
        SQ_sub = None
        self.SQ_array = np.zeros((self.SMU.shape[0], 7))

        for i in range(7):
            # If it is SQ2, SQ3 or SQ7, the final SQ calculation is different from others
            if i in [1,2,6]:
                self.SQ_array[:,i] = np.sum(SQ_array[:,:,i])
            else:
                SQ_top = SQ_array[0,:,i]
                SQ_sub = np.mean(SQ_array[1:, :,i], axis = 0)
                self.SQ_array[:,i] = np.mean([SQ_top, SQ_sub], axis = 0)

        self.SQ_array_pd = pd.DataFrame({'SMU': self.SMU, 
                                            'SQ1':self.SQ_array[:,0],
                                            'SQ2':self.SQ_array[:,1],
                                            'SQ3':self.SQ_array[:,2],
                                            'SQ4':self.SQ_array[:,3],
                                            'SQ5':self.SQ_array[:,4],
                                            'SQ6':self.SQ_array[:,5],
                                            'SQ7':self.SQ_array[:,6]})


    # Sub-routines for each aspect of soil quality calculation
    # All background calculations of seven soil qualities for subsoil and topsoil
    # All SQ calculations are fully referred to new GAEZ soil evaluation with new dataset.
        
    # SQ1: Nutrient availability (Changes done)
    def soil_qty_1(self, TXT_val, OC_val, pH_val, TEB_val, condition, top_sub):

        if condition == 'I':
            para = self.SQ1_irr.copy()
        else:
            para = self.SQ1_rain.copy()
        
        TXT_intp = 100 if pd.isna(TXT_val) else para['TXT_fct'][np.where(para['TXT_val'] == TXT_val)[0][0]]

        # check if current pH is either in monotonically increasing or decresing trend
        pH_intp = None
        if pH_val < 7:
            pH_intp = 100 if math.isnan(pH_val) else np.interp(pH_val, para['pH_L_val'], para['pH_L_fct'])
        else:
            pH_intp = 100  if math.isnan(pH_val) else np.interp(pH_val, para['pH_H_val'], para['pH_H_fct'])

        OC_intp  = 100 if math.isnan(OC_val) else np.interp(OC_val , para['OC_val'], para['OC_fct'])
        TEB_intp = 100 if math.isnan(TEB_val) else np.interp(TEB_val, para['TEB_val'], para['TEB_fct']) 


        if top_sub == 'top':
            min_factor = np.min([TXT_intp, pH_intp, OC_intp, TEB_intp])
            summation = np.sum([TXT_intp, pH_intp, OC_intp, TEB_intp]) - min_factor
            final_factor = min_factor * (summation/3) * 10**-2
        else:
            min_factor = np.min([TXT_intp, pH_intp, TEB_intp])
            summation = np.sum([TXT_intp, pH_intp, TEB_intp]) - min_factor
            final_factor = min_factor * (summation/2) * 10**-2
        
        return final_factor
    
    # SQ2: Nutrient retention capacity (Changes done.)
    def soil_qty_2(self, RSD_val, TXT_val, BS_val, CECclay_val, CECsoil_val, pH_val, condition, top_sub, soil_depth_class):
        
        if condition == 'I':
            para = self.SQ2_irr.copy()
        else:
            para = self.SQ2_rain.copy()
        

        TXT_intp = 100 if pd.isna(TXT_val) else para['TXT_fct'][np.where(para['TXT_val'] == TXT_val)[0][0]]

        BS_intp = 100 if math.isnan(BS_val) else np.interp(BS_val, para['BS_val'], para['BS_fct'])
        CECclay_intp = 100 if math.isnan(CECclay_val) else np.interp(CECclay_val, para['CECclay_val'], para['CECclay_fct'])
        CECsoil_intp = 100 if math.isnan(CECclay_val) else np.interp(CECsoil_val, para['CECsoil_val'], para['CECsoil_fct'])
        
        # check if current pH is either in monotonically increasing or decresing trend
        pH_intp = None
        if pH_val < 7:
            pH_intp = 100 if math.isnan(pH_val) else np.interp(pH_val, para['pH_L_val'], para['pH_L_fct'])
        else:
            pH_intp = 100 if math.isnan(pH_val) else np.interp(pH_val, para['pH_H_val'], para['pH_H_fct'])


        if top_sub == 'top':
            min_factor = np.min([TXT_intp, BS_intp, CECsoil_intp])
            summation = np.sum([TXT_intp, BS_intp, CECsoil_intp]) - min_factor
            final_factor = min_factor * (summation/2) * 10**-2
        else:
            min_factor = np.min([TXT_intp, BS_intp, CECclay_intp, pH_intp])
            summation = np.sum([TXT_intp, BS_intp, CECclay_intp, pH_intp]) - min_factor
            final_factor = min_factor * (summation/3) * 10**-2
        
        # Calculated soil qualtiy is adjusted for a specific HWSD soil depth class based on rooting depth condition.
        RD_keys = list(self.RMSDRD_table.keys())
        RD_idx = None # String index for table dictionary
        if RSD_val >= 100:
            RD_idx = RD_keys[0]
        elif RSD_val in range(50,100):
            RD_idx = RD_keys[1]
        elif RSD_val in range(10,50):
            RD_idx = RD_keys[2]
        else:
            RD_idx = RD_keys[3]

        # (New logic) the calculated SQ from the 
        table = self.RMSDRD_table[RD_idx]

        i = None
        if self.root_depth in range(0,41):
            i =0
        elif self.root_depth in range(41,61):
            i = 1
        elif self.root_depth in range(61,81):
            i = 2
        elif self.root_depth in range(81,101):
            i = 3
        else:
            i = 4
        adj_fct = table[i, soil_depth_class]
        final_factor = final_factor * adj_fct/100

        return final_factor

    # SQ3: Rooting Conditions (Changes done)
    # Note: SQ3 calculation procedure is the same, regardless of topsoil or subsoil
    # All variables are categorical in nature, thus no adjustment based on soil depth classes is required.
    def soil_qty_3(self, RSD_val, TXT_val, VSP_val, GSP_val, SPH_val, OTR_val, ISL_val, condition, soil_depth_class):

        if condition == 'I':
            para = self.SQ3_irr.copy()
        else:
            para = self.SQ3_rain.copy()

        # # accessing adjustment factor for the designated soil depth layer class
        # TXT_adj = self.attr_ratings['TXT'][soil_depth_class]

        RSD_intp = 100 if math.isnan(RSD_val) else para['RSD_fct'][np.where(para['RSD_val'] == RSD_val)[0][0]]
        SPH_intp = 100 if pd.isna(SPH_val) else  para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]
        OTR_intp = 100 if math.isnan(OTR_val) else np.interp(OTR_val, para['OTR_val'], para['OTR_fct'])
        TXT_intp = 100 if pd.isna(TXT_val) else  para['TXT_fct'][np.where(para['TXT_val'] == TXT_val)[0][0]]
        VSP_intp = 100 if pd.isna(VSP_val) else  para['VSP_fct'][np.where(para['VSP_val'] == VSP_val)[0][0]]
        GSP_intp = 100 if pd.isna(GSP_val) else  para['GSP_fct'][np.where(para['GSP_val'] == GSP_val)[0][0]]
        ISL_intp = 100 if pd.isna(ISL_val) else  para['ISL_fct'][np.where(para['ISL_val'] == ISL_val)[0][0]]

    
        final_factor = RSD_intp * np.min([SPH_intp, OTR_intp, TXT_intp, VSP_intp, GSP_intp, ISL_intp]) * 10**-2

        # Calculated soil qualtiy is adjusted for a specific HWSD soil depth class based on rooting depth condition.
        RD_keys = list(self.RMSDRD_table.keys())
        RD_idx = None # String index for table dictionary
        if RSD_val >= 100:
            RD_idx = RD_keys[0]
        elif RSD_val in range(50,100):
            RD_idx = RD_keys[1]
        elif RSD_val in range(10,50):
            RD_idx = RD_keys[2]
        else:
            RD_idx = RD_keys[3]

        # (New logic) the calculated SQ from the 
        table = self.RMSDRD_table[RD_idx]

        i = None
        if self.root_depth in range(0,41):
            i =0
        elif self.root_depth in range(41,61):
            i = 1
        elif self.root_depth in range(61,81):
            i = 2
        elif self.root_depth in range(81,101):
            i = 3
        else:
            i = 4
        adj_fct = table[i, soil_depth_class]
        final_factor = final_factor * adj_fct/100

        return final_factor
    
    # SQ4: Oxygen availability (Nothing's changed)
    # All variables are categorical in nature, thus no adjustment based on soil depth classes is required.
    def soil_qty_4(self, DRG_val, SPH_val, condition):
        
        if condition == 'I':
            para = self.SQ4_irr.copy()
        else:
            para = self.SQ4_rain.copy()

        DRG_intp = 100 if pd.isna(DRG_val) else para['DRG_fct'][np.where(para['DRG_val'] == DRG_val)[0][0]]
        SPH_intp = 100 if pd.isna(SPH_val) else para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]


        final_factor = np.min([DRG_intp, SPH_intp])

        return final_factor
    
    # SQ5: Presence of salinity and sodicity (Changes done)
    def soil_qty_5(self, ESP_val, EC_val, SPH_val, condition):

        if condition == 'I':
            para = self.SQ5_irr.copy()
        else:
            para = self.SQ5_rain.copy()

        ESP_intp = 100 if np.isnan(ESP_val) else np.interp(ESP_val, para['ESP_val'], para['ESP_fct'])

        # if Saline or salic soil phase occurr, the maximum EC value will be set up to d dS/m.
        if SPH_val in ['Saline', 'Salic', 'saline', 'salic']:
            EC_intp = np.interp(4., para['EC_val'], para['EC_fct'])
        else:
            EC_intp = 100 if np.isnan(EC_val) else np.interp(EC_val, para['EC_val'], para['EC_fct'])

        SPH_intp = 100 if pd.isna(SPH_val) else para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]

        final_factor = np.min([ESP_intp, EC_intp, SPH_intp])

        return final_factor
    
    # SQ6: Presence of lime and gypsum (Changes done)
    def soil_qty_6(self, CCB_val, GYP_val, SPH_val, condition):

        if condition == 'I':
            para = self.SQ6_irr.copy()
        else:
            para = self.SQ6_rain.copy()

        CCB_intp = 100 if np.isnan(CCB_val) else np.interp(CCB_val, para['CCB_val'], para['CCB_fct'])
        GYP_intp = 100 if np.isnan(GYP_val) else np.interp(GYP_val, para['GYP_val'], para['GYP_fct'])
        SPH_intp = 100 if pd.isna(SPH_val) else para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]

        final_factor = np.min([CCB_intp, GYP_intp, SPH_intp])


        return final_factor

    # SQ7: Soil Workability (Changes done)
    # All variables are categorical in nature, thus no adjustment based on soil depth classes is required.
    def soil_qty_7(self, RSD_val, GRC_val, SPH_val, TXT_val, VSP_val, condition, soil_depth_class):

        if condition == 'I':
            para = self.SQ7_irr.copy()
        else:
            para = self.SQ7_rain.copy()

        RSD_intp = 100 if np.isnan(RSD_val) else para['RSD_fct'][np.where(para['RSD_val'] == RSD_val)[0][0]]
        GRC_intp = 100 if np.isnan(GRC_val) else np.interp(GRC_val, para['GRC_val'], para['GRC_fct'])
        SPH_intp = 100 if pd.isna(SPH_val) else para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]
        TXT_intp = 100 if pd.isna(TXT_val) else para['TXT_fct'][np.where(para['TXT_val'] == TXT_val)[0][0]]
        VSP_intp = 100 if np.isnan(VSP_val) else np.interp(VSP_val, para['VSP_val'], para['VSP_fct'])

        min_factor = np.min([RSD_intp, GRC_intp, SPH_intp, TXT_intp, VSP_intp])
        summation = np.sum([RSD_intp, GRC_intp, SPH_intp, TXT_intp, VSP_intp]) - min_factor
        final_factor = min_factor * (summation/4) * 10**-2

        # Calculated soil qualtiy is adjusted for a specific HWSD soil depth class based on rooting depth condition.
        RD_keys = list(self.RMSDRD_table.keys())
        RD_idx = None # String index for table dictionary
        if RSD_val >= 100:
            RD_idx = RD_keys[0]
        elif RSD_val in range(50,100):
            RD_idx = RD_keys[1]
        elif RSD_val in range(10,50):
            RD_idx = RD_keys[2]
        else:
            RD_idx = RD_keys[3]

        # (New logic) the calculated SQ from the 
        table = self.RMSDRD_table[RD_idx]

        i = None
        if self.root_depth in range(0,41):
            i =0
        elif self.root_depth in range(41,61):
            i = 1
        elif self.root_depth in range(61,81):
            i = 2
        elif self.root_depth in range(81,101):
            i = 3
        else:
            i = 4
        adj_fct = table[i, soil_depth_class]
        final_factor = final_factor * adj_fct/100

        return final_factor
    
    #------------------------------------ SUBROUTINE FUNCTIONS ENDS HERE-------------------------------------#
    
    
    #--------------------------------------  MAIN FUNCTIONS STARTS HERE  ------------------------------------#
    def importSoilReductionSheet(self, rain_sheet_path, irr_sheet_path):
        """
        Upload the soil reduction factor as excel sheet into Module IV object class.
        All soil reduction factors are rated based on crop/LUT-specifc edaphic suitability
        to a particular soil characteristic.
        Args:
            rain_sheet_path (String): File path of soil reduction factor for rainfed condition in excel xlsx format.
            irr_sheet_path (String): File path of soil reduction factor for rainfed condition in excel xlsx format.
            """
        
        # reading each individual excel sheet databases
        rain_df = pd.read_excel(rain_sheet_path, header = None, sheet_name= None)
        irr_df = pd.read_excel(irr_sheet_path, header = None, sheet_name= None)

        # All soil characteristics are stored as tuples corresponding to a particular soil quality
        
        # SQ 1: Nutrient Availability (4 parameters x 2) (Changes in pH)
        self.SQ1_rain ={
        'TXT_val' : (rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'TXT_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'TXT_fct':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'TXT_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float), # numerical
        'OC_val':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'OC_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'OC_fct':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'OC_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_H_val':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'pH_H_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_H_fct':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'pH_H_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_L_val':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'pH_L_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_L_fct':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'pH_L_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'TEB_val':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'TEB_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'TEB_fct':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'TEB_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)# numerical
            }
        # (Changes in pH)
        self.SQ1_irr =  {
        'TXT_val' : (irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'TXT_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'TXT_fct':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'TXT_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float), # numerical
        'OC_val':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'OC_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'OC_fct':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'OC_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_H_val':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'pH_H_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_H_fct':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'pH_H_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_L_val':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'pH_L_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_L_fct':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'pH_L_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'TEB_val':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'TEB_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'TEB_fct':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'TEB_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)# numerical
            }
        # SQ1 ok
        
        # SQ 2: Nutrient Retention Capacity (5 parameters x 2 rows) (Changes in pH)
        self.SQ2_rain ={
        'TXT_val' : (rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'TXT_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'TXT_fct':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'TXT_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'BS_val':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'BS_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'BS_fct':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'BS_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CECsoil_val':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'CECsoil_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CECsoil_fct':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'CECsoil_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'pH_H_val':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'pH_H_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_H_fct':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'pH_H_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_L_val':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'pH_L_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_L_fct':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'pH_L_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'CECclay_val':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'CECclay_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),        
        'CECclay_fct':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'CECclay_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)
            }
        
        self.SQ2_irr =  {
        'TXT_val' : (irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'TXT_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'TXT_fct':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'TXT_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'BS_val':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'BS_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'BS_fct':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'BS_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CECsoil_val':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'CECsoil_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CECsoil_fct':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'CECsoil_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'pH_H_val':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'pH_H_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_H_fct':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'pH_H_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_L_val':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'pH_L_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_L_fct':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'pH_L_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'CECclay_val':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'CECclay_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CECclay_fct':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'CECclay_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
            }
        
        # SQ 3: Rooting Conditions (4 parameters x 2) (changed to new soil evaluations.)
        self.SQ3_rain ={
        'RSD_val' : (rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'RSD_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'RSD_fct':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'RSD_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'OTR_val':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'OTR_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'OTR_fct':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'OTR_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GRC_val':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'GRC_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GRC_fct':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'GRC_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'TXT_val':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'TXT_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'TXT_fct':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'TXT_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'VSP_val':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'VSP_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'VSP_fct':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'VSP_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GSP_val':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'GSP_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GSP_fct':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'GSP_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'ISL_val':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'ISL_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'ISL_fct':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'ISL_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)
        }
        # Changed to new soil evaluations.
        self.SQ3_irr =  {
        'RSD_val' : (irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'RSD_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'RSD_fct':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'RSD_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'OTR_val':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'OTR_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'OTR_fct':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'OTR_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GRC_val':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'GRC_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GRC_fct':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'GRC_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'TXT_val':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'TXT_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'TXT_fct':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'TXT_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'VSP_val':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'VSP_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'VSP_fct':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'VSP_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GSP_val':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'GSP_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GSP_fct':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'GSP_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'ISL_val':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'ISL_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'ISL_fct':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'ISL_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)
        }
        
        # SQ 4: Oxygen Availability (2 parameters x 2) (Nothing's changed)
        self.SQ4_rain ={
        'DRG_val' : (rain_df['SQ4'].loc[rain_df['SQ4'][0] == 'DRG_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'DRG_fct':(rain_df['SQ4'].loc[rain_df['SQ4'][0] == 'DRG_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(rain_df['SQ4'].loc[rain_df['SQ4'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(rain_df['SQ4'].loc[rain_df['SQ4'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        }
        # No changes.
        self.SQ4_irr =  {
        'DRG_val' : (irr_df['SQ4'].loc[irr_df['SQ4'][0] == 'DRG_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'DRG_fct':(irr_df['SQ4'].loc[irr_df['SQ4'][0] == 'DRG_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(irr_df['SQ4'].loc[irr_df['SQ4'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(irr_df['SQ4'].loc[irr_df['SQ4'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        }

        # SQ 5: Presence of Salinity and Sodicity (3 parameters x 2) (nothing's changed)
        self.SQ5_rain ={
        'ESP_val' : (rain_df['SQ5'].loc[rain_df['SQ5'][0] == 'ESP_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'ESP_fct':(rain_df['SQ5'].loc[rain_df['SQ5'][0] == 'ESP_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'EC_val':(rain_df['SQ5'].loc[rain_df['SQ5'][0] == 'EC_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'EC_fct':(rain_df['SQ5'].loc[rain_df['SQ5'][0] == 'EC_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(rain_df['SQ5'].loc[rain_df['SQ5'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(rain_df['SQ5'].loc[rain_df['SQ5'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)
        }
        # Nothing's changed.
        self.SQ5_irr =  {
        'ESP_val' : (irr_df['SQ5'].loc[irr_df['SQ5'][0] == 'ESP_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'ESP_fct':(irr_df['SQ5'].loc[irr_df['SQ5'][0] == 'ESP_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'EC_val':(irr_df['SQ5'].loc[irr_df['SQ5'][0] == 'EC_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'EC_fct':(irr_df['SQ5'].loc[irr_df['SQ5'][0] == 'EC_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(irr_df['SQ5'].loc[irr_df['SQ5'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(irr_df['SQ5'].loc[irr_df['SQ5'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)
        }

        # SQ 6: Presence of Lime and Gypsum (3 parameters x 2) (No changes)
        self.SQ6_rain ={
        'CCB_val' : (rain_df['SQ6'].loc[rain_df['SQ6'][0] == 'CCB_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CCB_fct':(rain_df['SQ6'].loc[rain_df['SQ6'][0] == 'CCB_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GYP_val':(rain_df['SQ6'].loc[rain_df['SQ6'][0] == 'GYP_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GYP_fct':(rain_df['SQ6'].loc[rain_df['SQ6'][0] == 'GYP_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(rain_df['SQ6'].loc[rain_df['SQ6'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(rain_df['SQ6'].loc[rain_df['SQ6'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)
            }
        # No changes.
        self.SQ6_irr =  {
        'CCB_val' : (irr_df['SQ6'].loc[irr_df['SQ6'][0] == 'CCB_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CCB_fct':(irr_df['SQ6'].loc[irr_df['SQ6'][0] == 'CCB_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GYP_val':(irr_df['SQ6'].loc[irr_df['SQ6'][0] == 'GYP_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GYP_fct':(irr_df['SQ6'].loc[irr_df['SQ6'][0] == 'GYP_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(irr_df['SQ6'].loc[irr_df['SQ6'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(irr_df['SQ6'].loc[irr_df['SQ6'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)
            }
        
        # SQ 7: Workability (5 parameters x 2) (No changes)
        self.SQ7_rain ={
        'RSD_val' : (rain_df['SQ7'].loc[rain_df['SQ7'][0] == 'RSD_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'RSD_fct':(rain_df['SQ7'].loc[rain_df['SQ7'][0] == 'RSD_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GRC_val':(rain_df['SQ7'].loc[rain_df['SQ7'][0] == 'GRC_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GRC_fct':(rain_df['SQ7'].loc[rain_df['SQ7'][0] == 'GRC_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(rain_df['SQ7'].loc[rain_df['SQ7'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(rain_df['SQ7'].loc[rain_df['SQ7'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'TXT_val':(rain_df['SQ7'].loc[rain_df['SQ7'][0] == 'TXT_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'TXT_fct':(rain_df['SQ7'].loc[rain_df['SQ7'][0] == 'TXT_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'VSP_val':(rain_df['SQ7'].loc[rain_df['SQ7'][0] == 'VSP_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'VSP_fct':(rain_df['SQ7'].loc[rain_df['SQ7'][0] == 'VSP_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)
        }
        # No changes.
        self.SQ7_irr =  {
        'RSD_val' : (irr_df['SQ7'].loc[irr_df['SQ7'][0] == 'RSD_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'RSD_fct':(irr_df['SQ7'].loc[irr_df['SQ7'][0] == 'RSD_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GRC_val':(irr_df['SQ7'].loc[irr_df['SQ7'][0] == 'GRC_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GRC_fct':(irr_df['SQ7'].loc[irr_df['SQ7'][0] == 'GRC_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(irr_df['SQ7'].loc[irr_df['SQ7'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(irr_df['SQ7'].loc[irr_df['SQ7'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'TXT_val':(irr_df['SQ7'].loc[irr_df['SQ7'][0] == 'TXT_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'TXT_fct':(irr_df['SQ7'].loc[irr_df['SQ7'][0] == 'TXT_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'VSP_val':(irr_df['SQ7'].loc[irr_df['SQ7'][0] == 'VSP_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'VSP_fct':(irr_df['SQ7'].loc[irr_df['SQ7'][0] == 'VSP_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)
        }
    
    def calculateSoilRatings(self, input_level):
        """
        Calculate the soil suitability rating factors based on soil qualities and
        selected input-level/management of the crop.
        
        Requires:   1. importSoilReductionSheet
                    2. calculateSoilQualities
        
        Args:
            input-level(string): L (Low-level), I (Intermediate-level), H (High-level)
        
        Return:
            None.
        """

        self.SR = np.zeros((self.SMU.shape[0])) # first column for soil unit code and other column for SR

        for i in range(0, self.SR.shape[0]):

            if input_level == 'L':
                min_factor = np.min([self.SQ_array[i,2], self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]])
                summation = np.sum([self.SQ_array[i,2], self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]])- min_factor

                fsq = min_factor * (summation/4)

                self.SR[i] = self.SQ_array[i,0] *  fsq * 10**-4

            elif input_level == 'I':
                min_factor = np.min([self.SQ_array[i,2], self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]])
                summation = np.sum([self.SQ_array[i,2], self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]]) - min_factor

                fsq = min_factor * (summation/4)

                self.SR[i] = 0.5 * (self.SQ_array[i,0]+self.SQ_array[i,1]) * fsq * 10**-4

            elif input_level == 'H':
                min_factor = np.min([self.SQ_array[i,2], self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]])
                summation = np.sum([self.SQ_array[i,2],self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]]) - min_factor

                fsq = min_factor * (summation/4)
                self.SR[i] = self.SQ_array[i,1] * fsq * 10**-4
            else:
                print('Wrong Input Level !')
        
        self.SR_pd = pd.DataFrame({'SMU':self.SMU,
                                  'SR': self.SR})
    
    def getSoilQualities(self):
        """
        Obtain the calculated seven soil qualities for each soil mapping unit
        as pandas dataframe.
        Args:
            None.
        Return:
            Soil Ratings: Pandas DataFrame in such [SMU, SQ1, SQ2, SQ3, SQ4, SQ5, SQ6, SQ7]."""
        return self.SQ_array_pd

    def getSoilRatings(self):
        """
        Obtain the calculated soil-mapping unit specific soil suitability ratings.
        Soil ratings ranges from 0 (Not-Suitable) to 1 (Most Suitable)
        
        Args:
            None.
        Return:
            Soil Ratings: Pandas DataFrame in such [SMU, SR].
        """
        return self.SR_pd
    
    def applySoilConstraints(self, soil_map, yield_in):
        """
        Apply yield reduction to input yield map with specific input-management
        level soil ratings.
        
        Args:
            soil_map (Numerical): 2-D NumPy array. Soil map with unique soil mapping units.
            yield_in (Numerical): 2-D NumPy array. Input yield map (kg/ha).
        
        Returns:
            Soil adjusted yield: 2-D NumPy array (Unit: same as input yield)
        """

        yield_final = np.copy(yield_in)
        self.soilsuit_map = np.zeros(soil_map.shape)

        for i1 in range(0, self.SR.shape[0]):
            temp_idx = soil_map==self.SMU[i1]
            self.soilsuit_map[temp_idx] = self.SR[i1]
            yield_final[temp_idx] = yield_in[temp_idx] * (self.SR[i1]/100)
        

        return yield_final
    
    def getSoilSuitabilityMap(self):
        """
        Obtain the soil suitability map based on calculation of soil qualities and 
        ratings based on defined input/management level.
        Values range from 0 (Not Suitable) to 1 (Very suitable).

        Args:
            None.
        Return:
            Soil reduction factor: 2-D NumPy Array.
        """
        return np.round(self.soilsuit_map/100, 2)

    
    #--------------------------------------  MAIN FUNCTIONS STARTS HERE  ------------------------------------#
    #--------------------------------------  END OF SOIL CONSTRAINTS  ---------------------------------------#


        






        

    


    



