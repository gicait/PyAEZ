"""
PyAEZ version 3.0.0 (June 2023)
Soil Constraints
2016: N. Lakmal Deshapriya
2023: Swun Wunna Htet

Modifications
1.  All reduction factors will be externally imported from excel sheets instead of providing
    python scripts.
2.  All reduction factors from excel sheets are recorded as python dictionaries. Algorithm will be the same as
    previous version. But the access of variables will be heavily depending on pandas incorporation and dictionaries.
3.  Soil evaluation from HWSD v1.2 and v2.0 will be added as options for users to decided for soil constraint analysis.
4.  In SMU's soil properties, if no values are provided, the default edaphic suitability rating will be default to 100.

"""
import numpy as np
import pandas as pd
import math

class SoilConstraints(object):

    def __init__(self, method = 1):

        """
        Select which soil evaluation method to apply soil constraints.

        Args:
            method (int): 1 (soil evaluation method from HWSD v1.2, two-soil layer method)
                          2 (soil evaulation method from HWSD v2.0, seven-soil depth classes method) (Default)
        """
        if method == 1:
            self.method = 1
            print('Soil Evaluation Method from HWSD v1.2, two-soil layer method, is selected.')
        elif method ==2:
            self.method = 2
            print('Soil Evaluation Method from HWSD v2.0, seven-soil layer method is selected')
        else:
            print('Wrong input method detected. Please revise it again.')
        
    
    def calculateSoilQualitiesI(self, irr_or_rain, soil_characteristics_path):
        """
        Intermediate function.
        
        Applying HWSD v1.2 soil evaluation method (2-layer approach).
        
        Args:
            irr_or_rain (String): Irrigated (I) or Rainfed (R)
            soil_characteristics_path (String): File path of SMU's soil characteristics
        """
        # reading soil properties from excel sheet
        main_df = pd.read_excel(soil_characteristics_path, sheet_name= None)
        topsoil_df = main_df['D1']
        subsoil_df = main_df['D2']

        # check if two dataframes have same number of SMUs, if not raise error.
        if topsoil_df.shape != subsoil_df.shape:
            raise Exception(r'Please recheck the number of entries of top-soil and sub soil excel sheets')

        self.SMU = topsoil_df['CODE']
        # zero array of all individual soil qualities for each self.SMU
        
        # 1st =  7 for 7 soil layers to evaluate, 2nd = self.SMUs, 3rd = Soil Qualities
        SQ_array = np.zeros((2, self.SMU.shape[0] ,7))


        for i in range(SQ_array.shape[0]):

            if i==0:
                df = topsoil_df.copy()
                top_sub = 'top'
            else:
                df = subsoil_df.copy()
                top_sub = 'sub'
            
            for j in range(self.SMU.shape[0]):

                t_code = df.loc[df['CODE'] == self.SMU[j]]

                # SQ1 calculation
                SQ_array[i,j,0] = self.soil_qty_1(TXT_val= t_code['TXT'].iloc[0], OC_val = t_code['OC'].iloc[0], pH_val=t_code['pH'].iloc[0],
                                        TEB_val= t_code['TEB'].iloc[0], condition= irr_or_rain, top_sub= top_sub)
            
                # SQ2 calculation
                SQ_array[i,j,1] = self.soil_qty_2(TXT_val= t_code['TXT'].iloc[0], BS_val=t_code['BS'].iloc[0], CECclay_val=t_code['CEC_clay'].iloc[0], 
                                        CECsoil_val= t_code['CEC_soil'].iloc[0], pH_val=t_code['pH'].iloc[0], condition = irr_or_rain, top_sub= top_sub)
            
                # SQ3 calculation
                SQ_array[i,j,2] = self.soil_qty_3(RSD_val=t_code['RSD'].iloc[0], SPR_val=t_code['SPR'].iloc[0], SPH_val=t_code['SPH'].iloc[0], 
                                        OSD_val=t_code['OSD'].iloc[0], condition= irr_or_rain)

                # SQ4 calculation
                SQ_array[i,j,3] = self.soil_qty_4(DRG_val= t_code['DRG'].iloc[0], SPH_val=t_code['SPH'].iloc[0], condition= irr_or_rain)


                # SQ5 calculation
                SQ_array[i,j,4] = self.soil_qty_5(ESP_val=t_code['ESP'].iloc[0], EC_val=t_code['EC'].iloc[0], SPH_val=t_code['SPH'].iloc[0], condition = irr_or_rain)

                # SQ6 calculation
                SQ_array[i,j,5] = self.soil_qty_6(CCB_val=t_code['CCB'].iloc[0], GYP_val=t_code['GYP'].iloc[0], SPH_val=t_code['SPH'].iloc[0], condition=irr_or_rain)
                
                # SQ7 calculation
                SQ_array[i,j,6] = self.soil_qty_7(RSD_val=t_code['RSD'].iloc[0], GRC_val=t_code['GRC'].iloc[0], SPH_val=t_code['SPH'].iloc[0], TXT_val=t_code['TXT'].iloc[0], 
                                        VSP_val=t_code['VSP'].iloc[0], condition = irr_or_rain)

        self.SQ_array = np.mean(SQ_array, axis = 0) # SQ1 mean

        self.SQ_array_pd = pd.DataFrame({'SMU': self.SMU, 
                                         'SQ1':self.SQ_array[:,0],
                                         'SQ2':self.SQ_array[:,1],
                                         'SQ3':self.SQ_array[:,2],
                                         'SQ4':self.SQ_array[:,3],
                                         'SQ5':self.SQ_array[:,4],
                                         'SQ6':self.SQ_array[:,5],
                                         'SQ7':self.SQ_array[:,6]})
        
    def calculateSoilQualitiesII(self, irr_or_rain, soil_characteristics_path):
        """
        Intermediate function.
        
        Applying HWSD v2.0 soil evaluation method (7-layer approach).
        
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
                SQ_array[i,j,1] = self.soil_qty_2(TXT_val= t_code['TXT'].iloc[0], BS_val=t_code['BS'].iloc[0], CECclay_val=t_code['CEC_clay'].iloc[0], 
                                        CECsoil_val= t_code['CEC_soil'].iloc[0], pH_val=t_code['pH'].iloc[0], condition = irr_or_rain, top_sub= top_sub)
            
                # SQ3 calculation
                SQ_array[i,j,2] = self.soil_qty_3(RSD_val=t_code['RSD'].iloc[0], SPR_val=t_code['SPR'].iloc[0], SPH_val=t_code['SPH'].iloc[0], 
                                        OSD_val=t_code['OSD'].iloc[0], condition= irr_or_rain)

                # SQ4 calculation
                SQ_array[i,j,3] = self.soil_qty_4(DRG_val= t_code['DRG'].iloc[0], SPH_val=t_code['SPH'].iloc[0], condition= irr_or_rain)


                # SQ5 calculation
                SQ_array[i,j,4] = self.soil_qty_5(ESP_val=t_code['ESP'].iloc[0], EC_val=t_code['EC'].iloc[0], SPH_val=t_code['SPH'].iloc[0], condition = irr_or_rain)

                # SQ6 calculation
                SQ_array[i,j,5] = self.soil_qty_6(CCB_val=t_code['CCB'].iloc[0], GYP_val=t_code['GYP'].iloc[0], SPH_val=t_code['SPH'].iloc[0], condition=irr_or_rain)
                
                # SQ7 calculation
                SQ_array[i,j,6] = self.soil_qty_7(RSD_val=t_code['RSD'].iloc[0], GRC_val=t_code['GRC'].iloc[0], SPH_val=t_code['SPH'].iloc[0], TXT_val=t_code['TXT'].iloc[0], 
                                        VSP_val=t_code['VSP'].iloc[0], condition = irr_or_rain)

        SQ_top = SQ_array[0,:,:]
        SQ_sub = np.mean(SQ_array[1:, :,:], axis = 0)
        self.SQ_array = np.mean([SQ_top, SQ_sub], axis = 0)

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
        
    # SQ1: Nutrient availability
    def soil_qty_1(self, TXT_val, OC_val, pH_val, TEB_val, condition, top_sub):

        if condition == 'I':
            para = self.SQ1_irr.copy()
        else:
            para = self.SQ1_rain.copy()
        
        TXT_intp = 100 if pd.isna(TXT_val) else para['TXT_fct'][np.where(para['TXT_val'] == TXT_val)[0][0]]

        pH_intp = 100 if math.isnan(pH_val) else np.interp(pH_val, para['pH_val'], para['pH_fct'])
        OC_intp  = 100 if math.isnan(OC_val) else np.interp(OC_val , para['OC_val'], para['OC_fct'])
        TEB_intp = 100 if math.isnan(TEB_val) else np.interp(TEB_val, para['TEB_val'], para['TEB_fct'])


        if top_sub == 'top':
            min_factor = np.min([TXT_intp, pH_intp, OC_intp, TEB_intp])
            summation = np.sum([TXT_intp, pH_intp, OC_intp, TEB_intp]) - min_factor
            final_factor = (min_factor + (summation/3))/2
        else:
            min_factor = np.min([TXT_intp, pH_intp, TEB_intp])
            summation = np.sum([TXT_intp, pH_intp, TEB_intp]) - min_factor
            final_factor = (min_factor + (summation/2))/2
        
        return final_factor
    
    # SQ2: Nutrient retention capacity
    def soil_qty_2(self, TXT_val, BS_val, CECclay_val, CECsoil_val, pH_val, condition, top_sub):
        
        if condition == 'I':
            para = self.SQ2_irr.copy()
        else:
            para = self.SQ2_rain.copy()

        TXT_intp = 100 if pd.isna(TXT_val) else para['TXT_fct'][np.where(para['TXT_val'] == TXT_val)[0][0]]

        BS_intp = 100 if math.isnan(BS_val) else np.interp(BS_val, para['BS_val'], para['BS_fct'])
        CECclay_intp = 100 if math.isnan(CECclay_val) else np.interp(CECclay_val, para['CECclay_val'], para['CECclay_fct'])
        CECsoil_intp = 100 if math.isnan(CECclay_val) else np.interp(CECsoil_val, para['CECsoil_val'], para['CECsoil_fct'])
        pH_intp = 100 if math.isnan(pH_val) else np.interp(pH_val, para['pH_val'], para['pH_fct'])


        if top_sub == 'top':
            min_factor = np.min([TXT_intp, BS_intp, CECsoil_intp])
            summation = np.sum([TXT_intp, BS_intp, CECsoil_intp]) - min_factor
            final_factor = (min_factor + (summation/2))/2
        else:
            min_factor = np.min([TXT_intp, BS_intp, CECclay_intp, pH_intp])
            summation = np.sum([TXT_intp, BS_intp, CECclay_intp, pH_intp]) - min_factor
            final_factor = (min_factor + (summation/3))/2

        return final_factor

    # SQ3: Rooting Conditions
    # Note: SQ3 calculation procedure is the same, regardless of topsoil or subsoil
    def soil_qty_3(self, RSD_val, SPR_val, SPH_val, OSD_val, condition):

        if condition == 'I':
            para = self.SQ3_irr.copy()
        else:
            para = self.SQ3_rain.copy()

        RSD_intp = 100 if math.isnan(RSD_val) else np.interp(RSD_val, para['RSD_val'], para['RSD_fct'])
        SPR_intp = 100 if math.isnan(SPR_val) else np.interp(SPR_val, para['SPR_val'], para['SPR_fct'])
        SPH_intp = 100 if pd.isna(SPH_val) else  para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]
        OSD_intp = 100 if math.isnan(OSD_val) else np.interp(OSD_val, para['OSD_val'], para['OSD_fct'])

        final_factor = RSD_intp * np.min([SPR_intp, SPH_intp, OSD_intp])

        return final_factor
    
    # SQ4: Oxygen availability
    def soil_qty_4(self, DRG_val, SPH_val, condition):
        
        if condition == 'I':
            para = self.SQ4_irr.copy()
        else:
            para = self.SQ4_rain.copy()

        DRG_intp = 100 if pd.isna(DRG_val) else para['DRG_fct'][np.where(para['DRG_val'] == DRG_val)[0][0]]
        SPH_intp = 100 if pd.isna(SPH_val) else para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]


        final_factor = np.min([DRG_intp, SPH_intp])

        return final_factor
    
    # SQ5: Presenceo f salinity and sodicity
    def soil_qty_5(self, ESP_val, EC_val, SPH_val, condition):

        if condition == 'I':
            para = self.SQ5_irr.copy()
        else:
            para = self.SQ5_rain.copy()

        ESP_intp = 100 if np.isnan(ESP_val) else np.interp(ESP_val, para['ESP_val'], para['ESP_fct'])
        EC_intp = 100 if np.isnan(EC_val) else np.interp(EC_val, para['EC_val'], para['EC_fct'])
        SPH_intp = 100 if pd.isna(SPH_val) else para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]


        final_factor = np.min([ESP_intp*EC_intp, SPH_intp])

        return final_factor
    
    # SQ6: Presence of lime and gypsum
    def soil_qty_6(self, CCB_val, GYP_val, SPH_val, condition):

        if condition == 'I':
            para = self.SQ6_irr.copy()
        else:
            para = self.SQ6_rain.copy()

        CCB_intp = 100 if np.isnan(CCB_val) else np.interp(CCB_val, para['CCB_val'], para['CCB_fct'])
        GYP_intp = 100 if np.isnan(GYP_val) else np.interp(GYP_val, para['GYP_val'], para['GYP_fct'])
        SPH_intp = 100 if pd.isna(SPH_val) else para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]

        final_factor = np.min([CCB_intp*GYP_intp, SPH_intp])

        return final_factor

    # SQ7:
    def soil_qty_7(self, RSD_val, GRC_val, SPH_val, TXT_val, VSP_val, condition):

        if condition == 'I':
            para = self.SQ7_irr.copy()
        else:
            para = self.SQ7_rain.copy()

        RSD_intp = 100 if np.isnan(RSD_val) else np.interp(RSD_val, para['RSD_val'], para['RSD_fct'])
        GRC_intp = 100 if np.isnan(GRC_val) else np.interp(GRC_val, para['GRC_val'], para['GRC_fct'])
        SPH_intp = 100 if pd.isna(SPH_val) else para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]
        TXT_intp = 100 if pd.isna(TXT_val) else para['TXT_fct'][np.where(para['TXT_val'] == TXT_val)[0][0]]
        VSP_intp = 100 if np.isnan(VSP_val) else np.interp(VSP_val, para['VSP_val'], para['VSP_fct'])

        min_factor = np.min([RSD_intp, GRC_intp, SPH_intp, TXT_intp, VSP_intp])
        summation = np.sum([RSD_intp, GRC_intp, SPH_intp, TXT_intp, VSP_intp]) - min_factor
        final_factor = (min_factor + (summation/4))/2

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
        
        # SQ 1: Nutrient Availability (4 parameters x 2)
        self.SQ1_rain ={
        'TXT_val' : (rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'TXT_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'TXT_fct':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'TXT_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float), # numerical
        'OC_val':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'OC_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'OC_fct':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'OC_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_val':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'pH_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_fct':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'pH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'TEB_val':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'TEB_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'TEB_fct':(rain_df['SQ1'].loc[rain_df['SQ1'][0] == 'TEB_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)# numerical
            }
        
        self.SQ1_irr =  {
        'TXT_val' : (irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'TXT_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'TXT_fct':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'TXT_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float), # numerical
        'OC_val':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'OC_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'OC_fct':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'OC_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_val':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'pH_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_fct':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'pH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'TEB_val':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'TEB_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'TEB_fct':(irr_df['SQ1'].loc[irr_df['SQ1'][0] == 'TEB_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)# numerical
            }
        # SQ1 ok
        
        # SQ 2: Nutrient Retention Capacity (5 parameters x 2 rows)
        self.SQ2_rain ={
        'TXT_val' : (rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'TXT_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'TXT_fct':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'TXT_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'BS_val':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'BS_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'BS_fct':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'BS_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CECsoil_val':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'CECsoil_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CECsoil_fct':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'CECsoil_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'pH_val':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'pH_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_fct':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'pH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'CECclay_val':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'CECclay_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CECclay_fct':(rain_df['SQ2'].loc[rain_df['SQ2'][0] == 'CECclay_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
            }
        
        self.SQ2_irr =  {
        'TXT_val' : (irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'TXT_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'TXT_fct':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'TXT_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'BS_val':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'BS_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'BS_fct':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'BS_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CECsoil_val':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'CECsoil_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CECsoil_fct':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'CECsoil_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'pH_val':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'pH_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'pH_fct':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'pH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),# numerical
        'CECclay_val':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'CECclay_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CECclay_fct':(irr_df['SQ2'].loc[irr_df['SQ2'][0] == 'CECclay_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
            }
        
        # SQ 3: Rooting Conditions (4 parameters x 2)
        self.SQ3_rain ={
        'RSD_val' : (rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'RSD_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'RSD_fct':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'RSD_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'OSD_val':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'OSD_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'OSD_fct':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'OSD_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPR_val':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'SPR_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPR_fct':(rain_df['SQ3'].loc[rain_df['SQ3'][0] == 'SPR_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        }
        
        self.SQ3_irr =  {
        'RSD_val' : (irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'RSD_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'RSD_fct':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'RSD_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'OSD_val':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'OSD_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'OSD_fct':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'OSD_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPR_val':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'SPR_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPR_fct':(irr_df['SQ3'].loc[irr_df['SQ3'][0] == 'SPR_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        }
        
        # SQ 4: Oxygen Availability (2 parameters x 2)
        self.SQ4_rain ={
        'DRG_val' : (rain_df['SQ4'].loc[rain_df['SQ4'][0] == 'DRG_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'DRG_fct':(rain_df['SQ4'].loc[rain_df['SQ4'][0] == 'DRG_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(rain_df['SQ4'].loc[rain_df['SQ4'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(rain_df['SQ4'].loc[rain_df['SQ4'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        }
        
        self.SQ4_irr =  {
        'DRG_val' : (irr_df['SQ4'].loc[irr_df['SQ4'][0] == 'DRG_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'DRG_fct':(irr_df['SQ4'].loc[irr_df['SQ4'][0] == 'DRG_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(irr_df['SQ4'].loc[irr_df['SQ4'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(irr_df['SQ4'].loc[irr_df['SQ4'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        }

        # SQ 5: Presence of Salinity and Sodicity (3 parameters x 2)
        self.SQ5_rain ={
        'ESP_val' : (rain_df['SQ5'].loc[rain_df['SQ5'][0] == 'ESP_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'ESP_fct':(rain_df['SQ5'].loc[rain_df['SQ5'][0] == 'ESP_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'EC_val':(rain_df['SQ5'].loc[rain_df['SQ5'][0] == 'EC_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'EC_fct':(rain_df['SQ5'].loc[rain_df['SQ5'][0] == 'EC_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(rain_df['SQ5'].loc[rain_df['SQ5'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(rain_df['SQ5'].loc[rain_df['SQ5'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)
        }
        
        self.SQ5_irr =  {
        'ESP_val' : (irr_df['SQ5'].loc[irr_df['SQ5'][0] == 'ESP_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'ESP_fct':(irr_df['SQ5'].loc[irr_df['SQ5'][0] == 'ESP_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'EC_val':(irr_df['SQ5'].loc[irr_df['SQ5'][0] == 'EC_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'EC_fct':(irr_df['SQ5'].loc[irr_df['SQ5'][0] == 'EC_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(irr_df['SQ5'].loc[irr_df['SQ5'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(irr_df['SQ5'].loc[irr_df['SQ5'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)
        }

        # SQ 6: Presence of Lime and Gypsum (3 parameters x 2)
        self.SQ6_rain ={
        'CCB_val' : (rain_df['SQ6'].loc[rain_df['SQ6'][0] == 'CCB_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CCB_fct':(rain_df['SQ6'].loc[rain_df['SQ6'][0] == 'CCB_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GYP_val':(rain_df['SQ6'].loc[rain_df['SQ6'][0] == 'GYP_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GYP_fct':(rain_df['SQ6'].loc[rain_df['SQ6'][0] == 'GYP_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(rain_df['SQ6'].loc[rain_df['SQ6'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(rain_df['SQ6'].loc[rain_df['SQ6'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)
            }
        
        self.SQ6_irr =  {
        'CCB_val' : (irr_df['SQ6'].loc[irr_df['SQ6'][0] == 'CCB_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'CCB_fct':(irr_df['SQ6'].loc[irr_df['SQ6'][0] == 'CCB_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GYP_val':(irr_df['SQ6'].loc[irr_df['SQ6'][0] == 'GYP_val']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'GYP_fct':(irr_df['SQ6'].loc[irr_df['SQ6'][0] == 'GYP_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float),
        'SPH_val':(irr_df['SQ6'].loc[irr_df['SQ6'][0] == 'SPH_val']).dropna(axis = 1).to_numpy()[0,1:], # string
        'SPH_fct':(irr_df['SQ6'].loc[irr_df['SQ6'][0] == 'SPH_fct']).dropna(axis = 1).to_numpy()[0,1:].astype(float)
            }
        
        # SQ 7: Workability (5 parameters x 2)
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
    

    def calculateSoilQualities(self, irr_or_rain, soil_characteristics_path):
        """
        Calculate the Soil Qualities for each SMU in soil map using soil characteristics
        from either HWSD v1.2 method or HWSD v2.0 method from previous setting.
        
        Args:
            soil_characteristics_path (String): I for Irrigated, R for Rainfed
            topsoil_path (String): file-path of top-soil characteristics excel sheet(xlsx format)
            subsoil_paht (String): file-path of sub-soil characteristics excel sheet (xlsx format)
        
        Return:
            None."""
        
        if self.method == 1:
            self.calculateSoilQualitiesI(irr_or_rain= irr_or_rain, soil_characteristics_path= soil_characteristics_path)
        else:
            self.calculateSoilQualitiesII(irr_or_rain= irr_or_rain, soil_characteristics_path= soil_characteristics_path)

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
                min_factor = np.min([self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]])
                summation = np.sum([np.sum([self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]])]) - min_factor

                # fsq = (min_factor + (np.sum([self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]]) - min_factor)/3)/2
                fsq = (min_factor + (summation/3))/2

                self.SR[i] = self.SQ_array[i,0] * self.SQ_array[i,2] * fsq

            elif input_level == 'I':
                min_factor = np.min([self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]])
                summation = np.sum([self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]]) - min_factor

                # fsq = (min_factor + (np.sum([self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]]) - min_factor)/3)/2
                fsq = (min_factor + (summation/3))/2

                self.SR[i] = 0.5 * (self.SQ_array[i,0]+self.SQ_array[i,1]) * self.SQ_array[i,2] * fsq

            elif input_level == 'H':
                min_factor = np.min([self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]])
                summation = np.sum([self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]]) - min_factor
                # fsq = (min_factor + (np.sum([self.SQ_array[i,3], self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6]]) - min_factor)/3)/2

                fsq = (min_factor + (summation/3))/2
                self.SR[i] = self.SQ_array[i,1] * self.SQ_array[i,2] * fsq
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
            yield_final[temp_idx] = yield_in[temp_idx] * self.SR[i1]

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
        return self.soilsuit_map
    
    #--------------------------------------  MAIN FUNCTIONS STARTS HERE  ------------------------------------#
    #--------------------------------------  END OF SOIL CONSTRAINTS  ---------------------------------------#


        






        

    


    



