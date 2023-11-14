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
"""
import numpy as np
import pandas as pd

class SoilConstraints(object):

    
    # Sub-routines for each aspect of soil quality calculation
    # All background calculations of seven soil qualities for subsoil and topsoil
    def soil_qty_1(self, TXT_val, OC_val, pH_val, TEB_val, condition, top_sub):

        if condition == 'I':
            para = self.SQ1_irr
        else:
            para = self.SQ1_rain
        
        TXT_intp = para['TXT_fct'][np.where(para['TXT_val'] == TXT_val)[0][0]]/100
        pH_intp = np.interp(pH_val, para['pH_val'], para['pH_fct'])/100
        OC_intp  = np.interp(OC_val , para['OC_val'], para['OC_fct'])/100
        TEB_intp = np.interp(TEB_val, para['TEB_val'], para['TEB_fct'])/100

        if top_sub == 'top':
            min_factor = np.min([TXT_intp, pH_intp, OC_intp, TEB_intp])
            final_factor = (min_factor + (np.sum([TXT_intp, pH_intp, OC_intp, TEB_intp]) - min_factor)/3)/2
        else:
            min_factor = np.min([TXT_intp, pH_intp, TEB_intp])
            final_factor = (min_factor + (np.sum([TXT_intp, pH_intp, TEB_intp]) - min_factor)/2)/2
        
        return final_factor
    
    def soil_qty_2(self, TXT_val, BS_val, CECclay_val, CECsoil_val, pH_val, condition, top_sub):
        
        if condition == 'I':
            para = self.SQ2_irr
        else:
            para = self.SQ2_rain

        TXT_intp = para['TXT_fct'][np.where(para['TXT_val'] == TXT_val)[0][0]]/100
        BS_intp = np.interp(BS_val, para['BS_val'], para['BS_fct'])/100
        CECclay_intp = np.interp(CECclay_val, para['CECclay_val'], para['CECclay_fct'])/100
        CECsoil_intp = np.interp(CECsoil_val, para['CECsoil_val'], para['CECsoil_fct'])/100
        pH_intp = np.interp(pH_val, para['pH_val'], para['pH_fct'])/100

        if top_sub == 'top':
            min_factor = np.min([TXT_intp, BS_intp, CECsoil_intp])
            final_factor = (min_factor + (np.sum([TXT_intp, BS_intp, CECsoil_intp]) - min_factor)/2)/2
        else:
            min_factor = np.min([TXT_intp, BS_intp, CECclay_intp, pH_intp])
            final_factor = (min_factor + (np.sum([TXT_intp, BS_intp, CECclay_intp, pH_intp]) - min_factor)/3)/2

        return final_factor

    # Note: SQ3 calculation procedure is the same, regardless of topsoil or subsoil
    def soil_qty_3(self, RSD_val, SPR_val, SPH_val, OSD_val, condition):

        if condition == 'I':
            para = self.SQ3_irr
        else:
            para = self.SQ3_rain

        RSD_intp = np.interp(RSD_val, para['RSD_val'], para['RSD_fct'])/100
        SPR_intp = np.interp(SPR_val, para['SPR_val'], para['SPR_fct'])/100
        SPH_intp = para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]/100
        OSD_intp = np.interp(OSD_val, para['OSD_val'], para['OSD_fct'])/100

        final_factor = RSD_intp * np.min([SPR_intp, SPH_intp, OSD_intp])

        return final_factor
    
    def soil_qty_4(self, DRG_val, SPH_val, condition):
        
        if condition == 'I':
            para = self.SQ4_irr
        else:
            para = self.SQ4_rain

        DRG_intp = para['DRG_fct'][np.where(para['DRG_val'] == DRG_val)[0][0]]/100
        SPH_intp = para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]/100

        final_factor = np.min([DRG_intp, SPH_intp])

        return final_factor
    
    def soil_qty_5(self, ESP_val, EC_val, SPH_val, condition):

        if condition == 'I':
            para = self.SQ5_irr
        else:
            para = self.SQ5_rain

        ESP_intp = np.interp(ESP_val, para['ESP_val'], para['ESP_fct'])/100
        EC_intp = np.interp(EC_val, para['EC_val'], para['EC_fct'])/100
        SPH_intp = para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]/100


        final_factor = np.min([ESP_intp*EC_intp, SPH_intp])

        return final_factor
    
    def soil_qty_6(self, CCB_val, GYP_val, SPH_val, condition):

        if condition == 'I':
            para = self.SQ6_irr
        else:
            para = self.SQ6_rain

        CCB_intp = np.interp(CCB_val, para['CCB_val'], para['CCB_fct'])/100
        GYP_intp = np.interp(GYP_val, para['GYP_val'], para['GYP_fct'])/100
        SPH_intp = para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]/100

        final_factor = np.min([CCB_intp*GYP_intp, SPH_intp])

        return final_factor

    def soil_qty_7(self, RSD_val, GRC_val, SPH_val, TXT_val, VSP_val, condition):

        if condition == 'I':
            para = self.SQ7_irr
        else:
            para = self.SQ7_rain

        RSD_intp = np.interp(RSD_val, para['RSD_val'], para['RSD_fct'])/100
        GRC_intp = np.interp(GRC_val, para['GRC_val'], para['GRC_fct'])/100
        SPH_intp = para['SPH_fct'][np.where(para['SPH_val'] == SPH_val)[0][0]]/100
        TXT_intp = para['TXT_fct'][np.where(para['TXT_val'] == TXT_val)[0][0]]/100
        VSP_intp = np.interp(VSP_val, para['VSP_val'], para['VSP_fct'])/100

        min_factor = np.min([RSD_intp, GRC_intp, SPH_intp, TXT_intp, VSP_intp])
        final_factor = (min_factor + (np.sum([RSD_intp, GRC_intp, SPH_intp, TXT_intp, VSP_intp]) - min_factor)/4)/2

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
    

    def calculateSoilQualities(self, irr_or_rain, topsoil_path, subsoil_path):
        
        # reading soil properties from excel sheet
        topsoil_df = pd.read_excel(topsoil_path)
        subsoil_df = pd.read_excel(subsoil_path)

        # check if two dataframes have same number of SMUs, if not raise error.
        if topsoil_df.shape != subsoil_df.shape:
            raise Exception(r'Please recheck the number of entries of top-soil and sub soil excel sheets')

        SMU = topsoil_df.CODE
        # zero array of all individual soil qualities for each SMU
        self.SQ_array = np.zeros((SMU.shape[0],8))
        self.SQ_array[:,0] = SMU

        for i in range(SMU.shape[0]):
            t_code = topsoil_df.loc[topsoil_df['CODE'] == SMU[i]]
            s_code = subsoil_df.loc[subsoil_df['CODE'] == SMU[i]]

            # SQ1 calculation
            # top-soil
            SQ1_t = self.soil_qty_1(TXT_val= t_code['TXT'].iloc[0], OC_val = t_code['OC'].iloc[0], pH_val=t_code['pH'].iloc[0],
                                    TEB_val= t_code['TEB'].iloc[0], condition= irr_or_rain, top_sub= 'top')
            # sub-soil
            SQ1_s = self.soil_qty_1(TXT_val= s_code['TXT'].iloc[0], OC_val = s_code['OC'].iloc[0], pH_val=s_code['pH'].iloc[0],
                                    TEB_val= s_code['TEB'].iloc[0], condition= irr_or_rain, top_sub= 'sub')
            
            SQ1 = (SQ1_t + SQ1_s)/2
            self.SQ_array[i,1] = SQ1

            # SQ2 calculation
            # top-soil
            SQ2_t = self.soil_qty_2(TXT_val= t_code['TXT'].iloc[0], BS_val=t_code['BS'].iloc[0], CECclay_val=t_code['CEC_clay'].iloc[0], 
                                    CECsoil_val= t_code['CEC_soil'].iloc[0], pH_val=t_code['pH'].iloc[0], condition = irr_or_rain, top_sub= 'top')
            
            # sub-soil
            SQ2_s = self.soil_qty_2(TXT_val= s_code['TXT'].iloc[0], BS_val=s_code['BS'].iloc[0], CECclay_val=s_code['CEC_clay'].iloc[0], 
                        CECsoil_val= s_code['CEC_soil'].iloc[0], pH_val=s_code['pH'].iloc[0], condition = irr_or_rain, top_sub= 'sub')
            
            SQ2 = (SQ2_t + SQ2_s)/2
            self.SQ_array[i,2] = SQ2

            # SQ3 calculation
            # top-soil
            SQ3_t = self.soil_qty_3(RSD_val=t_code['RSD'].iloc[0], SPR_val=t_code['SPR'].iloc[0], SPH_val=t_code['SPH'].iloc[0], 
                                    OSD_val=t_code['OSD'].iloc[0], condition= irr_or_rain)

            SQ3_s = self.soil_qty_3(RSD_val=s_code['RSD'].iloc[0], SPR_val=s_code['SPR'].iloc[0], SPH_val=s_code['SPH'].iloc[0], 
                        OSD_val=s_code['OSD'].iloc[0], condition= irr_or_rain)
            
            SQ3 = (SQ3_t + SQ3_s)/2

            self.SQ_array[i,3] = SQ3

            # SQ4 calculation
            # top-soil
            SQ4_t = self.soil_qty_4(DRG_val= t_code['DRG'].iloc[0], SPH_val=t_code['SPH'].iloc[0], condition= irr_or_rain)

            # sub-soil
            SQ4_s = self.soil_qty_4(DRG_val= s_code['DRG'].iloc[0], SPH_val=s_code['SPH'].iloc[0], condition= irr_or_rain)

            SQ4 = (SQ4_t + SQ4_s)/2

            self.SQ_array[i,4] = SQ4

            # SQ5 calculation
            # top-soil
            SQ5_t = self.soil_qty_5(ESP_val=t_code['ESP'].iloc[0], EC_val=t_code['EC'].iloc[0], SPH_val=t_code['SPH'].iloc[0], condition = irr_or_rain)
            # sub-soil
            SQ5_s = self.soil_qty_5(ESP_val=s_code['ESP'].iloc[0], EC_val=s_code['EC'].iloc[0], SPH_val=s_code['SPH'].iloc[0], condition = irr_or_rain)

            SQ5 = (SQ5_t + SQ5_s)/2

            self.SQ_array[i,5] = SQ5

            # SQ6 calculation
            # top-soil
            SQ6_t = self.soil_qty_6(CCB_val=t_code['CCB'].iloc[0], GYP_val=t_code['GYP'].iloc[0], SPH_val=t_code['SPH'].iloc[0], condition=irr_or_rain)
            
            # sub-soil
            SQ6_s = self.soil_qty_6(CCB_val=s_code['CCB'].iloc[0], GYP_val=s_code['GYP'].iloc[0], SPH_val=s_code['SPH'].iloc[0], condition=irr_or_rain)

            SQ6 = (SQ6_t + SQ6_s)/2

            self.SQ_array[i,6] = SQ6

            # SQ7 calculation
            # top-soil
            SQ7_t = self.soil_qty_7(RSD_val=t_code['RSD'].iloc[0], GRC_val=t_code['GRC'].iloc[0], SPH_val=t_code['SPH'].iloc[0], TXT_val=t_code['TXT'].iloc[0], 
                                    VSP_val=t_code['VSP'].iloc[0], condition = irr_or_rain)
            # sub-soil
            SQ7_s = self.soil_qty_7(RSD_val=s_code['RSD'].iloc[0], GRC_val=s_code['GRC'].iloc[0], SPH_val=s_code['SPH'].iloc[0], TXT_val=s_code['TXT'].iloc[0], 
                        VSP_val=s_code['VSP'].iloc[0], condition = irr_or_rain)
            
            SQ7 = (SQ7_t + SQ7_s)/2

            self.SQ_array[i,7] = SQ7
    
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

        self.SR = np.zeros((self.SQ_array.shape[0],2)) # first column for soil unit code and other column for SR
        self.SR[:,0] = self.SQ_array[:,0] # adding soil unit code

        for i in range(0, self.SQ_array.shape[0]):

            if input_level == 'L':
                min_factor = np.min([self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6], self.SQ_array[i,7]])
                fsq = (min_factor + (np.sum([self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6], self.SQ_array[i,7]]) - min_factor)/3)/2
                self.SR[i,1] = self.SQ_array[i,1] * self.SQ_array[i,3] * fsq
            elif input_level == 'I':
                min_factor = np.min([self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6], self.SQ_array[i,7]])
                fsq = (min_factor + (np.sum([self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6], self.SQ_array[i,7]]) - min_factor)/3)/2
                self.SR[i,1] = 0.5 * (self.SQ_array[i,1]+self.SQ_array[i,2]) * self.SQ_array[i,3] * fsq
            elif input_level == 'H':
                min_factor = np.min([self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6], self.SQ_array[i,7]])
                fsq = (min_factor + (np.sum([self.SQ_array[i,4], self.SQ_array[i,5], self.SQ_array[i,6], self.SQ_array[i,7]]) - min_factor)/3)/2
                self.SR[i,1] = self.SQ_array[i,2] * self.SQ_array[i,3] * fsq
            else:
                print('Wrong Input Level !')
    
    def getSoilQualities(self):
        """
        Obtain the calculated seven soil qualities for each soil mapping unit
        as 2-D NumPy array. Each column corresponds to a particular soil quality
        such a way as [SQ1, SQ2, SQ3, SQ4, SQ5, SQ6, SQ7].
        Args:
            None.
        Return:
            2-D numpy array."""
        return self.SQ_array

    def getSoilRatings(self):
        """
        Obtain the calculated soil-mapping unit specific soil suitability ratings.
        Soil ratings ranges from 0 (Not-Suitable) to 1 (Most Suitable)
        
        Args:
            None.
        Return:
            Soil Ratings: 2-D NumPy array.
        """
        return self.SR
    
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
            temp_idx = soil_map==self.SR[i1,0]
            self.soilsuit_map[temp_idx] = self.SR[i1,1]
            yield_final[temp_idx] = yield_in[temp_idx] * self.SR[i1,1]

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


        






        

    


    



