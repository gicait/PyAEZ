"""
PyAEZ
Written by N. Lakmal Deshapriya
"""

import numpy as np
import csv

from . import ALL_REDUCTION_FACTORS_IRR as crop_P_IRR
from . import ALL_REDUCTION_FACTORS_RAIN as crop_P_RAIN

class SoilConstraints(object):

    '''functions for soil qualities'''

    def soil_qty_1_top(self, TXT_val, OC_val, pH_val, TEB_val):

        TXT_intp = self.crop_P.TXT1_factor[self.crop_P.TXT1_value.index(TXT_val)]/100
        pH_intp = np.interp(pH_val, self.crop_P.pH_value, self.crop_P.pH_factor)/100
        OC_intp = np.interp(OC_val, self.crop_P.OC_value, self.crop_P.OC_factor)/100
        TEB_intp = np.interp(TEB_val, self.crop_P.TEB_value, self.crop_P.TEB_factor)/100

        min_factor = np.min([TXT_intp, pH_intp, OC_intp, TEB_intp])
        final_factor = (min_factor + (np.sum([TXT_intp, pH_intp, OC_intp, TEB_intp]) - min_factor)/3)/2

        return final_factor

    def soil_qty_1_sub(self, TXT_val, pH_val, TEB_val):

        TXT_intp = self.crop_P.TXT1_factor[self.crop_P.TXT1_value.index(TXT_val)]/100
        pH_intp = np.interp(pH_val, self.crop_P.pH_value, self.crop_P.pH_factor)/100
        TEB_intp = np.interp(TEB_val, self.crop_P.TEB_value, self.crop_P.TEB_factor)/100

        min_factor = np.min([TXT_intp, pH_intp, TEB_intp])
        final_factor = (min_factor + (np.sum([TXT_intp, pH_intp, TEB_intp]) - min_factor)/2)/2

        return final_factor

    def soil_qty_2_top(self, TXT_val, BS_val, CECsoil_val):

        TXT_intp = self.crop_P.TXT2_factor[self.crop_P.TXT2_value.index(TXT_val)]/100
        BS_intp = np.interp(BS_val, self.crop_P.BS_value, self.crop_P.BS_factor)/100
        CECsoil_intp = np.interp(CECsoil_val, self.crop_P.CECsoil_value, self.crop_P.CECsoil_factor)/100

        min_factor = np.min([TXT_intp, BS_intp, CECsoil_intp])
        final_factor = (min_factor + (np.sum([TXT_intp, BS_intp, CECsoil_intp]) - min_factor)/2)/2

        return final_factor

    def soil_qty_2_sub(self, TXT_val, BS_val, CECclay_val, pH_val):

        TXT_intp = self.crop_P.TXT2_factor[self.crop_P.TXT2_value.index(TXT_val)]/100
        BS_intp = np.interp(BS_val, self.crop_P.BS_value, self.crop_P.BS_factor)/100
        CECclay_intp = np.interp(CECclay_val, self.crop_P.CECclay_value, self.crop_P.CECclay_factor)/100
        pH_intp = np.interp(pH_val, self.crop_P.pH_value, self.crop_P.pH_factor)/100

        min_factor = np.min([TXT_intp, BS_intp, CECclay_intp, pH_intp])
        final_factor = (min_factor + (np.sum([TXT_intp, BS_intp, CECclay_intp, pH_intp]) - min_factor)/3)/2

        return final_factor

    def soil_qty_3_top_sub(self, RSD_val, SPR_val, SPH_val, OSD_val):

        RSD_intp = np.interp(RSD_val, self.crop_P.RSD_value, self.crop_P.RSD_factor)/100
        SPR_intp = np.interp(SPR_val, self.crop_P.SPR_value, self.crop_P.SPR_factor)/100
        SPH_intp = self.crop_P.SPH3_factor[self.crop_P.SPH3_value.index(SPH_val)]/100
        OSD_intp = np.interp(OSD_val, self.crop_P.OSD_value, self.crop_P.OSD_factor)/100

        final_factor = RSD_intp * np.min([SPR_intp, SPH_intp, OSD_intp])

        return final_factor

    def soil_qty_4_top_sub(self, DRG_val, SPH_val):

        DRG_intp = self.crop_P.DRG_factor[self.crop_P.DRG_value.index(DRG_val)]/100
        SPH_intp = self.crop_P.SPH4_factor[self.crop_P.SPH4_value.index(SPH_val)]/100

        final_factor = np.min([DRG_intp, SPH_intp])

        return final_factor

    def soil_qty_5_top_sub(self, ESP_val, EC_val, SPH_val):

        ESP_intp = np.interp(ESP_val, self.crop_P.ESP_value, self.crop_P.ESP_factor)/100
        EC_intp = np.interp(EC_val, self.crop_P.EC_value, self.crop_P.EC_factor)/100
        SPH_intp = self.crop_P.SPH5_factor[self.crop_P.SPH5_value.index(SPH_val)]/100

        final_factor = np.min([ESP_intp*EC_intp, SPH_intp])

        return final_factor

    def soil_qty_6_top_sub(self, CCB_val, GYP_val, SPH_val):

        CCB_intp = np.interp(CCB_val, self.crop_P.CCB_value, self.crop_P.CCB_factor)/100
        GYP_intp = np.interp(GYP_val, self.crop_P.GYP_value, self.crop_P.GYP_factor)/100
        SPH_intp = self.crop_P.SPH6_factor[self.crop_P.SPH6_value.index(SPH_val)]/100

        final_factor = np.min([CCB_intp*GYP_intp, SPH_intp])

        return final_factor

    def soil_qty_7_top_sub(self, RSD_val, GRC_val, SPH_val, TXT_val, VSP_val):

        RSD_intp = np.interp(RSD_val, self.crop_P.RSD_value, self.crop_P.RSD_factor)/100
        GRC_intp = np.interp(GRC_val, self.crop_P.GRC_value, self.crop_P.GRC_factor)/100
        SPH_intp = self.crop_P.SPH7_factor[self.crop_P.SPH7_value.index(SPH_val)]/100
        TXT_intp = self.crop_P.TXT7_factor[self.crop_P.TXT7_value.index(TXT_val)]/100
        VSP_intp = np.interp(VSP_val, self.crop_P.VSP_value, self.crop_P.VSP_factor)/100

        min_factor = np.min([RSD_intp, GRC_intp, SPH_intp, TXT_intp, VSP_intp])
        final_factor = (min_factor + (np.sum([RSD_intp, GRC_intp, SPH_intp, TXT_intp, VSP_intp]) - min_factor)/4)/2

        return final_factor

    def calculateSoilQualities(self, irr_or_rain, topsoil_path='./sample_data/input/soil_characteristics_topsoil.csv', subsoil_path='./sample_data/input/soil_characteristics_subsoil.csv'):

        if irr_or_rain == 'I':
            self.crop_P = crop_P_IRR
        elif irr_or_rain == 'R':
            self.crop_P = crop_P_RAIN

        # getting number of soil units
        file_temp = open(topsoil_path)
        numline = len(file_temp.readlines())

        self.sq_mat = np.zeros((numline-1,1+7)) # first column for soil unit code and other columns for SQs

        with open(topsoil_path) as csv_top:
            csv_sub = open(subsoil_path)
            top_reader = csv.reader(csv_top, delimiter=',')
            sub_reader = csv.reader(csv_sub, delimiter=',')

            row_count = 0
            name_list = []
            for row_top in top_reader:
                row_sub = next(sub_reader)

                row_count = row_count + 1
                if row_count == 1:
                    name_list = row_top
                    continue

                self.sq_mat[row_count-2, 0] = float(row_top[0])

                '''SQ 1'''

                TXT_val = row_top[name_list.index('TXT')]
                OC_val = float(row_top[name_list.index('OC')])
                pH_val = float(row_top[name_list.index('pH')])
                TEB_val = float(row_top[name_list.index('TEB')])
                sq1_top = self.soil_qty_1_top(TXT_val, OC_val, pH_val, TEB_val)

                TXT_val = row_sub[name_list.index('TXT')]
                pH_val = float(row_sub[name_list.index('pH')])
                TEB_val = float(row_sub[name_list.index('TEB')])
                sq1_sub = self.soil_qty_1_sub(TXT_val, pH_val, TEB_val)

                sq1 = (sq1_top + sq1_sub)/2

                self.sq_mat[row_count-2, 1] = sq1

                '''SQ 2'''

                TXT_val = row_top[name_list.index('TXT')]
                BS_val = float(row_top[name_list.index('BS')])
                CECsoil_val = float(row_top[name_list.index('CEC_soil')])
                sq2_top = self.soil_qty_2_top(TXT_val, BS_val, CECsoil_val)

                TXT_val = row_sub[name_list.index('TXT')]
                BS_val = float(row_sub[name_list.index('BS')])
                CECclay_val = float(row_sub[name_list.index('CEC_soil')])
                pH_val = float(row_sub[name_list.index('pH')])
                sq2_sub = self.soil_qty_2_sub(TXT_val, BS_val, CECclay_val, pH_val)

                sq2 = (sq2_top + sq2_sub)/2

                self.sq_mat[row_count-2, 2] = sq2

                '''SQ 3'''

                RSD_val = float(row_top[name_list.index('RSD')])
                SPR_val = float(row_top[name_list.index('SPR')])
                SPH_val = row_top[name_list.index('SPH')]
                OSD_val = float(row_top[name_list.index('OSD')])
                sq3_top = self.soil_qty_3_top_sub(RSD_val, SPR_val, SPH_val, OSD_val)

                RSD_val = float(row_sub[name_list.index('RSD')])
                SPR_val = float(row_sub[name_list.index('SPR')])
                SPH_val = row_sub[name_list.index('SPH')]
                OSD_val = float(row_sub[name_list.index('OSD')])
                sq3_sub = self.soil_qty_3_top_sub(RSD_val, SPR_val, SPH_val, OSD_val)

                sq3 = (sq3_top + sq3_sub)/2

                self.sq_mat[row_count-2, 3] = sq3

                '''SQ 4'''

                DRG_val = row_top[name_list.index('DRG')]
                SPH_val = row_top[name_list.index('SPH')]
                sq4_top = self.soil_qty_4_top_sub(DRG_val, SPH_val)

                DRG_val = row_sub[name_list.index('DRG')]
                SPH_val = row_sub[name_list.index('SPH')]
                sq4_sub = self.soil_qty_4_top_sub(DRG_val, SPH_val)

                sq4 = (sq4_top + sq4_sub)/2

                self.sq_mat[row_count-2, 4] = sq4

                '''SQ 5'''

                ESP_val = float(row_top[name_list.index('ESP')])
                EC_val = float(row_top[name_list.index('EC')])
                SPH_val = row_top[name_list.index('SPH')]
                sq5_top = self.soil_qty_5_top_sub(ESP_val, EC_val, SPH_val)

                ESP_val = float(row_sub[name_list.index('ESP')])
                EC_val = float(row_sub[name_list.index('EC')])
                SPH_val = row_sub[name_list.index('SPH')]
                sq5_sub = self.soil_qty_5_top_sub(ESP_val, EC_val, SPH_val)

                sq5 = (sq5_top + sq5_sub)/2

                self.sq_mat[row_count-2, 5] = sq5

                '''SQ 6'''

                CCB_val = float(row_top[name_list.index('CCB')])
                GYP_val = float(row_top[name_list.index('GYP')])
                SPH_val = row_top[name_list.index('SPH')]
                sq6_top = self.soil_qty_6_top_sub(CCB_val, GYP_val, SPH_val)

                CCB_val = float(row_sub[name_list.index('CCB')])
                GYP_val = float(row_sub[name_list.index('GYP')])
                SPH_val = row_sub[name_list.index('SPH')]
                sq6_sub = self.soil_qty_6_top_sub(CCB_val, GYP_val, SPH_val)

                sq6 = (sq6_top + sq6_sub)/2

                self.sq_mat[row_count-2, 6] = sq6

                '''SQ 7'''

                RSD_val = float(row_top[name_list.index('RSD')])
                GRC_val = float(row_top[name_list.index('GRC')])
                SPH_val = row_top[name_list.index('SPH')]
                TXT_val = row_top[name_list.index('TXT')]
                VSP_val = float(row_top[name_list.index('VSP')])
                sq7_top = self.soil_qty_7_top_sub(RSD_val, GRC_val, SPH_val, TXT_val, VSP_val)

                RSD_val = float(row_sub[name_list.index('RSD')])
                GRC_val = float(row_sub[name_list.index('GRC')])
                SPH_val = row_sub[name_list.index('SPH')]
                TXT_val = row_sub[name_list.index('TXT')]
                VSP_val = float(row_sub[name_list.index('VSP')])
                sq7_sub = self.soil_qty_7_top_sub(RSD_val, GRC_val, SPH_val, TXT_val, VSP_val)

                sq7 = (sq7_top + sq7_sub)/2

                self.sq_mat[row_count-2, 7] = sq7

    def calculateSoilRatings(self, input_level):

        self.SR = np.zeros((self.sq_mat.shape[0],2)) # first column for soil unit code and other column for SR
        self.SR[:,0] = self.sq_mat[:,0] # adding soil unit code

        for i1 in range(0, self.sq_mat.shape[0]):

            if input_level == 'L':
                min_factor = np.min([self.sq_mat[i1][4], self.sq_mat[i1][5], self.sq_mat[i1][6], self.sq_mat[i1][7]])
                fsq = (min_factor + (np.sum([self.sq_mat[i1][4], self.sq_mat[i1][5], self.sq_mat[i1][6], self.sq_mat[i1][7]]) - min_factor)/3)/2
                self.SR[i1,1] = self.sq_mat[i1][1] * self.sq_mat[i1][3] * fsq
            elif input_level == 'I':
                min_factor = np.min([self.sq_mat[i1][4], self.sq_mat[i1][5], self.sq_mat[i1][6], self.sq_mat[i1][7]])
                fsq = (min_factor + (np.sum([self.sq_mat[i1][4], self.sq_mat[i1][5], self.sq_mat[i1][6], self.sq_mat[i1][7]]) - min_factor)/3)/2
                self.SR[i1,1] = 0.5 * (self.sq_mat[i1][1]+self.sq_mat[i1][2]) * self.sq_mat[i1][3] * fsq
            elif input_level == 'H':
                min_factor = np.min([self.sq_mat[i1][4], self.sq_mat[i1][5], self.sq_mat[i1][6], self.sq_mat[i1][7]])
                fsq = (min_factor + (np.sum([self.sq_mat[i1][4], self.sq_mat[i1][5], self.sq_mat[i1][6], self.sq_mat[i1][7]]) - min_factor)/3)/2
                self.SR[i1,1] = self.sq_mat[i1][2] * self.sq_mat[i1][3] * fsq
            else:
                print('Wrong Input Level !')

    def getSoilQualities(self):
        return self.sq_mat

    def getSoilRatings(self):
        return self.SR

    def applySoilConstraints(self, soil_map, yield_in):

        yield_final = np.copy(yield_in)

        for i1 in range(0, self.SR.shape[0]):
            temp_idx = soil_map==self.SR[i1,0]
            yield_final[temp_idx] = yield_in[temp_idx] * self.SR[i1,1]

        return yield_final
