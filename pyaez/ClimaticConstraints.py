"""
PyAEZ
Written by N. Lakmal Deshapriya
"""

import numpy as np

import ALL_REDUCTION_FACTORS_IRR as crop_P_IRR
import ALL_REDUCTION_FACTORS_RAIN as crop_P_RAIN

class ClimaticConstraints(object):

    def applyClimaticConstraints(self, lgp_eq, yield_in, irr_or_rain):

        if irr_or_rain == 'I':
            crop_P = crop_P_IRR
        elif irr_or_rain == 'R':
            crop_P = crop_P_RAIN

        class_break = np.array(crop_P.lgp_eq_class)
        reduction_fact = np.array(crop_P.lgp_eq_red_fr)

        yield_final = np.copy(yield_in)
        lgp_eq = np.round(lgp_eq)
        lgp_eq[np.isnan(lgp_eq)] = -99 # This suppresses warning with NaN values

        '''min is replaced by product based on GAEZ doc, still keeping commented min code also just in case'''
        # min_yield_fact = np.min(reduction_fact, axis=0) / 100
        min_yield_fact = np.prod(reduction_fact/100, axis=0)

        for class_num in range(class_break.shape[0]):
            temp_idx = np.logical_and(class_break[class_num,0]<=lgp_eq, lgp_eq<=class_break[class_num,1])
            yield_final[temp_idx] = yield_in[temp_idx] * min_yield_fact[class_num]

        return yield_final
