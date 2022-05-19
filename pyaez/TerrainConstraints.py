"""
PyAEZ
Written by N. Lakmal Deshapriya
"""

import numpy as np

from  . import ALL_REDUCTION_FACTORS_IRR as crop_P_IRR
from  . import ALL_REDUCTION_FACTORS_RAIN as crop_P_RAIN

class TerrainConstraints(object):

    def setClimateTerrainData(self, precipitation, slope):
        self.prec_monthly = precipitation
        self.slope = slope # Percentage Slope

        self.im_height = slope.shape[0]
        self.im_width = slope.shape[1]

    def calculateFI(self):
        # calculation of Fournier index

        sum_Psquare = np.sum(np.square(self.prec_monthly), axis=2)
        sum_P = np.sum(self.prec_monthly, axis=2)

        self.FI = 12 * (sum_Psquare / sum_P)

    def getFI(self):
        # returning Fournier index

        return self.FI

    def applyTerrainConstraints(self, yield_in, irr_or_rain):

        if irr_or_rain == 'I':
            crop_P = crop_P_IRR
        elif irr_or_rain == 'R':
            crop_P = crop_P_RAIN

        yield_final = np.copy(yield_in)

        self.FI[np.isnan(self.FI)] = -99 # This suppresses warning with NaN values
        self.slope[np.isnan(self.slope)] = -99 # This suppresses warning with NaN values

        FI_count = -1
        for FI_cls1 in crop_P.FI_class:
            FI_count = FI_count + 1

            slope_count = -1
            for slope_cls1 in crop_P.Slope_class:
                slope_count = slope_count + 1

                FI_idx = np.logical_and(FI_cls1[0]<=self.FI, self.FI<=FI_cls1[1])
                slope_idx = np.logical_and(slope_cls1[0]<=self.slope, self.slope<=slope_cls1[1])
                temp_idx = np.logical_and(FI_idx, slope_idx)

                yield_final[temp_idx] = yield_in[temp_idx] * (crop_P.Terrain_factor[FI_count][slope_count] / 100)

        return yield_final
