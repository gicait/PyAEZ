"""
PyAEZ
Written by N. Lakmal Deshapriya
"""

import numpy as np

import pandas as pd
from pandas.core.frame import DataFrame

import UtilitiesCalc 


import ALL_REDUCTION_FACTORS_IRR as crop_P_IRR
import ALL_REDUCTION_FACTORS_RAIN as crop_P_RAIN

class ClimaticConstraints(object):
    def Read_from_csv(self, path, irr_or_rain, Crop_name):
        xls = pd.ExcelFile(path)
        df =  pd.read_excel(xls, irr_or_rain)
        crop_df_index = df.index[df['Crop'] == Crop_name].tolist()
        crop_df = df.loc[df['Crop'] == Crop_name]
        self.lower = crop_df['lower'][crop_df_index].tolist()
        self.higher = crop_df['higher'][crop_df_index].tolist()
        self.b = crop_df['b'][crop_df_index].tolist()
        self.c = crop_df['c'][crop_df_index].tolist()
        self.d = crop_df['d'][crop_df_index].tolist()
        
        
    

    def applyClimaticConstraints(self, lgp_eq, yield_in, lgp):

        #find LGPagcusing the formula in the doc
        #compare the value with the csv and find reduction factor
        #find the reduction factor 
        height = lgp.shape[0]
        width = lgp.shape[1]
        #print(type(self.height), type(self.width))
        #self.f3 = np.array(self.height, self.width)
        self.f3 = np.empty(shape=(height,width))
        LGPagc = np.empty(shape=(height,width))
        final_yeild = np.empty(shape=(height, width))

        
        for i in range(height):
            for j in range(width):
                #calculating Lgpagc
                if lgp[i,j] <= 120:
                    LGPagc[i,j] = min (120, max(lgp[i,j],lgp_eq[i,j]))
                    #print(lgp_eq[i,j], lgp[i,j])
                elif lgp[i,j] >=120 and lgp[i,j]<=210:
                    LGPagc[i, j] = 120
                elif lgp[i,j] >= 210:
                    LGPagc[i,j] = max(210, min(lgp[i,j], lgp_eq[i, j]))
                else:
                    print("Exceptional Error Out of bound")
        #checking for rainfeed or irrigation
                
                #finding the reduction factor
                for k in range (len(self.lower)):
                    if self.lower[k] <= LGPagc[i,j] and self.higher[k] >= LGPagc[i,j] :
                        self.f3[i, j] = ((1-(self.b[k]/100))* (1-(self.c[k]/100)) * (1-(self.d[k]/100)))
                    elif LGPagc[i,j]< self.lower[0]:
                        print('out of bound, LGPagc is smaller than the lowest value in CSV ')
                    
                #final_yeild calculated      
                final_yeild[i,j] = yield_in[i,j] *self.f3[i, j] 
            #obj_utilities = UtilitiesCalc.UtilitiesCalc()
            #obj_utilities.saveRaster('D:/3. Py-AEZ/PyAEZ/sample_data/input/LGP.tif','D:/3. Py-AEZ/PyAEZsample_data/output/NB3/fc3.tif', f3)  
                
        return final_yeild   

    def getreductionfactor(self):
            return self.f3   
              
        

    def fcombined_rainfed(self, f1, f2, f3):
        self.f0 = (f1* f2*f3)/1000
        return self.f0
                    
    def fcombined_irrigate(self, f1, f3):
        self.fc0= (f1*f3)/1000
        return self.fc0
