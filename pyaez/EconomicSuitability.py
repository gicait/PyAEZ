"""
PyAEZ version 2.2 (Dec 2023)
2020: Thaileng Thol
2023 (Dec): Swun Wunna Htet

"""

import numpy as np
from scipy import stats

class EconomicSuitability(object):

    def __init__(self):
        self.crop_name_list = []
        self.net_revenue_list = []
        self.revenue_min_list = []

    def addACrop(self, crop_name, crop_cost, crop_yield, farm_price, yield_map):
        """
        Adding a LUT's crop economic data into the module for economic analysis.

        Args:
            crop_name (String): Unique name of crop (Will be used later to extract output)
            crop_cost (1-D NumPy Array): Cost of production for each obtained yield in crop_yield variable. (Currency/ton or kg).
            crop_yield (1-D NumPy Array): Yield obtained from the production. Must correspond to crop_cost variable (ton or kg).
            farm_price (1-D NumPy Array): Historical crop price that farmers sell (Currency/ ton or kg).
            yield_map (2-D NumPy Array): Yield map of the selected LUT/crop. (kg/ha or ton/ha)
        
            Note: In some cases, unit conversion adjustment will be required.
        
        Return:
            None.

        """

        slope, intercept, r_value1, p_value, std_err = stats.linregress(crop_yield, crop_cost)

        '''Minimum attainable yield Calculation- break-even analysis'''
        y_min = intercept/(np.mean(farm_price)-slope)


        '''Attainable net revenue'''
        net_revenue = (yield_map - y_min) * np.mean(farm_price)
        net_revenue[net_revenue<0] = 0

        self.crop_name_list.append(crop_name)
        self.net_revenue_list.append(net_revenue)
        self.revenue_min_list.append(y_min * np.mean(farm_price))

        return None

    def getNetRevenue(self, crop_name):

        """
        Obtaining the net revenue from the select crop name.
        
        Args:
            crop_name (String): the crop name provided in previous function.
        
        Return:
            Net Revenue (2-D NumPy Array): net revenue (Unit: revenue/hectare)
        """
        return self.net_revenue_list[self.crop_name_list.index(crop_name)]

    def getClassifiedNetRevenue(self, crop_name):

        """Obtain the classified output of Net Attainable Revenue Map
        
        Args:
            crop_name (String): the crop name provided in previous function. """

        ''' Classification of Net Attainable Revenue map
        class 7 = very high = yields are more of the overall maximum yield,
        class 6 = high = yields between 63% and 75%,
        class 5 = good = yields between 50% and 63%,
        class 4 = medium = yields between 35% and 50%,
        class 3 = moderate = yields between 20% and 35%,
        class 2 = marginal  = yields between 10% and 20%,
        class 1 = very marginal = yields between 0% and 10%,
        class 0 = not suitable = yields less than 0%.
        '''

        net_revenue = self.net_revenue_list[self.crop_name_list.index(crop_name)]
        net_revenue_class = np.zeros(net_revenue.shape)

        if len(net_revenue[net_revenue>0]) > 0:
            net_revenue_max = np.amax(net_revenue[net_revenue>0])
            net_revenue_min = np.amin(net_revenue[net_revenue>0])
        else:
            net_revenue_max = 0
            net_revenue_min = 0
        
        net_revenue_10P = (net_revenue_max-net_revenue_min)*(10/100) + net_revenue_min
        net_revenue_20P = (net_revenue_max-net_revenue_min)*(20/100) + net_revenue_min
        net_revenue_35P = (net_revenue_max-net_revenue_min)*(35/100) + net_revenue_min
        net_revenue_50P = (net_revenue_max-net_revenue_min)*(50/100) + net_revenue_min
        net_revenue_63P = (net_revenue_max-net_revenue_min)*(63/100) + net_revenue_min
        net_revenue_75P = (net_revenue_max-net_revenue_min)*(75/100) + net_revenue_min

        net_revenue_class[ np.all([net_revenue<=0], axis=0) ] = 0 # not suitable
        net_revenue_class[ np.all([0<net_revenue, net_revenue<=net_revenue_10P], axis=0) ] = 1 # ver marginal
        net_revenue_class[ np.all([net_revenue_10P<net_revenue, net_revenue<=net_revenue_20P], axis=0) ] = 2 # marginal
        net_revenue_class[ np.all([net_revenue_20P<net_revenue, net_revenue<=net_revenue_35P], axis=0) ] = 3 # moderate
        net_revenue_class[ np.all([net_revenue_35P<net_revenue, net_revenue<=net_revenue_50P], axis=0) ] = 4 # medium
        net_revenue_class[ np.all([net_revenue_50P<net_revenue, net_revenue<=net_revenue_63P], axis=0) ] = 5 # good
        net_revenue_class[ np.all([net_revenue_63P<net_revenue, net_revenue<=net_revenue_75P], axis=0) ] = 6 # high
        net_revenue_class[ np.all([net_revenue_75P<net_revenue], axis=0)] = 7 # very high

        return net_revenue_class

    def getNormalizedNetRevenue(self, crop_name):

        """Getting the normalized net revenue of the provided crop_name.
        
        Args:
            crop_name (String): the crop name provided in previous function.
        
        Return:
            2-D NumPy Array: normalized net revenue (Value between 0= Not suitable and 1 = Suitable)"""

        net_revenue = self.net_revenue_list[self.crop_name_list.index(crop_name)]
        net_revenue_max =  np.max(np.array(self.net_revenue_list), axis=0)

        net_revenue_norm = np.zeros(net_revenue.shape)

        net_revenue_norm[net_revenue_max>0] = np.divide(net_revenue[net_revenue_max>0], net_revenue_max[net_revenue_max>0])

        net_revenue_norm_class = np.zeros(net_revenue.shape)

        net_revenue_norm_class[ np.all([net_revenue_norm<=0], axis=0) ] = 0 # not suitable
        net_revenue_norm_class[ np.all([0<net_revenue_norm, net_revenue_norm<=0.2], axis=0) ] = 1 #  marginal
        net_revenue_norm_class[ np.all([0.2<net_revenue_norm, net_revenue_norm<=0.4], axis=0) ] = 2 # moderate
        net_revenue_norm_class[ np.all([0.4<net_revenue_norm, net_revenue_norm<=0.6], axis=0) ] = 3 # fair
        net_revenue_norm_class[ np.all([0.6<net_revenue_norm, net_revenue_norm<=0.8], axis=0) ] = 4 # good
        net_revenue_norm_class[ np.all([0.8<net_revenue_norm, net_revenue_norm<=1.0], axis=0) ] = 5 # good

        return net_revenue_norm_class

#-------------------------------------- End of Code-------------------------------------------------------
