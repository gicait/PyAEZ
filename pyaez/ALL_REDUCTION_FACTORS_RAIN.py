'''--------------------------------------------------------'''
'''Reduction Factors for Climatic Constraints'''
'''--------------------------------------------------------'''

#defining yield reduction factors based of LGP Equivalent class
lgp_eq_class = [[0,29],[30,59],[60,89],[90,119],[120,149],[150,179],[180,209],[210,239],[240,269],[270,299],[300,329],[330,366]]
lgp_eq_red_fr = [[25,25,25,25,25,25,25,50,50,50,75,75],
                 [100,100,100,100,100,100,100,100,100,100,100,100],
                 [50,50,50,50,50,75,75,100,100,100,100,75],
                 [100,100,100,100,100,100,100,100,100,100,100,75]]

'''--------------------------------------------------------'''
'''Reduction Factors for Soil Constraints'''
'''--------------------------------------------------------'''

# value - values of soil characteristics (mush be ascending order)
# factor - yield reduction factors corresponding to each value

# soil texture for SQ1
TXT1_value = ['Fine', 'Medium', 'Coarse']
TXT1_factor = [90, 70, 30]

# soil texture for SQ2
TXT2_value = ['Fine', 'Medium', 'Coarse']
TXT2_factor = [90, 70, 30]

# soil texture for SQ7
TXT7_value = ['Fine', 'Medium', 'Coarse']
TXT7_factor = [90, 70, 30]

# soil organic carbon
OC_value = [0, 0.8, 1.5, 2]
OC_factor = [50, 70, 90, 100]

# soil pH
pH_value = [3.6, 4.1, 4.5, 5, 5.5, 6]
pH_factor = [10, 30, 50, 70, 90, 100]

# total exchangeable bases
TEB_value = [0, 1.6, 2.8, 4, 6.5]
TEB_factor = [30, 50, 70, 90, 100]

# base saturation
BS_value = [0, 35, 50, 80]
BS_factor = [50, 70, 90, 100]

# cation exchange capacity of soil
CECsoil_value = [0, 2, 4, 8, 10]
CECsoil_factor = [30, 50, 70, 90, 100]

# cation exchange capacity of clay
CECclay_value = [0, 16, 24]
CECclay_factor = [70, 90, 100]

# effective soil depth
RSD_value = [35, 70, 85]
RSD_factor = [50, 90, 100]

# soil coarse material (Gravel)
GRC_value = [10, 30, 90] # %
GRC_factor = [100, 35, 10]

# drainage
# VP: very poor, P: Poor, I: Imperfectly, MW: Moderately well, W: Well, SE: Somewhat Excessive, E: Excessive
DRG_value = ['VP', 'P', 'I', 'MW', 'W', 'SE', 'E']
DRG_factor = [50, 90, 100, 100, 100, 100, 100]

# exchangeable sodium percentage
ESP_value = [10, 20, 30, 40, 100] # %
ESP_factor = [100, 90, 70, 50, 10]

# electric conductivity
EC_value = [1, 2, 4, 6, 12, 100] # dS/m
EC_factor = [100, 90, 70, 50, 30, 10]

# soil phase rating for SQ3
SPH3_value = ['Lithic', 'skeletic', 'hyperskeletic']
SPH3_factor = [100, 50, 30]

# soil phase rating for SQ4
SPH4_value = ['Lithic', 'skeletic', 'hyperskeletic']
SPH4_factor = [100, 50, 30]

# soil phase rating for SQ5
SPH5_value = ['Lithic', 'skeletic', 'hyperskeletic']
SPH5_factor = [100, 50, 30]

# soil phase rating for SQ6
SPH6_value = ['Lithic', 'skeletic', 'hyperskeletic']
SPH6_factor = [100, 50, 30]

# soil phase rating for SQ7
SPH7_value = ['Lithic', 'skeletic', 'hyperskeletic']
SPH7_factor = [100, 50, 30]

# other soil depth/volume related characteristics rating
OSD_value = [0]
OSD_factor = [100]

# soil property rating - vertic or not
SPR_value = [0, 1]
SPR_factor = [100, 90]

# calcium carbonate
CCB_value = [3, 6, 15, 25, 100] # %
CCB_factor = [100, 90, 70, 50, 10]

# gypsum
GYP_value = [1, 3, 10, 15, 100] # %
GYP_factor = [100, 90, 70, 50, 10]

# vertical properties
VSP_value = [0, 1]
VSP_factor = [100, 90]

'''--------------------------------------------------------'''
'''Reduction Factors for Terrain Constraints'''
'''--------------------------------------------------------'''

Slope_class = [[0,0.5],[0.5,2],[2,5],[5,8],[8,16],[16,30],[30,45],[45,100]] # classes of slopes (Percentage Slope)
FI_class = [[0,1300],[1300,1800],[1800,2200],[2200,2500],[2500,2700],[1700,100000]] # classes of Fournier index
# sample data are for irrigated-intermediate input-wetland rice
# rows corresponding to FI class and columns corresponding to slope class
Terrain_factor = [[100,	100,	75,	50,	25,	0,	0,	0],
                [100,	100,	100,	100,	100,	75,	0,	0],
                [100,	100,	100,	100,	75,	25,	0,	0],
                [100,	100,	100,	100,	50,	0,	0,	0],
                [100,	100,	100,	100,	25,	0,	0,	0],
                [100,	100,	100,	100,	25,	0,	0,	0]]
