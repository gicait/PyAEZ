[![Downloads](https://pepy.tech/badge/pyaez)](https://pepy.tech/project/pyaez)
[![PyPI version](https://badge.fury.io/py/PyAEZ.svg)](https://pypi.org/project/PyAEZ/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gicait/PyAEZ/blob/master/LICENSE)

# PyAEZ

PyAEZ is a python package consisted of many algorithms related to Agro-ecalogical zoning (AEZ) framework. PyAEZ tries to encapsulate all complex calculations in AEZ and try to provide user friendly, and intuitive ways to input data and output results after calculations.

# Installation

Now the package can be installed using `pip` command as below,

```shell
pip install pyaez==2.2
```

Alternatively, can be installed using using `conda` command as below,

```shell
conda install -c conda-forge pyaez
```

### Introduction

PyAEZ includes 6 main modules as below. Additionally to that, UtilityCalculations module is also included in PyAEZ to perform additional related utility calculations.

- Module I: Climate Regime
- Module II: Crop Simulations
- Module III: Climate Constraints
- Module IV: Soil Constraints
- Module V: Terrain Constraints
- Module VI: Economic Suitability Analysis
- UtilityCalculations Module

Other than 6 main modules and utility module, following 3 major algorithms related to AEZ also are included in PyAEZ. Those 3 major algorithms can be utilized individually without running whole PyAEZ.

- Biomass Calculations
- Evapotranspiration Calculations
- CropWat Calculations

### Dependencies

- numpy
- scipy
- gdal
- numba 

### Step-by-step Process

Following 6 Jupyter notebooks in the GitHub repository can be used as worked full example for PyAEZ 6 major modules.

- NB1_ClimateRegime.ipynb
- NB2_CropSimulation.ipynb
- NB3_ClimaticConstraints.ipynb
- NB4_SoilConstraints.ipynb
- NB5_TerrainConstraints.ipynb
- NB6_EconomicSuitability.ipynb

**Note**: _NB2_CropSimulation.ipynb_ takes a huge amount of time due to automatic crop calendar calculations. Hence, we have rewritten core parts of PyAEZ (_CropWatCalc.py_, _BioMassCalc.py_ and _ETOCalc.py_) with [Numba](http://numba.pydata.org/) compatible manner as well. Numba translates Python functions to optimized machine code at runtime, allowing calculation speeds close to C or FORTRAN.

### Release Note PyAEZv2.2
The improvement of the overall accuracy and reliability of the AEZ methodology is covered in new version in Module 2, 3, and input data usage is modified in Module 4 & 5. The key updates are listed as follows: 

**Module 2**: 
- New parameters added in reading crop/crop cycle parameters from excel sheet (minimum and maximum cycle lengths, plant height) 
- Revised algorithm flow for overall perennial/annual crop simulations. 
- New logic: kc factor adjustment based on local climate (Crop Water Requirement) 
- Revised logic: Daily value interpolations of Ac, Bc, Bo and Pm based on GAEZ (Biomass Calculation) 

**Module 3**: 
- Look-up tables of agro-climatic constraints can be provided as excel sheets. 
- Validated calculation procedures of fc3 map. 

**Module 4**: 
- Soil Requirement Suitability factors can now be prepared and provided as excel sheets. 
- New routine: 7-soil layer inputs of soil characteristics to evaluate soil suitability (based on HWSD v2.0) (Experimental). 
- New output: soil suitability rating map (fc4). 

**Module 5**: 
- Terrain constraint factors can now be prepared and provided as excel sheets. 
- New output: terrain suitability map (fc5).  


### Documentation

API Documentation is located in "docs" folder.

### Citation

Use this bibtex to cite us.

```
@misc{PyAEZ(v2.2),
  title={PyAEZ Python Package for Agro-ecological zoning (AEZ)},
  author={Swun Wunna Htet, Kittiphon Boonma, Guenther Fischer, Gianluca Franceschini, N. Lakmal Deshapriya, Thaileng Thol, Kavinda Gunasekara, Rajendra Shrestha,  Freddy Nachtergaele, Monica Petri, Beau Damen},
  year={2023},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/gicait/PyAEZ}},
}
```

### Developed by

[Geoinformatics Center](www.geoinfo.ait.ac.th) of [Asian Institute of Technology](www.ait.ac.th).


### Funding

Food and Agriculture Organization of the United Nations ([FAO](http://www.fao.org/home/en/)) and [FAO SAMIS](http://www.fao.org/in-action/samis/en/) (Strengthening Agro-climatic Monitoring and Information System) Project in Lao PDR.

### Sample Data Source

- Abatzoglou, J.T., S.Z. Dobrowski, S.A. Parks, K.C. Hegewisch, 2018, Terraclimate, a high-resolution global dataset of monthly climate and climatic water balance from 1958-2015, Scientific Data,

### Major AEZ References

- de Wit, C. T. (1965). Photosynthesis of leaf canopies. Agricultural Research Report No. 663. PUDOC, Wageningen, 57 .
- FAO. (1992). Cropwat: A computer program for irrigation planning and management. Land and Water Development Division, Rome, Italy, FAO Irrigation and Drainage Paper no 46 .
- FAO. (1998). Crop evapotranspiration. FAO Irrigation and Drainage Paper no.56 Rome, Italy.
- FAO. (2017). Final Report: National Agro-Economic Zoning for Major Crops in Thailand (NAEZ).
- Fischer, G., van Velthuizen, H., Shah, M., & Nachtergaele, F. (2002a). Global agroecological assessment for agriculture in the 21st century: Methodology and results. IIASA RR-02-02, IIASA, Laxenburg, Austria.
- Monteith, J. L. (1965). Evapotranspiration and the environment. In The State and Movement of Water in Living Organisms, 205-234.
- Monteith, J. L. (1981). Evapotranspiration and surface temperature. Quarterly Journal Royal Meteorological Society, 107 , 1-27.
