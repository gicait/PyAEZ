# PyAEZ
PyAEZ is a python package consisted of many algorithms related to Agro-ecalogical zoning (AEZ) framework. PyAEZ tries to encapsulate all complex calculations in AEZ and try to provide user friendly, and intuitive ways to input data and output results after calculations.

### Introduction
PyAEZ includes 5 main modules as below. Additionally to that, UtilityCalculations module is also included in PyAEZ to perform additional related utility calculations.

* Module I: Climate Regime
* Module II: Crop Simulations
* Module III: Climate Constraints
* Module IV: Soil Constraints
* Module V: Terrain Constraints
* UtilityCalculations Module

Other than 5 main modules and utility module, following 3 major algorithms related to AEZ also are included in PyAEZ. Those 3 major algorithms can be utilized individually without running whole PyAEZ.

* Biomass Calculations
* Evapotranspiration Calculations
* CropWat Calculations

### Dependencies
* numpy
* scipy
* gdal

### Step-by-step Process
Following 5 Jupyter notebooks can be used as worked full example for PyAEZ 5 major modules.
* NB1_ClimateRegime.ipynb
* NB2_CropSimulation.ipynb
* NB3_ClimaticConstraints.ipynb
* NB4_SoilConstraints.ipynb
* NB5_TerrainConstraints.ipynb

### Documentation
API Documentation is located in "docs" folder.

### Citation
Use this bibtex to cite us.
```
@misc{PyAEZ_2020,
  title={PyAEZ Python Package for Agro-ecological zoning (AEZ)},
  author={N. Lakmal Deshapriya, Thaileng Thol, Kavinda Gunasekara, Rajendra Shrestha, Gianluca Franceschini, Freddy Nachtergaele, Monica Petri, Beau Damen},
  year={2020},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/gicait/PyAEZ}},
}
```

### Funding
Food and Agriculture Organization of the United Nations (FAO).

### Main References
* de Wit, C. T. (1965). Photosynthesis of leaf canopies. Agricultural Research Report No. 663. PUDOC, Wageningen, 57 .
* FAO. (1992). Cropwat: A computer program for irrigation planning and management. Land and Water Development Division, Rome, Italy, FAO Irrigation and Drainage Paper no 46 .
* FAO. (1998). Crop evapotranspiration. FAO Irrigation and Drainage Paper no.56 Rome, Italy.
* Fischer, G., van Velthuizen, H., Shah, M., & Nachtergaele, F. (2002a). Global agroecological assessment for agriculture in the 21st century: Methodology and results. IIASA RR-02-02, IIASA, Laxenburg, Austria.
* Monteith, J. L. (1965). Evapotranspiration and the environment. In The State and Movement of Water in Living Organisms, 205-234.
* Monteith, J. L. (1981). Evapotranspiration and surface temperature. Quarterly Journal Royal Meteorological Society, 107 , 1{27.
