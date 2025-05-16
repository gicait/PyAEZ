# v2.3

- **_Module I: Climate Regime_**
    - Added new agro-climatic indicators
        - Annual Temperature Amplitude
        - Annual Reference Evapotranspiration (ETo)
        - Annual Reference Actual Evapotranspiration (ETa)
        - Annual Moisture Availability Index (P/ETo)
        - Reference Annual Water Deficit (Wde)
        - Net Primary Productivity (NPP)
        - LGP of the Longest Component (lgd)
        - Beginning Date of the Longest LGP component (lgb)
        - Total number of hibernating periods
        - Beginning date of the hibernation

- **_Module II (Crop Simulation)_**
    - Revised perennial crop simulation workflow.
    - Revised crop calendar determination for perennial crops.
    - Introduced a new crop simulation routine for hibernating crops.
    - Added new outputs from crop simulation for both rainfed and irrigated conditions:
        - Actual crop evapotranspiration from precipitation (i.e., excluding irrigation) (eta)
        - LUT water deficit/ net irrigation requirement during crop cycle (wde)

- **_Module III: Climatic Constraints_**

    - Added a functional module that automatically extracts LUT/management specific agro-climatic constraint tables from GAEZ database.

- **_Module IV: Soil Constraints_**

    - Updated soil evaluation methodology & input excel sheet settings based on GAEZ v5 soil evaluation requirements.
    - New function: Available Water Holding Capacity (AWC) calculation.

- **_Module V: Terrain Constraints_**

    - Terrain suitability evaluation with percent slope map is substituted with eight layers of slope class distribution by GAEZ methodology.
