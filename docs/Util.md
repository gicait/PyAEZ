# Introduction

UtlitiesCalc module is the compulsory module which comprises of technical calculations, used in all the PyAEZ modules.

Starting from v2.3, the object class declaration to function access is revoked; instead, users only need to import the module directly

```py title="Import UtilitiesCalc" linenums="1"
import pyaez.UtilitiesCalc as util
```

## Major Functionalities

### Monthly to Daily Interpolation

This function performs interpolation of monthly climatic data into daily climatic data using quadratic spline interpolation recommended by GAEZ framework. Interpolation is performed from starting date to end date of the cycle.

```py title="Monthly to daily interpolation" linenums="1"
daily_vector = util.interpMonthlyToDaily(monthly_vector, cycle_begin, cycle_end, no_minus_values= False)
```

**Input**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|monthly_vector|monthly climate data with 12 elements|-|1D-NumPy Array|
|cycle_begin|beginning date of the cycle|Julian Day|Integer|
|cycle_end|end date of the cycle|Julian Day|Integer|
|no_minus_values|If `True`, negative values are forced to be zero to get rid of any unrealistic negative values in the climatic parameters (e.g., Precipitation). If `False`, negative values are allowed.|-|Boolean|

**Output**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|daily_vector|Output daily interpolated cliamte data between `cycle_begin` and `cycle_end` dates.|-|1D-NumPy Array|

___

### Daily to Monthly Aggregation

This function aggregates daily climatic data into monthly climate data. The aggregation is done by averaging the daily to corresponding months, whether it's leap year or non-leap year.

```py title="Daily to Monthly value" linenums="1"
monthly_vector = util.averageDailyToMonthly(daily_vector, leap_year = False)
```

**Input**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|daily_vector|365/366 daily values|-|1D-NumPy Array|
|leap_year|`True` for leap year, `False` for non-leap_year. Default value is `False`.|-|Boolean|

**Output**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|monthly_vector|Monthly aggregated data|-|1D-NumPy Array|

___

### Creating Latitude Map

This function creates the latitude map done by linear interpolation using minimum and maximum latitude values of the study area. The calculation is done based on GAEZ FORTRAN routine.

```py title="Latitude map creation" linenums="1"
lat_map = util.generateLatitudeMap(lat_min, lat_max, im_height, im_width)
```

**Input**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|lat_min|Minimum latitude|Decimal degree|float|
|lat_max|Maximum latitude|Decimal degree|float|
|im_height|Resultant height dimension of laitude map|Number of pixels|Integer|
|im_width|Resultant width dimension of latitude map|Number of pixels|Integer|

**Output**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|lat_map|Latitude map. The resuting dimension of the latitdue map will be accoriding to settings of `im_height` and `im_width`.|Decimal degree|2D-NumPy Array|

__

### Classification of Final Yield Crop

This function classifies yield estimation and produces crop suitability maps according to AEZ framework. Note that this function can be executed once the yield map has passed from Module 2 to Module 5. The classificaiton consists fo five classes tabulated in table below. The classification uses the maximum value of the yield obtained in a specified area and used in the classification.

```py title="Crop suitability map creation" linenums="1"
yld_class = util.classifyFinalYield(input_yld)
```

**Input**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|input_yld|Input yield|Kg/ha|2D-NumPy Array|

**Output**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|yld_class|Suitability map after yield classification||2D-NumPy Array|

**Table: Crop Suitabiltiy Classes based on Yield Classification**

|Pixel Value|Suitability Class|Description|
|---|---|---|
|1|Not Suitable|Yield between 0% and 20% of overall maximum yield|
|2|Marginally Suitable|Yield between 20% and 40% of overall maximum yield|
|3|Moderately Suitable|Yield between 40% and 60% of overall maximum yield|
|4|Suitable|Yield between 60% and 80% of overall maximum yield|
|5|Very Suitable|Yield equivalent greater than 80% or more of overall maximum yield|

___

### Saving GeoTIFF Rasters

This function allows saving any 2D NumPy arrays as georeferenced GeoTIFF raster files.

```py title="Saving the Rasters" linenums="1"
util.saveRaster(ref_rastter_path, out_path, numpy_raster)
```

**Input**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|ref_raster_path|Full file path of the locating reference raster. Must be GeoTIFF file. CRS information ins copied from this raster to the final raster.|-|String|
|out_path|Full file path of the desired location to save the output GeoTIFF file (including file name with .tif extension)|-|String|
|numpy_raster|Raster users want to save. Please make sure that the numpy array to insert CRS information the same as CRS informaiton from `ref_raster_path` to mitigate error.|-|2D-NumPy Array|

**Output**

File will be save as GeoTIFF file at the directory specified in `out_path`.

___

### Averaging Raster Files

This funciton averages list of raster files in time dimension. Some calcualtions in the AEZ methodology are recommended to performed with averaged climate data for 30 years.

```py title="Average rasters" linenums="1"
avg_raster = util.averageRasters(raster_3d)
```

**Input**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|raster_3d|Any climate data. Averaging will be done along time dimension.|-|3D-NumPy Array|

**Output**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|avg_raster|Averaged climate data|-|2D-NumPy Array|

___

### Calculate Wind Speed at Two-Meter Altitude

This function converts wind speed from a particular altitude measurement into two-meter altitude. All wind speed values used in PyAEZ calculations are at two-meter altitude. However, some climatic resources offers wind speed at ten-meter altitude, which in such case, the empirical relationship from FAO 56 report.

```py title="Converion to wind speed at two-meter latitude" linenums="1"
wind_sp_2m = util.windSpeedAt2m(wind_speed, altitude)
```

**Input**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|wind_speed|Wind speed input at x-meter measurement|ms^-1^|1D or 2D or 3D- NumPy Array|
|altitude|Altitude of measurement above ground.|meters|Integer|

**Output**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|wind_speed_2m|Wind speed measured at two-meter altitude|ms^-1^|1D or 2D or 3D-NumPy Array|

__

### Yield Gap

This function calculates the yield gap production (difference between the potential yield and the actual yield production).

```py title="Get yield gap" linenums="1"
yld_gap = util.getYieldGap(potential, actual)
```

**Input**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|potential|Potential yield|kg/ha|2D-NumPy Array|
|actual|Actual yield production|kg/ha|2D-NumPy Array|

**Output**

|Arguments|Description|Unit|Data Format|
|---|---|---|---|
|yield_gap|Yield gap|kg/ha|2D-NumPy Array|

___
