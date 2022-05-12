# ML4L

Machine learning for land surface modeling.


```
python main.py --process_data --join_data --train_model
```



# Raw data

We have two primary sets of data. The first is a selection of ERA fields. These can be thought of as our features or inputs. The second is land surface temperature measurements from [MODIS](https://modis.gsfc.nasa.gov/about/). Both sets of data are provided by ECMWF having undergone some processing and re-gridding, but are generally publically available.   


## ERA data

The raw ERA data is divided among the following files, all on a reduced Gaussain grid:

* `ERA_sfc`.  Monthly files between 2016-2021, hourly grain.
* `ERA_skin`. Monthly files between 2016-2021, hourly grain.
* `ERA_skt`.  Monthly files between 2016-2021, hourly grain.
* `climateV15`. Selection of constant-in-time features - for example orography - split over multiple files.
* `climateV20`. As above, but more recent version.
* `Monthly lakes`. 12 monthly files describing how lake cover varies month-to-month.
* `Salt lakes`. Single file with time-constant salt lake fraction


## MODIS data

Daily files for [MODIS Aqua](https://aqua.nasa.gov/) day observations between 2016-2021 on an hourly grain. Resolution is 10800 longitude and 5400 pixels latitude (60 pixels per degree). Average LST error is $$\leq 3$$K. See [User's guide](https://lpdaac.usgs.gov/documents/715/MOD11_User_Guide_V61.pdf) and the original [data product](https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MYD11A1#bands







See `Workflow.ipynb` for more detail on the raw data.



# Processed data

To get all this disparate data into a more manageable form call `python main.py --process_data`

This first creates:
* Time Variable ERA fields
* Time Constant ERA fields V15, V20
* +monthly lakes, salt lakes which are unchanged




## Example subtitle

Lorem ipsum


## Restructure

* Bring in constant fields later, rather than in the join?