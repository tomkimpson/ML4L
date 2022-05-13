# ML4L
![GitHub repo size](https://img.shields.io/github/repo-size/tomkimpson/ML4L)
Machine learning for land surface modeling.


```
python main.py --process_data --join_data --train_model
```



# 1. Raw data

We have two primary sets of data. The first is a selection of ERA fields. These can be thought of as our features or inputs. The second is land surface temperature measurements from [MODIS](https://modis.gsfc.nasa.gov/about/). Both sets of data are provided by ECMWF having undergone some pre-processing and re-gridding, but are generally publically available.   


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

Daily files for [MODIS Aqua](https://aqua.nasa.gov/) day observations between 2016-2021 on an hourly grain. Resolution is 10800 longitude and 5400 pixels latitude (60 pixels per degree). Average LST error is $\leq 3$K. 
See the [User's guide](https://lpdaac.usgs.gov/documents/715/MOD11_User_Guide_V61.pdf) and the original [data product](https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MYD11A1)


See `Workflow.ipynb` for more details on the raw data.


---
# 2. Processing and joining raw data

## Pre-process raw data

To get all this disparate data into a more manageable form call `python main.py --process_raw_data`

This creates:
* Time Variable ERA fields. One file per month
* Time Constant ERA fields. One file per version (V15,V20)

+ monthly lakes, salt lakes which are unchanged. The MODIS files are also untouched.

## Join ERA-MODIS
In order to use the ERA-MODIS data together to train a model, it is necessary to join the data in time and space. That is, given a collection of ERA features at time $t$ and grid point $x$ what is the corresponding real world observation provided by MODIS? This is done by the call `python main.py --join_data`.

The general method involves taking an hour of ERA data (which covers the whole globe) and an hour of MODIS data (which covers just a strip) and then using a [GPU-accelerated k-nearest neighbours algorithm](https://github.com/facebookresearch/faiss) to find the nearest ERA grid point for every MODIS point. The 'nearness' measure is an L2 squared norm on the latitude/longitude coordinates rather than a [Haversine metric](https://en.wikipedia.org/wiki/Haversine_formula). We filter out any matches where the Haversine distance is > 50 km, and then group by the ERA coordinates to get an average temperature value. 

This outputs monthly `parquet` files which hold generally `x,t,features,target`. 
Example image

## Make data ML-ready

For the purposes of ML it is useful then modify these files via either a 'greedy' or a 'sensible' method:

* **Greedy** involved amalgamating all training data into a single file, all validation data into a single file, etc. We can then load this single file when training, reading into memory only the desired columns.  

* **Sensible** involves converting our `parquet` files into a [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format which can then be easily loaded batchwise into an ML pipeline.


WE typically use greedy...
Need to also normalize...


---

# 3. Training a model




Preprocessing step

## Example subtitle

Lorem ipsum


## Restructure

* Bring in constant fields later, rather than in the join?