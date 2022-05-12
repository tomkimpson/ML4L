# ML4L

Machine learning for land surface modeling.


```
python main.py --process_data --join_data --train_model
```



# Raw data

We have two primary sets of data. The first is a selection of ERA fields. These can be thought of as our features or inputs. The second is land surface temperature measurements from [MODIS](https://modis.gsfc.nasa.gov/about/). Both sets of data are provided by ECMWF having undergone some processing and re-gridding, but are generally publically available.   


## ERA data

The ERA data is divided among the following files

* `ERA_sfc`
* `ERA_skin`
* `ERA_skt`
* `climateV15`
* `climateV20`
* `Monthly lakes`
* `Salt lakes`


## MODIS data



See `Workflow.ipynb` for more detail on the raw data.

## Example subtitle

Lorem ipsum


## Restructure

* Bring in constant fields later, rather than in the join?