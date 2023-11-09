# Description
This competition is about the **detection of extreme weather events from atmospherical data**. The goal is to automatically classify a set of climate variables corresponding to a time point and location, latitude and longitude, into one of three classes:

* Standard background conditions
* [Tropical cyclone](https://en.wikipedia.org/wiki/Tropical_cyclone)
* [Atmospheric river](https://en.wikipedia.org/wiki/Atmospheric_river)

Models that are capable of accurately detecting such events are crucial for our understanding of how they may evolve under various climate change scenarios.

The data set for this competition is a relatively small portion of a bigger data set, [ClimateNet](https://portal.nersc.gov/project/ClimateNet/). The complete data set amounts to nearly 30 GB because it contains climate variables at almost 900,000 locations around the globe. The subset for this competition contains just 120 locations while keeping the data at all time points.

**The training set contains 44,760 data points** from 1996 to 2009, and the **test set contains 10,320 data points** from 2010 to 2013. Each data point consists of **16 atmospheric variables** such as pressure, temperature and humidity, besides the latitude, longitude and time. The complete set of variables is the following:

* lat: latitude
* lon: longitude
* time [YYYYMMDD]
* TMQ: total (vertically integrated) precipitable water [kg/m^2]
* U850: zonal wind at 850 mbar pressure surface [m/s]
* V850: meridional wind at 850 mbar pressure surface [m/s]
* UBOT: lowest level zonal wind [m/s]
* VBOT: lowest model level meridional wind [m/s]
* QREFHT: reference height humidity [kg/kg]
* PS: surface pressure [Pa]
* PSL: sea level pressure [Pa]
* T200: temperature at 200 mbar pressure surface [K]
* T500: temperature at 500 mbar pressure surface [K]
* PRECT: total (convective and large-scale) precipitation rate (liq + ice) [m/s]
* TS: surface temperature (radiative) [K]
* TREFHT: reference height temperature [K]
* Z1000: geopotential Z at 1000 mbar pressure surface [m]
* Z200: geopotential Z at 200 mbar pressure surface [m]
* ZBOT: lowest modal level height [m]


# How to run the code
First you have to install the packages
```python
pip install -r requirement.txt
```

Then you have to make an environment file .env and write your train and test data path into it

```python
TRAIN_PATH = 'Dataset/train.csv'
TEST_PATH = 'Dataset/test.csv'
```
After that you can run the code in the main.py file but you have to give it inputs regarding the work you want to do. The pattern is like this.
```python
python main.py svm hptuning
```
The first parameter is the model that you want to work with which can be among these three models.

1. logistic
2. randomforest
3. svm

The second parameter shows that you want to tune the hyper-parameters or you want to run the code with specific set of hyper-parameters and get the predictions on the test set and save it in a .csv file. This input has two states.

1. hptuning
2. submit   

If you choose the hptuning the code will run with the set of hyper-parameters that are defined in the code and give you the results. But if you choose the submit you have to provide the set of hyper-parameters that you want to feed to your model. This set of hyper-parameters can be among the provided hyper-parameters.

* --learning_rate
* --rho
* --alpha
* --loggamma
* --n_estimators
* --max_depth
* --min_samples_split
* --min_samples_leaf
* --kernel
* --degree
* --C
* --svmgamma

These are all the hyper-parameters but keep that in mind you don't need to provide them all. You only need to provide the hyper-parameters correspond to the model that you want to use. For example if you want to use logistic regression for submission you have to run a code like this.
```python
python main.py logistic submit --learning_rate 0.10 --rho 0.99 --loggamma 0.98 --alpha 0.00001 
```
For SVM you have to run something like this:
```python
python main.py svm submit --kernel rbf --c 100 --svmgamma 0.01
```
or for polynomial kernel:
```python
python main.py svm submit --kernel poly --degree 2
```
And for random forest you can run the code like this:
```python
python main.py randomforest submit --n_estimators 100 --max_depth 10 --min_samples_split 20 --min_samples_leaf 5
```