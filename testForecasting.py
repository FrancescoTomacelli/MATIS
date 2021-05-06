import Funzioni
from pandas import read_csv
from matplotlib import pyplot as plt
import pmdarima as pm
from pandas import Series
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error


series = read_csv('Datasets/DailyDelhiClimateTrain.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

seriesOriginal=series
result=Funzioni.StationarizeWithPSO_Original(series)

seriesTrasf2=result[0]
seriesTrasf1=result[1]
particle=result[2]
lamb=result[3]

Funzioni.TestPrediction_AutoArima_Prophet_LSTM(seriesOriginal,seriesTrasf1,seriesTrasf2,particle,lamb)
