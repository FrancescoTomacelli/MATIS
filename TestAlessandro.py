import Funzioni
from pandas import read_csv
from matplotlib import pyplot as plt

series = read_csv('D:/Universitaa/TESI/tests/Datasets/prova.csv',header=0, index_col=0, parse_dates=True, squeeze=True)



train = int(len(series)*0.9)
train_set = series[:train]
test_set = series[train:]

#TRASFORMAZIONE
seriesTrasf2,seriesTrasf1,list_diff,list_lamb,scaler= Funzioni.ATSS(train_set)
diff = list_diff[-1]
lamb = list_lamb[-1]
print(list_diff)
print(list_lamb)

#PREDIZIONE
result = Funzioni.ProphetPredictSeries_Window90(seriesTrasf2, test_set)
forecast = result[0]
forecast.index = test_set.index

#INVERSIONE
forecastInverted = Funzioni.ATSS_Invert_Prediction(forecast, seriesTrasf1, diff, lamb, scaler)
forecastInverted.index = test_set.index

