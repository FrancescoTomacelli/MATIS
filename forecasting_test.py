import Funzioni
from pandas import read_csv
from pandas import Series
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import pmdarima as pm
import time
start = time.time()
series = read_csv('D:/Universitaa/TESI/tests/Datasets/serie _reali/Manufacturer value of shipment.csv', header=0, index_col=0, parse_dates=True, squeeze=True)




particle = 12
trasf = Funzioni.PrintSeriesTrasform(series,0.99,particle)
series = trasf[0]


train_size = int(len(series) * 0.90)
train_set = series[0:train_size]
test_set = series[train_size:]

#train_set=train_set[particle:]

result1 = Funzioni.ProphetPredictSeries_Window90(train_set,test_set)
forecastOriginalProp = result1[0]
seriesPredictedOriginalProp = Series(forecastOriginalProp)
seriesPredictedOriginalProp.index = test_set.index
seriesPredicted = seriesPredictedOriginalProp

plt.title("Prophet")
test_set.plot(color='blue')
seriesPredicted.plot(color='red')
plt.show()


modelOriginal = pm.auto_arima(train_set, error_action='ignore', trace=True, suppress_warnings=True)
forecastOriginal = modelOriginal.predict(test_set.shape[0])

seriesPredictedOriginal = Series(forecastOriginal)
seriesPredictedOriginal.index = test_set.index

seriesPredicted = seriesPredictedOriginal

plt.title('ARIMA')
test_set.plot(color='blue')
seriesPredicted.plot(color='red')
plt.show()

seriesPredictedOriginalLSTM = Funzioni.LSTM_Prediction2_Window90_Ale(train_set, test_set)
seriesPredicted = seriesPredictedOriginalLSTM

plt.title('LSTM')
test_set.plot(color='blue')
seriesPredicted.plot(color='red')
plt.show()

rmseOriginalProp = sqrt(mean_squared_error(test_set, seriesPredicted))
maeOriginalProp = mean_absolute_error(test_set, seriesPredicted)
autocorr_residOriginal_Prop = Funzioni.Quantif_Autocorr_Residual(seriesPredicted, test_set)

print(rmseOriginalProp)
print(maeOriginalProp)
print(autocorr_residOriginal_Prop)


end = time.time()

elapsed_time= end-start
print("elapsed time")
print(elapsed_time)

