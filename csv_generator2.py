import Funzioni
import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
from numpy import random
from pandas import Series


period1 = 6
period2 = 18
period3 = 13
period4 = 0
period5 = 0

trend1 = 2
trend2 = -1
trend3 = 0
trend4 = 0
trend5 = 0

lenght=100

random_noise=1
i_test=0
i_ciclo=0

synSeries1 = Funzioni.GenerateSynSeries(lenght, random_noise, trend1, 100, 1, period1, i_test,i_ciclo)
synSeries2 = Funzioni.GenerateSynSeries(500, random_noise, trend2, 100, 1, period2, i_test,i_ciclo)
synSeries3 = Funzioni.GenerateSynSeries(500, random_noise, trend3, 100, 1, period3, i_test,i_ciclo)
#synSeries4 = Funzioni.GenerateSynSeries(lenght, random_noise, trend4, 100, 1, period4, i_test,i_ciclo)

synSeriesConc = synSeries1
synSeriesConc = Funzioni.concatSeries(synSeriesConc, synSeries2)
synSeriesConc = Funzioni.concatSeries(synSeriesConc, synSeries3)
#synSeriesConc = Funzioni.concatSeries(synSeriesConc, synSeries4)



armaSign = Funzioni.GenerateArmaSignal(len(synSeriesConc))
synSeriesConc = Funzioni.AddArmaSignal_toSeries(armaSign,synSeriesConc)

dti1 = pd.date_range("2018-01-01", periods=len(synSeriesConc), freq="D")
synSeriesConc.index = dti1

series = synSeriesConc

series.plot()
plt.show()

series.to_csv('D:/Universitaa/TESI/tests/Datasets/serie sintetiche non stazionarie arma signal/100_6_18_13.csv')