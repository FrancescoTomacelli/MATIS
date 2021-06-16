import Funzioni
import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
from numpy import random
from pandas import Series


period1 = 0
period2 = 0
period3 = 0
period4 = 0
period5 = 0

trend1 = 0
trend2 = 0
trend3 = 0
trend4 = 0
trend5 = 0

lenght=1000

random_noise=1
i_test=0
i_ciclo=0

synSeries1 = Funzioni.GenerateSynSeries(lenght, random_noise, trend1, 100, 1, period1, i_test,i_ciclo)
#synSeries2 = Funzioni.GenerateSynSeries(lenght, random_noise, trend2, 100, 1, period2, i_test,i_ciclo)
#synSeries3 = Funzioni.GenerateSynSeries(lenght, random_noise, trend3, 100, 1, period3, i_test,i_ciclo)
#synSeries4 = Funzioni.GenerateSynSeries(lenght, random_noise, trend4, 100, 1, period4, i_test,i_ciclo)

synSeriesConc = synSeries1
#synSeriesConc = Funzioni.concatSeries(synSeriesConc, synSeries2)
#synSeriesConc = Funzioni.concatSeries(synSeriesConc, synSeries3)
#synSeriesConc = Funzioni.concatSeries(synSeriesConc, synSeries4)



armaSign = Funzioni.GenerateArmaSignal(len(synSeriesConc))
synSeriesConc = Funzioni.AddArmaSignal_toSeries(armaSign,synSeriesConc)

dti1 = pd.date_range("2018-01-01", periods=len(synSeriesConc), freq="D")
synSeriesConc.index = dti1

series = synSeriesConc

series.plot()
plt.show()

#series.to_csv('D:/Universitaa/TESI/tests/Wind90 Test/test riassuntivi/test buoni LSTM ale/serie sintetiche stazionarie arma signal/arma.csv')

oracle = yf.download("ORCL", start="2016-01-01", end="2021-01-01")
oracle = oracle.loc[:, "Close"]
oracle = Funzioni.FillStockSeries(oracle)
oracle.to_csv('D:/Universitaa/TESI/tests/Datasets/series stock/oracle_2016_2021.csv')

mcDonald = yf.download("MCD", start="2016-01-01", end="2019-01-01")
mcDonald = mcDonald.loc[:,"Close"]
mcDonald = Funzioni.FillStockSeries(mcDonald)
mcDonald.to_csv('D:/Universitaa/TESI/tests/Datasets/series stock/mcDonald_2016_2019.csv')

disney= yf.download("DIS", start="1990-01-01", end="1997-01-01")
disney = disney.loc[:,"Close"]
disney = Funzioni.FillStockSeries(disney)
disney.to_csv('D:/Universitaa/TESI/tests/Datasets/series stock/disney_1990_1997.csv')

amazon = yf.download("AMZN",start="2013-01-01", end="2018-01-01")
amazon = amazon.loc[:,"Close"]
amazon = Funzioni.FillStockSeries(amazon)
amazon.to_csv('D:/Universitaa/TESI/tests/Datasets/series stock/amazon_2013_2018.csv')

google = yf.download("GOOG",start="2004-01-01",end= "2021-01-01")
google = google.loc[:,"Close"]
google = Funzioni.FillStockSeries(google)
google.to_csv('D:/Universitaa/TESI/tests/Datasets/series stock/google_2004_2021')

