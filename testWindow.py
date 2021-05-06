import Funzioni
from pandas import read_csv
from pandas import Series
import pandas as pd
from matplotlib import pyplot as plt

global counter_photo
counter_photo=0
counter_photo2=100

seriesRead = read_csv('Datasets/Airline_Passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)


synSeries1= Funzioni.GenerateSynSeries(500,50,1,200,1,1/15)
synSeries2= Funzioni.GenerateSynSeries(500,50,2,100,1,1/9)
synSeries3= Funzioni.GenerateSynSeries(500,50,-1,100,1,1/23)

synSeriesConc = Funzioni.concatSeries(synSeries1,synSeries2)
synSeriesConc = Funzioni.concatSeries(synSeriesConc,synSeries3)



dti1 = pd.date_range("2018-01-01", periods=len(synSeriesConc), freq="D")


synSeriesConc.index=dti1


series=synSeriesConc
#series=seriesRead

plt.title("Series original P1=15 P2=23 P3=9")
series.plot()
plt.savefig('D:/Universitaa/TESI/tests/immagini/Syn_12_7_30_'+ str(counter_photo)+'_.png')
counter_photo=counter_photo+1
plt.show()


series_extracted=Funzioni.Stationarize_PSO_Window(series,counter_photo)

for ser in series_extracted:
    #ser.plot(color='red')
    #plt.show()
    result = Funzioni.StationarizeWithPSO_Original(ser)
    seriesOriginal = ser
    seriesTrasf2 = result[0]
    seriesTrasf1 = result[1]
    particle = result[2]
    lamb = result[3]
    Funzioni.TestPrediction_AutoArima_Prophet_LSTM(seriesOriginal, seriesTrasf1, seriesTrasf2, particle, lamb,counter_photo2)
    counter_photo2=counter_photo2+5



