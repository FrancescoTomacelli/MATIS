import Funzioni
from pandas import read_csv
from pandas import Series
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot

global counter_photo
counter_photo=0
counter_photo2=100


fil= open("D:/Universitaa/TESI/tests/immagini/info.txt","w+")
fil.write('***** INFO ***** \n')
fil.close()

seriesRead = read_csv('Datasets/daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

period1=7
period2=32
period3=28
period4=13

synSeries1= Funzioni.GenerateSynSeries(500,50,1,100,1,1/period1)
synSeries2= Funzioni.GenerateSynSeries(500,50,2,100,1,1/period2)
synSeries3= Funzioni.GenerateSynSeries(500,50,-1,100,1,1/period3)
synSeries4= Funzioni.GenerateSynSeries(500,50,1,100,1,1/period4)

synSeriesConc = Funzioni.concatSeries(synSeries1,synSeries2)
synSeriesConc = Funzioni.concatSeries(synSeriesConc,synSeries3)
synSeriesConc = Funzioni.concatSeries(synSeriesConc,synSeries4)



dti1 = pd.date_range("2018-01-01", periods=len(synSeriesConc), freq="D")


synSeriesConc.index=dti1


series=synSeriesConc
#period1='daily_min_temperatures'
#series=seriesRead


autocorrelation_plot(series)
plt.show()

plt.title("Series original P1= "+str(period1)+" P2= "+str(period2)+" P3= "+str(period3)+" P4= "+str(period4)+"")
series.plot()
plt.savefig('D:/Universitaa/TESI/tests/immagini/0_SeriesOriginal_'+str(period1)+'_'+str(period2)+'_'+str(period3)+'_'+str(period4)+'_'+ str(counter_photo)+'_.png')
#plt.savefig('D:/Universitaa/TESI/tests/immagini/Manuf_val_ship'+ str(counter_photo)+'_.png')
counter_photo=counter_photo+1
plt.show()


series_extracted=Funzioni.Stationarize_PSO_Window3(series,counter_photo,period1,period2,period3,period4)
k=1
for ser in series_extracted:
    ser.plot(color='red')
    plt.savefig('D:/Universitaa/TESI/tests/immagini/series_extracted_' + str(k) + '_.png')
    plt.show()
    k=k+1
    result = Funzioni.StationarizeWithPSO_Original(ser)
    seriesOriginal = ser
    seriesTrasf2 = result[0]
    seriesTrasf1 = result[1]
    particle = result[2]
    lamb = result[3]
    Funzioni.TestPrediction_AutoArima_Prophet_LSTM(seriesOriginal, seriesTrasf1, seriesTrasf2, particle, lamb,counter_photo2)
    counter_photo2=counter_photo2+5



