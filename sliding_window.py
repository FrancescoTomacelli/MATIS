import Funzioni
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

period1=8
period2=15
period3=0
period4=0
period5=0

synSeries1= Funzioni.GenerateSynSeries(100,50,3,100,1,1/period1)
synSeries2= Funzioni.GenerateSynSeries(100,50,-3,100,1,1/period2)
#synSeries3= Funzioni.GenerateSynSeries(500,50,0,100,1,1/period3)
#synSeries4= Funzioni.GenerateSynSeries(500,50,2,100,1,1/period4)
#synSeries5= Funzioni.GenerateSynSeries(500,50,1,100,1,1/period5)


synSeriesConc = Funzioni.concatSeries(synSeries1,synSeries2)
#synSeriesConc = Funzioni.concatSeries(synSeriesConc,synSeries3)
#synSeriesConc = Funzioni.concatSeries(synSeriesConc,synSeries4)
#synSeriesConc = Funzioni.concatSeries(synSeriesConc,synSeries5)



dti1 = pd.date_range("2018-01-01", periods=len(synSeriesConc), freq="D")


synSeriesConc.index=dti1


series=synSeriesConc
#period1='daily_min_temperatures'
#series=seriesRead


plt.title("Series original P1= "+str(period1)+" P2= "+str(period2)+" P3= "+str(period3)+" P4= "+str(period4)+"P5=" +str(period5)+"")
series.plot()
plt.savefig('D:/Universitaa/TESI/tests/immagini/0_SeriesOriginal_'+str(period1)+'_'+str(period2)+'_'+str(period3)+'_'+str(period4)+'_'+str(period5)+'_'+ str(counter_photo)+'_.png')
#plt.savefig('D:/Universitaa/TESI/tests/immagini/Manuf_val_ship'+ str(counter_photo)+'_.png')
counter_photo=counter_photo+1
plt.show()


Funzioni.Sliding_Window(series,counter_photo,period1,period2,period3,period4)


Funzioni.alarm()
