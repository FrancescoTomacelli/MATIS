import Funzioni
from pandas import read_csv
from pandas import Series
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas.plotting import autocorrelation_plot
import numpy as np
from numpy import random
from scipy.stats import yeojohnson


global counter_photo
counter_photo=0
counter_photo2=100



seriesRead = read_csv('Datasets/covidNuoviPositivi.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

period1=0
period2=0
period3=0
period4=0
period5=0

trend1=0
trend2=0
trend3=0
trend4=0
trend5=0

fil= open("D:/Universitaa/TESI/tests/immagini/info.txt","w+")
fil.write('***** INFO ***** \n')
fil.write("Period 1 = "+str(period1)+" Period 2 = "+str(period2)+" Period 3 = "+str(period3)+ " Period 4 =" +str(period4)+"\n")
fil.write( "Trend 1 = "+str(trend1)+" Trend 2 = "+str(trend2)+" Trend 3 = "+str(trend3)+" Trend 4 = "+str(trend4)+" Trend 5 = "+str(trend5)+"\n")
fil.close()


random.seed(9001)
synSeries1= Funzioni.GenerateSynSeries(500,50,trend1,100,1,period1)
#synSeries2= Funzioni.GenerateSynSeries(500,50,trend2,100,1,period2)
#synSeries3= Funzioni.GenerateSynSeries(500,50,trend3,100,1,period3)
#synSeries4= Funzioni.GenerateSynSeries(500,50,trend3,100,1,period4)
#synSeries5= Funzioni.GenerateSynSeries(500,50,trend5,100,1,period5)


synSeriesConc = synSeries1
#synSeriesConc = Funzioni.concatSeries(synSeriesConc,synSeries2)
#synSeriesConc = Funzioni.concatSeries(synSeriesConc,synSeries3)
#synSeriesConc = Funzioni.concatSeries(synSeriesConc,synSeries4)
#synSeriesConc = Funzioni.concatSeries(synSeriesConc,synSeries5)


dti1 = pd.date_range("2018-01-01", periods=len(synSeriesConc), freq="D")



synSeriesConc.index=dti1

#series=synSeriesConc

period1='Covid_Nuovi_Positivi'
series=seriesRead


#divido la serie in train set e test set
train_size=int(len(series)*0.9)
train_set=series[0:train_size]
test_set=series[train_size:]


#creiamo delle serie di nan, che mi servono dopo per ricostruire la serie un volta divisa in pezzi e trasformata
seriesRecostructed=list()
seriesTrasf2Recostructed=list()
seriesTrasf1Recostructed=list()

for i in range(0,len(train_set)):
    seriesRecostructed.append(np.nan)
    seriesTrasf1Recostructed.append(np.nan)
    seriesTrasf2Recostructed.append(np.nan)


seriesRecostructed=Series(seriesRecostructed)
seriesRecostructed.index=train_set.index

seriesTrasf1Recostructed=Series(seriesTrasf1Recostructed)
seriesTrasf1Recostructed.index=train_set.index

seriesTrasf2Recostructed=Series(seriesTrasf2Recostructed)
seriesTrasf2Recostructed.index=train_set.index


#possiamo passare adesso alle trasformazioni






plt.title("Series original P1= "+str(period1)+" P2= "+str(period2)+" P3= "+str(period3)+" P4= "+str(period4)+"P5=" +str(period5)+"")
train_set.plot()
test_set.plot(color='red')
plt.savefig('D:/Universitaa/TESI/tests/immagini/0_SeriesOriginal_'+str(period1)+'_'+str(period2)+'_'+str(period3)+'_'+str(period4)+'_'+str(period5)+'_'+ str(counter_photo)+'_.png')
#plt.savefig('D:/Universitaa/TESI/tests/immagini/Manuf_val_ship'+ str(counter_photo)+'_.png')
counter_photo=counter_photo+1
plt.show()

#vado a estrarre le non stazionarità della serie
series_extracted=Funzioni.Stationarize_PSO_Window4(train_set,counter_photo,period1,period2,period3,period4)

#vado a trasformare una alla volta le non stazionarietà
extr_count=0
for ser in series_extracted:

    #seriesTrasf1, lamb = yeojohnson(ser)
    #seriesTrasf1 = Series(seriesTrasf1)
    #result = Funzioni.StationarizeWithPSO_withoutYeo(seriesTrasf1,lamb)
    result = Funzioni.StationarizeWithPSO_Original(ser)
    seriesTrasf1 = result[1]
    seriesTrasf2 = result[0]
    particle = result[2]
    lamb = result[3]


    #plottiamo la serie estratta e la trasformata
    plt.figure()
    plt.subplot(311)
    plt.title('Extracted Original')
    ser.plot(color='blue')
    plt.subplot(313)
    plt.title("Extracted Trasformed Particle = {}   lambda= {} ".format(particle, lamb))
    seriesTrasf2.plot(color='green')
    plt.savefig('D:/Universitaa/TESI/tests/immagini/series_extracted' +str(extr_count) + '.png')
    plt.show()
    extr_count = extr_count + 1

    #normalizziamo  serTrasf2
    seriesTrasf2,scaler2= Funzioni.Normalize_Series(seriesTrasf2)


    #andiamo a ricollegare i "pezzi" trasformati 1
    initT1=seriesTrasf1.index[0]
    finT1=seriesTrasf1.index[len(seriesTrasf1)-1]
    seriesTrasf1Recostructed[initT1:finT1]=seriesTrasf1[initT1:finT1]

    #andiamo a ricollegare i "pezzi" trasformati 2
    initT2=seriesTrasf2.index[0]
    finT2=seriesTrasf2.index[len(seriesTrasf2)-1]
    seriesTrasf2Recostructed[initT2:finT2]=seriesTrasf2[initT2:finT2]

#interpoliamo le due ricostruzioni per riempire i valori mancanti
seriesTrasf1Recostructed=seriesTrasf1Recostructed.interpolate()
seriesTrasf2Recostructed=seriesTrasf2Recostructed.interpolate()


#ora abbiamo la serie trasformata, che è stata trasformata prima dividendo in pezzi, poi traformando ogni singolo pezzo e poi rimettendole insieme

Funzioni.TestPrediction_AutoArima_Prophet_LSTM_Window90(series,seriesTrasf1Recostructed,seriesTrasf2Recostructed,particle,lamb,counter_photo,train_set,test_set,scaler2)
Funzioni.alarm()