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
import os
import yfinance as yf
from darts import TimeSeries
import shutil





oracle = yf.download("ORCL", start="2016-01-01", end="2021-01-01")
oracle = oracle.loc[:, "Close"]
oracle = Funzioni.FillStockSeries(oracle)


mcDonald = yf.download("MCD", start="2016-01-01", end="2019-01-01")
mcDonald = mcDonald.loc[:,"Close"]
mcDonald = Funzioni.FillStockSeries(mcDonald)

disney= yf.download("DIS", start="1990-01-01", end="1997-01-01")
disney = disney.loc[:,"Close"]
disney = Funzioni.FillStockSeries(disney)

amazon = yf.download("AMZN",start="2013-01-01", end="2018-01-01")
amazon = amazon.loc[:,"Close"]
amazon = Funzioni.FillStockSeries(amazon)

google = yf.download("GOOG",start="2004-01-01",end= "2021-01-01")
google = google.loc[:,"Close"]
google = Funzioni.FillStockSeries(google)

ciclo_esterno=5
i_ciclo=0
start_ciclo=4
serie0 = read_csv('Datasets/Manufacturer value of shipment.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
serie1 = read_csv('Datasets/mon_drug_sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
serie2 = amazon
serie3 = google
serie4 = 70
serie5 = disney
serie6 = read_csv('Datasets/Electric_Production.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
#serie6 = read_csv('Datasets/Covid19-italy.csv', header=0, index_col=0, parse_dates=True, squeeze=True)


serie7 = read_csv('Datasets/DailyDelhiClimateTrain.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
serie8 = read_csv('Datasets/monthly-beer-production-in-austr.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
serie9 = read_csv('Datasets/daily-total-female.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
serie10 = oracle
serie11 = disney
serie12 = mcDonald
dir_path = 'D:/Universitaa/Anaconda/Tesi_Codice/.darts'

for i_ciclo in range(start_ciclo,ciclo_esterno):
    counter_test = 2
    i_test = 1
    start=1
    Funzioni.alarm()
    for i_test in range(start, counter_test + 1):
        dir_path = 'D:/Universitaa/Anaconda/Tesi_Codice/.darts'
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))

        global counter_photo
        counter_photo = 0
        counter_photo2 = 100

        seriesRead = read_csv('Datasets/covidNuoviPositivi.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

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


        random_noise = 50
        lenght = 1500

        if(i_ciclo==4):
            random_noise=50

       # if(i_ciclo==5):
          #  random_noise=50

        newpath = r'D:/Universitaa/TESI/tests/immagini/ciclo_esterno' + str(i_ciclo) + '/test_' + str(i_test)
        os.makedirs(newpath)

        fil = open("D:/Universitaa/TESI/tests/immagini/ciclo_esterno" + str(i_ciclo) + "/test_" + str(i_test) + "/info.txt", "w+")
        fil.write('***** INFO ***** \n')
        fil.write("Period 1 = " + str(period1) + " Period 2 = " + str(period2) + " Period 3 = " + str(
            period3) + " Period 4 =" + str(period4) + "\n")
        fil.write("Trend 1 = " + str(trend1) + " Trend 2 = " + str(trend2) + " Trend 3 = " + str(
            trend3) + " Trend 4 = " + str(trend4) + " Trend 5 = " + str(trend5) + "\n")
        fil.close()

        random.seed()
        synSeries1 = Funzioni.GenerateSynSeries(lenght, random_noise, trend1, 100, 1, period1, i_test,i_ciclo)
        #synSeries2 = Funzioni.GenerateSynSeries(lenght, random_noise, trend2, 100, 1, period2, i_test,i_ciclo)
        #synSeries3 = Funzioni.GenerateSynSeries(lenght, random_noise, trend3, 100, 1, period3, i_test,i_ciclo)
        #synSeries4 = Funzioni.GenerateSynSeries(lenght, random_noise, trend4, 100, 1, period4, i_test,i_ciclo)
        # synSeries5 = Funzioni.GenerateSynSeries(lenght,50,trend5,100,1,period5)

        synSeriesConc = synSeries1
        #synSeriesConc = Funzioni.concatSeries(synSeriesConc, synSeries2)
        #synSeriesConc = Funzioni.concatSeries(synSeriesConc, synSeries3)
        #synSeriesConc = Funzioni.concatSeries(synSeriesConc, synSeries4)
        # synSeriesConc = Funzioni.concatSeries(synSeriesConc,synSeries5)

        dti1 = pd.date_range("2018-01-01", periods=len(synSeriesConc), freq="D")

        synSeriesConc.index = dti1

        series = synSeriesConc

        if(i_ciclo==0):
            series=serie0
            period1='Manufacturer_value_shipment'

        if(i_ciclo==1):
            series=serie1
            period1='mon_drug_sales'

        if(i_ciclo==2):
            series=serie2
            period1='AMZN'

        if(i_ciclo==3):
            series=serie3
            period1='GOOG'

        if(i_ciclo==6):
            series=serie6
            period1= 'ElectricProduction'

        if(i_ciclo==7):
            series=serie7
            period1='Delhi_daily_temperatures'

        if(i_ciclo==8):
            series=serie8
            period1='monthly_beer_production'

        if(i_ciclo==9):
            series=serie9
            period1='daily_total_female'

        if(i_ciclo==10):
            series=serie10
            period1='oracle'

        if(i_ciclo==11):
            series=serie11
            period1='disney'

        if(i_ciclo==12):
            series=serie12
            period1='McDonald'

        if(i_ciclo==5):
            series=serie5
            period1='Disney'





        # period1 = 'Covid_Nuovi_Positivi'
        # series = seriesRead

        # divido la serie in train set e test set
        train_size = int(len(series) * 0.9)
        train_set = series[0:train_size]
        test_set = series[train_size:]

        # creiamo delle serie di nan, che mi servono dopo per ricostruire la serie un volta divisa in pezzi e trasformata
        seriesRecostructed = list()
        seriesTrasf2Recostructed = list()
        seriesTrasf1Recostructed = list()

        for i in range(0, len(train_set)):
            seriesRecostructed.append(np.nan)
            seriesTrasf1Recostructed.append(np.nan)
            seriesTrasf2Recostructed.append(np.nan)

        seriesRecostructed = Series(seriesRecostructed)
        seriesRecostructed.index = train_set.index

        seriesTrasf1Recostructed = Series(seriesTrasf1Recostructed)
        seriesTrasf1Recostructed.index = train_set.index

        seriesTrasf2Recostructed = Series(seriesTrasf2Recostructed)
        seriesTrasf2Recostructed.index = train_set.index

        # possiamo passare adesso alle trasformazioni

        plt.title(
            "Series original P1= " + str(period1) + " P2= " + str(period2) + " P3= " + str(period3) + " P4= " + str(
                period4) + "P5=" + str(period5) + "")
        train_set.plot()
        test_set.plot(color='red')
        plt.savefig(
            'D:/Universitaa/TESI/tests/immagini/ciclo_esterno'+ str(i_ciclo) + '/test_' + str(i_test) + '/0_SeriesOriginal_' + str(period1) + '_' + str(
                period2) + '_' + str(period3) + '_' + str(period4) + '_' + str(period5) + '_' + str(
                counter_photo) + '_.png')
        # plt.savefig('D:/Universitaa/TESI/tests/immagini/Manuf_val_ship'+ str(counter_photo)+'_.png')
        counter_photo = counter_photo + 1
        plt.show()

        # vado a estrarre le non stazionarità della serie
        series_extracted = Funzioni.Stationarize_PSO_Window4(train_set, counter_photo, period1, period2, period3,
                                                             period4, i_test,i_ciclo)

        # vado a trasformare una alla volta le non stazionarietà
        extr_count = 0
        for ser in series_extracted:
            # seriesTrasf1, lamb = yeojohnson(ser)
            # seriesTrasf1 = Series(seriesTrasf1)
            # result = Funzioni.StationarizeWithPSO_withoutYeo(seriesTrasf1,lamb)
            result = Funzioni.StationarizeWithPSO_Original(ser, i_test,i_ciclo)
            seriesTrasf1 = result[1]
            seriesTrasf2 = result[0]
            particle = result[2]
            lamb = result[3]

            # plottiamo la serie estratta e la trasformata
            plt.figure()
            plt.subplot(311)
            plt.title('Extracted Original')
            ser.plot(color='blue')
            plt.subplot(313)
            plt.title("Extracted Trasformed Particle = {}   lambda= {} ".format(particle, lamb))
            seriesTrasf2.plot(color='green')
            plt.savefig('D:/Universitaa/TESI/tests/immagini/ciclo_esterno' + str(i_ciclo) + '/test_' + str(i_test) + '/series_extracted' + str(
                extr_count) + '.png')
            plt.show()
            extr_count = extr_count + 1

            # normalizziamo  serTrasf2
            seriesTrasf2, scaler2 = Funzioni.Normalize_Series(seriesTrasf2)

            # andiamo a ricollegare i "pezzi" trasformati 1
            initT1 = seriesTrasf1.index[0]
            finT1 = seriesTrasf1.index[len(seriesTrasf1) - 1]
            seriesTrasf1Recostructed[initT1:finT1] = seriesTrasf1[initT1:finT1]

            # andiamo a ricollegare i "pezzi" trasformati 2
            initT2 = seriesTrasf2.index[0]
            finT2 = seriesTrasf2.index[len(seriesTrasf2) - 1]
            seriesTrasf2Recostructed[initT2:finT2] = seriesTrasf2[initT2:finT2]

        # interpoliamo le due ricostruzioni per riempire i valori mancanti
        seriesTrasf1Recostructed = seriesTrasf1Recostructed.interpolate()
        seriesTrasf2Recostructed = seriesTrasf2Recostructed.interpolate()

        # ora abbiamo la serie trasformata, che è stata trasformata prima dividendo in pezzi, poi traformando ogni singolo pezzo e poi rimettendole insieme

        Funzioni.TestPrediction_AutoArima_Prophet_LSTM_Window90_Ale(series, seriesTrasf1Recostructed,
                                                                seriesTrasf2Recostructed,
                                                                particle, lamb, counter_photo, train_set, test_set,
                                                                scaler2, i_test,i_ciclo)



