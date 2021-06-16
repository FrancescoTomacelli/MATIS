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


serie0 = read_csv('D:/Universitaa/TESI/tests/Datasets/serie sintetiche non stazionarie arma signal/15_30_arma.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
serie1 = read_csv('D:/Universitaa/TESI/tests/Datasets/serie sintetiche non stazionarie arma signal/6_18_13_arma.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
serie2 = read_csv('D:/Universitaa/TESI/tests/Datasets/serie sintetiche non stazionarie arma signal/21_9_27_14_arma.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
serie3 = read_csv('D:/Universitaa/TESI/tests/Datasets/serie sintetiche arma signal e periodo/7_arma.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
serie4 = read_csv('D:/Universitaa/TESI/tests/Datasets/serie sintetiche arma signal e periodo/15_arma.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
serie5 = read_csv('D:/Universitaa/TESI/tests/Datasets/serie sintetiche arma signal e periodo/17_arma.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
serie6 = read_csv('D:/Universitaa/TESI/tests/Datasets/serie sintetiche stazionarie arma signal/arma.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
serie7 = read_csv('D:/Universitaa/TESI/tests/Datasets/series stock/normal_noise.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
dir_path = 'D:/Universitaa/Anaconda/Tesi_Codice/.darts'

ciclo_esterno=8
i_ciclo=0
start_ciclo=7

for i_ciclo in range(start_ciclo,ciclo_esterno):
    if (i_ciclo == 0):
        series = serie0
        period1 = '15_30_arma'

    if (i_ciclo == 1):
        series = serie1
        period1 = '6_18_13_arma'

    if (i_ciclo == 2):
        series = serie2
        period1 = '21_9_27_14_arma'

    if (i_ciclo == 3):
        series = serie3
        period1 = '7_arma'

    if (i_ciclo == 4):
        series = serie4
        period1 = '15_arma'

    if (i_ciclo == 5):
        series = serie5
        period1 = '17_arma'

    if (i_ciclo == 6):
        series = serie6
        period1 = 'arma'


    if(i_ciclo == 7):
        series = serie7
        period1 = 'amazon  1000 campioni'

    counter_test = 10
    i_test = 1
    start = 1

    workbook,worksheet = Funzioni.CreateTabellaRiassuntiva(i_ciclo,period1)

    for i_test in range(start, counter_test + 1):
        dir_path = 'D:/Universitaa/Anaconda/Tesi_Codice/.darts'
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))

        global counter_photo
        counter_photo = 0
        counter_photo2 = 100

        newpath = r'D:/Universitaa/TESI/tests/immagini/ciclo_esterno' + str(i_ciclo) + '/test_' + str(i_test)
        os.makedirs(newpath)

        fil = open("D:/Universitaa/TESI/tests/immagini/ciclo_esterno" + str(i_ciclo) + "/test_" + str(i_test) + "/info.txt", "w+")
        fil.write('***** INFO ***** \n')

        #series.plot(color='violet')
        #plt.show()

        # divido la serie in train set e test set
        train_size = int(len(series) * 0.90)
        train_set = series[0:train_size]
        test_set = series[train_size:]

        # creiamo delle serie di nan, che mi servono dopo per ricostruire la serie un volta divisa in pezzi e trasformata
        seriesReacostructed = Funzioni.Create_Nan_Series(train_set)
        seriesTrasf2Recostructed = Funzioni.Create_Nan_Series(train_set)
        seriesTrasf1Recostructed = Funzioni.Create_Nan_Series(train_set)

        # passiamo adesso alle trasformazioni

        plt.title("Series original P1= " + str(period1) + "")
        train_set.plot()
        test_set.plot(color='red')
        plt.savefig('D:/Universitaa/TESI/tests/immagini/ciclo_esterno'+ str(i_ciclo) + '/test_' + str(i_test) + '/0_SeriesOriginal_' + str(period1) + '_' + str(counter_photo) + '_.png')
        counter_photo = counter_photo + 1
        plt.show()

        # vado a estrarre le non stazionarità della serie
        series_extracted = Funzioni.Stationarize_PSO_Window4(train_set, counter_photo, period1, 0,0,0, i_test,i_ciclo)

        # vado a trasformare una alla volta le non stazionarietà
        extr_count = 0
        list_diff= list()
        list_lamb= list()
        for ser in series_extracted:
            # seriesTrasf1, lamb = yeojohnson(ser)
            # seriesTrasf1 = Series(seriesTrasf1)
            # result = Funzioni.StationarizeWithPSO_withoutYeo(seriesTrasf1,lamb)
            result = Funzioni.StationarizeWithPSO_Original(ser, i_test,i_ciclo)
            seriesTrasf1 = result[1]
            seriesTrasf2 = result[0]
            particle = result[2]
            lamb = result[3]
            list_diff.append(particle)
            list_lamb.append(lamb)

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

        Funzioni.TestPrediction_AutoArima_Prophet_LSTM_Window90_Ale(series, seriesTrasf1Recostructed,seriesTrasf2Recostructed,particle, lamb, counter_photo, train_set, test_set,scaler2, i_test,i_ciclo,workbook,worksheet,list_diff,list_lamb)



    workbook.close()



