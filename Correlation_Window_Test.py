import Funzioni
from numpy import random
import pandas as pd
from matplotlib import pyplot as plt
import os
from scipy.stats import pearsonr
import yfinance as yf
from scipy.stats import kendalltau
from scipy.stats import spearmanr
import xlsxwriter


#leggo tutte le serie reali
covidNuoviPositivi = pd.read_csv('Datasets/Covid19-italy.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
covidDailyTests= pd.read_csv('Datasets/Covid19_DailyTests.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
covidDailyDeaths = pd.read_csv('Datasets/Covid_Daily_deaths.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
airlinePassengers= pd.read_csv('Datasets/Airline_Passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
monthlyDrugSales= pd.read_csv('Datasets/mon_drug_sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
monthlyBeerProduction = pd.read_csv('Datasets/monthly-beer-production-in-austr.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
amazon = yf.download("AMZN",start="2005-01-01", end="2021-01-01")
amazon = amazon.loc[:,"Close"]
amazon = Funzioni.FillStockSeries(amazon)
google = yf.download("GOOG",start="2005-01-01",end= "2021-01-01")
google = google.loc[:,"Close"]
google = Funzioni.FillStockSeries(google)



i_test=0
i_ciclo=0
counter_photo=0

range_noise = 50
length = 500

#creo le serie sintetiche
random.seed()
x = [random.randint(range_noise) for i in range(0, length)]
y = [random.randint(30) for i in range(0, length)]

#diversa base stesso trend
synS1 = Funzioni.GenerateSynSeriesCorrelation(length, x, 1, 100, 1, 0, i_test,i_ciclo)
synS2 = Funzioni.GenerateSynSeriesCorrelation(length, y, 1, 100, 1, 0, i_test,i_ciclo)
#diversa base stessa season
synS3 = Funzioni.GenerateSynSeriesCorrelation(length, x, 0, 100, 1, 7, i_test,i_ciclo)
synS4 = Funzioni.GenerateSynSeriesCorrelation(length, y, 0, 100, 1, 7, i_test,i_ciclo)
#diversa base stesso trend stessa season
synS5 = Funzioni.GenerateSynSeriesCorrelation(length, x, 1, 100, 1, 7, i_test,i_ciclo)
synS6 = Funzioni.GenerateSynSeriesCorrelation(length, y, 1, 100, 1, 7, i_test,i_ciclo)

#diversa base diverso trend
synS7 = Funzioni.GenerateSynSeriesCorrelation(length, x, 2, 100, 1, 0, i_test,i_ciclo)
synS8 = Funzioni.GenerateSynSeriesCorrelation(length, y, -3, 100, 1, 0, i_test,i_ciclo)
#diversa base diversa season
synS9 = Funzioni.GenerateSynSeriesCorrelation(length, x, 0, 100, 1, 7, i_test,i_ciclo)
synS10 = Funzioni.GenerateSynSeriesCorrelation(length, y, 0, 100, 1, 12, i_test,i_ciclo)
#diversa base diverso trend diversa season
synS11 = Funzioni.GenerateSynSeriesCorrelation(length, x, 1, 100, 1, 7, i_test,i_ciclo)
synS12 = Funzioni.GenerateSynSeriesCorrelation(length, y, -1, 100, 1, 12, i_test,i_ciclo)

#cambiamo la base y, ora diventa correlata ad x
y = [y[i]+x[i] for i in range(0,length)]

#stessa base stesso trend
synS13 = Funzioni.GenerateSynSeriesCorrelation(length, x, 1, 100, 1, 0, i_test,i_ciclo)
synS14 = Funzioni.GenerateSynSeriesCorrelation(length, y, 1, 100, 1, 0, i_test,i_ciclo)
#stessa base stessa season
synS15 = Funzioni.GenerateSynSeriesCorrelation(length, x, 0, 100, 1, 7, i_test,i_ciclo)
synS16 = Funzioni.GenerateSynSeriesCorrelation(length, y, 0, 100, 1, 7, i_test,i_ciclo)
#stessa base stesso trend e season
synS17 = Funzioni.GenerateSynSeriesCorrelation(length, x, 1, 100, 1, 7, i_test,i_ciclo)
synS18 = Funzioni.GenerateSynSeriesCorrelation(length, y, 1, 100, 1, 7, i_test,i_ciclo)

#stessa base diverso trend
synS19 = Funzioni.GenerateSynSeriesCorrelation(length, x, 1, 100, 1, 0, i_test,i_ciclo)
synS20 = Funzioni.GenerateSynSeriesCorrelation(length, y, -1, 100, 1, 0, i_test,i_ciclo)
#stessa base diversa season
synS21 = Funzioni.GenerateSynSeriesCorrelation(length, x, 0, 100, 1, 7, i_test,i_ciclo)
synS22 = Funzioni.GenerateSynSeriesCorrelation(length, y, 0, 100, 1, 12, i_test,i_ciclo)
#stessa base diverso trend diversa season
synS23 = Funzioni.GenerateSynSeriesCorrelation(length, x, 1, 100, 1, 7, i_test,i_ciclo)
synS24 = Funzioni.GenerateSynSeriesCorrelation(length, y, 1, 100, 1, 12, i_test,i_ciclo)



#mi servono per la stampa su file (le vecchie funzioni lo richiedono)
period1 = 0
period2 = 0
period3 = 0
period4 = 0
trend1 = 0
trend2 = 0
trend3 = 0
trend4 = 0
trend5 = 0

start_ciclo=12
end_ciclo=13
for i_ciclo in range(start_ciclo,end_ciclo):

    #selezioniamo la coppia di serie
    if(i_ciclo==0):
        synSeries1= covidNuoviPositivi
        synSeries2= covidDailyDeaths
        testo = "CovidNuoviPositivi - CovidDailyDeaths"

    if (i_ciclo == 1):
        synSeries1 = covidNuoviPositivi
        synSeries2 = covidDailyTests
        testo = "CovidNuoviPositivi - CovidDailyTests"

    if (i_ciclo == 2):
        synSeries1 = airlinePassengers
        synSeries2 = monthlyBeerProduction
        testo = "airlinePassengers - monthlyBeerProduction"

    if (i_ciclo == 3):
        synSeries1 = airlinePassengers
        synSeries2 = monthlyDrugSales
        testo = "airlinePassengers - monthlyDrugSales"

    if (i_ciclo == 4):
        synSeries1 = monthlyBeerProduction
        synSeries2 = monthlyDrugSales
        testo = "monthlyBeerProduction - monthlyDrugSales"

    if (i_ciclo == 5):
        synSeries1 = amazon
        synSeries2 = google
        testo = "amazon 2005:2021 - google 2005:2021"

    if (i_ciclo == 7):
        synSeries1 = synS1
        synSeries2 = synS2
        testo = "synS1 - synS2 - DB ST"

    if (i_ciclo == 8):
        synSeries1 = synS3
        synSeries2 = synS4
        testo = "synS3 - synS4 - DB SS"

    if (i_ciclo == 9):
        synSeries1 = synS5
        synSeries2 = synS6
        testo = "synS5 - synS6 - DB ST SS"

    if (i_ciclo == 10):
        synSeries1 = synS7
        synSeries2 = synS8
        testo = "synS7 - synS8 - DB DT"

    if (i_ciclo == 11):
        synSeries1 = synS9
        synSeries2 = synS10
        testo = "synS9 - synS10 - DB DS"

    if (i_ciclo == 12):
        synSeries1 = synS11
        synSeries2 = synS12
        testo = "synS11 - synS12 - DB DT DS"

    if (i_ciclo == 13):
        synSeries1 = synS13
        synSeries2 = synS14
        testo = "synS13 - synS14 - SB ST"


    if (i_ciclo == 14):
        synSeries1 = synS15
        synSeries2 = synS16
        testo = "synS15 - synS16 - SB SS"

    if (i_ciclo == 15):
        synSeries1 = synS17
        synSeries2 = synS18
        testo = "synS17 - synS18 - SB ST SS"

    if (i_ciclo == 16):
        synSeries1 = synS19
        synSeries2 = synS20
        testo = "synS19 - synS20 - SB DT"

    if (i_ciclo == 17):
        synSeries1 = synS21
        synSeries2 = synS22
        testo = "synS21 - synS22 - SB DS"

    if (i_ciclo == 18):
        synSeries1 = synS21
        synSeries2 = synS22
        testo = "synS22 - synS23 - SB DT DS"

    minLenght = min(len(synSeries1), len(synSeries2))
    synSeries1 = synSeries1.iloc[:minLenght]
    synSeries2 = synSeries2.iloc[:minLenght]

    # creiamo un file excel con i risultati riassuntivi dei 10 test
    os.makedirs(r'D:/Universitaa/TESI/tests/immagini/ciclo_esterno' + str(i_ciclo))
    workbook = xlsxwriter.Workbook('D:/Universitaa/TESI/tests/immagini/ciclo_esterno' + str(i_ciclo) + '/tabella_riassuntiva.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A1',str(testo))
    worksheet.write('C1','val_Pearson')
    worksheet.write('D1', 'p_Pearson')
    worksheet.write('E1', 'val_Kendall')
    worksheet.write('F1', 'p_Kendall')
    worksheet.write('G1', 'val_Spearman')
    worksheet.write('H1', 'p_Spearman')
    worksheet.write('I1', 'val_PearsonTrasf')
    worksheet.write('J1', 'p_PearsonTrasf')
    worksheet.write('K1', 'val_KendallTrasf')
    worksheet.write('L1', 'p_KendallTrasf')
    worksheet.write('M1', 'val_SpearmanTrasf')
    worksheet.write('N1', 'p_SpearmanTrasf')
    worksheet.write('O1', 'diff S1')
    worksheet.write('P1', 'YJ S1')
    worksheet.write('Q1', 'diff S2')
    worksheet.write('R1', 'YJ S2')


    worksheet.write('B2', 'Media')
    worksheet.write('C2', '=MEDIA(C4:C13)')
    worksheet.write('D2', '=MEDIA(D4:D13)')
    worksheet.write('E2', '=MEDIA(E4:E13)')
    worksheet.write('F2', '=MEDIA(F4:F13)')
    worksheet.write('G2', '=MEDIA(G4:G13)')
    worksheet.write('H2', '=MEDIA(H4:H13)')
    worksheet.write('I2', '=MEDIA(I4:I13)')
    worksheet.write('J2', '=MEDIA(J4:J13)')
    worksheet.write('K2', '=MEDIA(K4:K13)')
    worksheet.write('L2', '=MEDIA(L4:L13)')
    worksheet.write('M2', '=MEDIA(M4:M13)')
    worksheet.write('N2', '=MEDIA(N4:N13)')
    worksheet.write('O2', '=MEDIA(O4:O13)')
    worksheet.write('P2', '=MEDIA(P4:P13)')
    worksheet.write('Q2', '=MEDIA(Q4:Q13)')
    worksheet.write('R2', '=MEDIA(R4:R13)')



    worksheet.write('B3', 'Dev std')
    worksheet.write('C3', '=DEV.ST.C(C4:C13)')
    worksheet.write('D3', '=DEV.ST.C(D4:D13)')
    worksheet.write('E3', '=DEV.ST.C(E4:E3)')
    worksheet.write('F3', '=DEV.ST.C(F4:F13)')
    worksheet.write('G3', '=DEV.ST.C(G4:G13)')
    worksheet.write('H3', '=DEV.ST.C(H4:H13)')
    worksheet.write('I3', '=DEV.ST.C(I4:I13)')
    worksheet.write('J3', '=DEV.ST.C(J4:J13)')
    worksheet.write('K3', '=DEV.ST.C(K4:K13)')
    worksheet.write('L3', '=DEV.ST.C(L4:L13)')
    worksheet.write('M3', '=DEV.ST.C(M4:M13)')
    worksheet.write('N3', '=DEV.ST.C(N4:N13)')
    worksheet.write('O3', '=DEV.ST.C(O4:O13)')
    worksheet.write('P3', '=DEV.ST.C(P4:P13)')
    worksheet.write('Q3', '=DEV.ST.C(Q4:Q13)')
    worksheet.write('R3', '=DEV.ST.C(R4:R13)')



    start_test = 0
    end_test = 10

    for i_test in range(start_test,end_test+1):

        newpath = r'D:/Universitaa/TESI/tests/immagini/ciclo_esterno' + str(i_ciclo) + '/test_' + str(i_test)
        os.makedirs(newpath)

        # calcoliamo correlazione iniziale
        result1 = pearsonr(synSeries1, synSeries2)
        result2 = kendalltau(synSeries1, synSeries2)
        result3 = spearmanr(synSeries1, synSeries2)

        val_Pearson = round(result1[0],2)
        p_Pearson = round(result1[1],2)

        val_kendall = round(result2[0],2)
        p_kendall = round(result2[1],2)

        val_spearman = round(result3[0],2)
        p_spearman = round(result3[1],2)


        # vado a separare le non staz, trasformarle e ricongiungerle
        fil = open("D:/Universitaa/TESI/tests/immagini/ciclo_esterno" + str(i_ciclo) + "/test_" + str(i_test) + "/info.txt","a+")
        fil.write("Serie 1 \n")
        fil.close()
        syn1Trasf2, syn1Trasf1, syn1diff, syn1lamb = Funzioni.SeparateNonStat_Stationarize_ConcatenateTrasf(synSeries1,counter_photo,period1,period2,period3,period4,i_test,i_ciclo)
        syn1lamb= round(syn1lamb,2)
        fil = open("D:/Universitaa/TESI/tests/immagini/ciclo_esterno" + str(i_ciclo) + "/test_" + str(i_test) + "/info.txt","a+")
        fil.write("\n--------------------------------------\n")
        fil.write('Serie2 \n')
        fil.close()
        syn2Trasf2, syn2Trasf1, syn2diff, syn2lamb = Funzioni.SeparateNonStat_Stationarize_ConcatenateTrasf(synSeries2, counter_photo,period1,period2, period3,period4,i_test,i_ciclo)
        syn2lamb= round(syn2lamb,2)
        # calcoliamo correlazione finale
        resultT1 = pearsonr(syn1Trasf2, syn2Trasf2)
        resultT2 = kendalltau(syn1Trasf2, syn2Trasf2)
        resultT3 = spearmanr(syn1Trasf2, syn2Trasf2)

        val_PearsonT = round(resultT1[0], 2)
        p_PearsonT = round(resultT1[1], 2)

        val_kendallT = round(resultT2[0], 2)
        p_kendallT = round(resultT2[1], 2)

        val_spearmanT = round(resultT3[0], 2)
        p_spearmanT = round(resultT3[1], 2)

        #plotto e salvo
        plt.figure()
        plt.subplot(311)
        plt.title('diff = {}  lamb = {}'.format(syn1diff, syn1lamb))
        synSeries1.plot(color='blue', label='S1 original')
        plt.legend()
        plt.subplot(313)
        syn1Trasf2.plot(color='red', label='S1 trasformed')
        plt.legend()
        plt.savefig(
            'D:/Universitaa/TESI/tests/immagini/ciclo_esterno' + str(i_ciclo) + '/test_' + str(i_test) + '/series1.png')
        plt.show()

        plt.figure()
        plt.subplot(311)
        plt.title('diff = {}  lamb= {}'.format(syn2diff, syn2lamb))
        synSeries2.plot(color='blue', label='S2 original')
        plt.legend()
        plt.subplot(313)
        syn2Trasf2.plot(color='red', label='S2 trasformed')
        plt.legend()
        plt.savefig(
            'D:/Universitaa/TESI/tests/immagini/ciclo_esterno' + str(i_ciclo) + '/test_' + str(i_test) + '/series2.png')
        plt.show()


        # scriviamo su file txt i risultati
        fil = open("D:/Universitaa/TESI/tests/immagini/ciclo_esterno" + str(i_ciclo) + "/test_" + str(
            i_test) + "/Correlation.txt", "w+")
        fil.write(str(testo))
        fil.write("\n--------------------------------------\n")
        fil.write("Pearson INIZIALE   " + str(result1) + "\n")
        fil.write("Kendall INIZIALE " + str(result2) + "\n")
        fil.write("Spearman INIZIALE " + str(result3) + "\n")

        fil.write("\nPearson FINALE " + str(resultT1) + "\n")
        fil.write("Kendall FINALE " + str(resultT2) + "\n")
        fil.write("Spearman FINALE " + str(resultT3) + "\n")
        fil.write("--------------------------------------\n")
        fil.close()

        #scriviamo su file excel i risultati
        cell_index= 4 + i_test
        worksheet.write('B'+str(cell_index), 'test'+str(i_test))
        worksheet.write('C' + str(cell_index), val_Pearson)
        worksheet.write('D' + str(cell_index), p_Pearson)
        worksheet.write('E' + str(cell_index), val_kendall)
        worksheet.write('F' + str(cell_index), p_kendall)
        worksheet.write('G' + str(cell_index), val_spearman)
        worksheet.write('H' + str(cell_index), p_spearman)
        worksheet.write('I' + str(cell_index), val_PearsonT)
        worksheet.write('J' + str(cell_index), p_PearsonT)
        worksheet.write('K' + str(cell_index), val_kendallT)
        worksheet.write('L' + str(cell_index), p_kendallT)
        worksheet.write('M' + str(cell_index), val_spearmanT)
        worksheet.write('N' + str(cell_index), p_spearmanT)
        worksheet.write('O' + str(cell_index), syn1diff)
        worksheet.write('P' + str(cell_index), round(syn1lamb,2))
        worksheet.write('Q' + str(cell_index), syn2diff)
        worksheet.write('R' + str(cell_index), round(syn2lamb,2))





    workbook.close()








