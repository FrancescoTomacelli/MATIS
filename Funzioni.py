import numpy
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib import pyplot
from pandas import Series
from scipy import signal
from scipy.stats import yeojohnson
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
import math
from pandas import DataFrame
from fbprophet import Prophet
import pmdarima as pm
from numpy import random
import pyswarms as ps
from sklearn.metrics import mean_absolute_error
from pandas import read_csv
from statistics import mean
from pandas.plotting import autocorrelation_plot
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
from fbprophet.plot import plot_plotly, plot_components_plotly
warnings.filterwarnings("ignore")
import winsound
import xlsxwriter
from datetime import datetime
from darts import TimeSeries
from statsmodels.tsa.arima_process import arma_generate_sample
import os



counter_photo=0

#prints

def PrintSeriesTrasform(series,p0,p1):
    "ripete le trasformazioni fatte dallo swarm una volta che ha trovato i parametri migliori"
    "poi plotta la serie originale e trasformata e fa un check ricalcolando la bianco loss"
    seriesOriginal=series
    if(CheckStationarity(series)==False):
        print("la serie originale non è stazionaria")

    #qui che devo solo stampare, uso la versione shiftata della diff


        #p0=1
        #p1=0
        seriesTrasf1 = YeojohnsonTrasform(series, p0)
        seriesTrasf2 = DifferencingByParticleValue(seriesTrasf1,round(p1))

        seriesTrasf1.index=seriesOriginal.index
        seriesTrasf2.index=seriesOriginal.index



        #pyplot.title('Serie originale')
        #seriesOriginal.plot()
        #pyplot.show()

        #pyplot.title('Serie Trasformata')
        #series.plot(color='red')
        #pyplot.show()

        #pyplot.title('Serie originale (blu) vs trasformata')
        #seriesOriginal.plot()
        #series.plot(color='red')
        #pyplot.show()


        #pyplot.figure()
        #pyplot.subplot(211)
        #seriesOriginal.plot()
        #pyplot.subplot(212)
        #series.plot(color='red')
        #pyplot.show()



        #if(CheckStationarity(series)==True):
           # print("la serie trasformata è stazionaria")
       # else:
           #print("la serie trasformata non è stazionaria")

    else:
        print("la serie originale è stazionaria, nessuna trasformazione è stata effettuata")
        seriesTrasf2=seriesOriginal
        seriesTrasf1=seriesOriginal
        #quando la serie è stazionaria, non devo applicare nessuna trasformazione
        #quindi forzo il risultato della pso a Diff=0 e Yeo=1, in modo che non applichi trasformazioni
        p0=1
        p1=0


        #pyplot.title('Serie originale')
        #seriesOriginal.plot()
        #pyplot.show()

    result=[seriesTrasf2,seriesTrasf1,p0,p1]
    return result

def PlotZoomed(series,start,end):
    series[start:end].plot(color='green')
    pyplot.show()


#Checks

def CheckSeasonality(series):
    # la funzione restituisce True se la serie contiene seasonality, altrimenti false
    # lavora con i picchi dell'autocorrelazione
    # da trial and error, ho visto che una soglia ottimale per la presenza di seasonality
    # è data da un numero superiore a 2 picchi, che superano il max tra la soglia dell'autocorrelazione
    # e la soglia 0.3 imposta da me per discriminare outliers

    num_peaks=FindAutocorrelationPeaksSeason(series)
    #se ci sono più di un picco di autocorrelazione, significa che c'è un periodo
    #tranne nel caso in cui il primo picco = 1, perchè in quel caso è un periodo fittizio, come avviene per il segnale generato da un processo arma
    if(len(num_peaks)>=1 and num_peaks[0]!=1):
        thereIsSeason=True
    else:
        thereIsSeason=False

    return thereIsSeason

def CheckStationarity(series):
    #questo è il check consigliato da alessandro, piu robusto del precedente adfuller, controlla se la serie è trend stazionaria
    #a cui aggiungo sempre anche il check sul periodo
    class StationarityTests:
        def __init__(self, significance=.05):
            self.SignificanceLevel = significance
            self.pValue = None
            self.isStationary = None

        def ADF_Stationarity_Test(self, timeseries, printResults=True):
            # Dickey-Fuller test:
            adfTest = adfuller(timeseries, autolag='AIC')

            self.pValue = adfTest[1]

            if (self.pValue < self.SignificanceLevel):
                self.isStationary = True
            else:
                self.isStationary = False

            if printResults:
                dfResults = pd.Series(adfTest[0:4],
                                      index=['ADF Test Statistic', 'P-Value', '# Lags Used', '# Observations Used'])
                # Add Critical Values
                for key, value in adfTest[4].items():
                    dfResults['Critical Value (%s)' % key] = value
                print('Augmented Dickey-Fuller Test Results:')
                print(dfResults)

    sTest = StationarityTests()
    sTest.ADF_Stationarity_Test(series, printResults=False)

    #vediamo se la serie è anche season stazionaria
    #Check sesasonalityv restituisce TRUE se è presente un periodo , false altrimenti
    thereIsSeasonality = CheckSeasonality(series)

    #se la serie è trend stazionaria e non ci sono periodi, allora è stazionaria
    if(sTest.isStationary==True and thereIsSeasonality==False):
        is_stationary=True
    else:
        is_stationary=False
    return is_stationary





#Trasformazioni

def DifferencingByParticleValue(series,particle):
    #questa funzione fa differencing di un passo pari al valore della particella del PSO che gli arriva
    if(round(particle)==0):
        return series
    else:
        particle=round(particle)
        X=series.values
        diff=list()
        #metto degli 0 all'inizio per evitare che la serie si scosti rispetto agli indici temporali della serie originale
        for i in range(particle):
            value= 0
            diff.append(value)

        for i in range(particle,len(X)):
            value=X[i]-X[i-particle]
            diff.append(value)
        return Series(diff)

def DifferencingByParticleValueNonShifted(series, particle):
        # questa funzione fa differencing di un passo pari al valore della particella del PSO che gli arriva
        #senza però shiftare in avanti del passo della particella
        if (round(particle) == 0):
            return series
        else:
            particle = round(particle)
            X = series.values
            diff = list()


            for i in range(particle, len(X)):
                value = X[i] - X[i - particle]
                diff.append(value)
            return Series(diff)

def YeojohnsonTrasform(series,lamb):
    #lambda = -1. is a reciprocal transform.
    #lambda = -0.5 is a reciprocal square root transform.
    #lambda = 0.0 is a log transform.
    #lambda = 0.5 is a square root transform.
    #lambda = 1.0 is no transform
    yeoseries=Series(yeojohnson(series,lamb))
    return yeoseries

def Normalize_Series(series):
    values = series.values
    values = values.reshape((len(values), 1))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)

    normalized = scaler.transform(values)

    serNorm = [i[0] for i in normalized]
    serNorm = Series(serNorm)
    serNorm.index = series.index

    return [serNorm, scaler]


#trasformazioni inverse

def InverseYeojohnson(sertrasf,lamb):
    #in realtà seriesOriginal deve essere sempre la serieTrasf
    inv = list()
    X = sertrasf.values
    X_trans = sertrasf.values
    for i in range(0, len(sertrasf)):
        if (X[i] >= 0 and lamb == 0):
            value = math.exp(X_trans[i]) - 1
        elif (X[i] >= 0 and lamb != 0):
            value = (X_trans[i] * lamb + 1) ** (1 / lamb) - 1
        elif (X[i] < 0 and lamb != 2):
            value = 1 - (-(2 - lamb) * X_trans[i] + 1) ** (1 / (2 - lamb))
        elif (X[i] < 0 and lamb == 2):
            value = 1 - math.exp(-X_trans[i])

        inv.append(value)
    inv=Series(inv)
    return inv

def InvertDiffByParticleValue(seriesOriginal,seriesTrasf,particle):
    inv=list()
    particle=round(particle)
   # for i in range(0, particle):
       # value = 0
       # inv.append(value)

    for i in range(particle,len(seriesTrasf)):
        value=seriesTrasf[i] + seriesOriginal[i-particle]
        inv.append(value)

    return Series(inv)

def InvDiffByParticlePredicted(seriesPredicted,trainSetModified,particle):
    # inverto la diff
    if (particle != 0):
        # per invertire i primi "particle" step della serie predetta
        # lavoro con gli ultimi "particle" value del train set della serie originale
        # (perchè per invertire il valore della predizione allo step [i] mi serve il valore della serie originale allo step [i-particle]
        # che sono cioè gli ultimi "particle" step del train set
        seriesPredictedInv = list()
        for i in range(0, min(particle,len(seriesPredicted))):
            value = seriesPredicted[i] + trainSetModified[len(trainSetModified) - particle + i]
            seriesPredictedInv.append(value)

        # una volta invertiti i primi "particle" step, per predirre i successivi non posso usare piu il train set
        # ma dovrei usare dei valori che rientrano nel test set, ma visto che in teoria noi i valori reali del test set non li conosciamo
        # andiamo ad utilizzare i primi valori della predizione inverita che ci siamo calcolati prima (al ciclo for sopra)
        for i in range(particle, len(seriesPredicted)):
            value = seriesPredicted[i] + seriesPredictedInv[i - particle]
            seriesPredictedInv.append(value)

    else:
        seriesPredictedInv = seriesPredicted

    seriesPredictedInv = Series(seriesPredictedInv)

    return seriesPredictedInv

def Invert_Normalize_Series(series_normalized, scaler):
    values = series_normalized.values
    values = values.reshape((len(values), 1))

    inversed = scaler.inverse_transform(values)

    serNormInv = [i[0] for i in inversed]
    serNormInv = Series(serNormInv)
    serNormInv.index = series_normalized.index

    return serNormInv

#Utilities per checkSeasonality

def autocorrelation_plot_Modified(series, ax=None, **kwds):
    #Ho modificato la funzine originale autocorrelation_plot
    #aggiungendo un result=[x,y] che viene ritornato per trovare successivamente i valori dei picchi

    n = len(series)
    data = np.asarray(series)

    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0

    x = np.arange(n) + 1
    y = [r(loc) for loc in x]
    result=[x,y]


    return result

def FindAutocorrelationMaxLag(series):
    #la funzione trova il lag dove c'è il picco massimo di autocorrelazione
    #viene usata per dare un suggerimento alla PSO per l'inizializzazione delle particelle
    #in modo che le particelle partono a fare differencing da questo lag
    result = autocorrelation_plot_Modified(series)

    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    lenSeries = len(series)

    threshold99 = z99 / np.sqrt(lenSeries)
    threshold95 = z95 / np.sqrt(lenSeries)
    # uso la treshold99 per discriminare i picchi sotto la soglia di rilevanza
    # uso la prominence settata da me a 0.01 per discriminare picchi "fittizi" cioè picchi locali molto piccoli e irrilevanti
    peaks = signal.find_peaks(result[1], height=threshold99, prominence=0.01)

    peakLags = peaks[0]
    peakHeights = peaks[1]["peak_heights"]





    if(len(peakHeights)>0):
        peakMaxValue = max(peakHeights)

        for i in range(0, len(peakHeights)):
            if (peakHeights[i] == peakMaxValue):
                positionMaxLag = i

        maxPeakLag = peakLags[positionMaxLag]

    else:
        maxPeakLag=0

    return maxPeakLag

def GetFirstAutocorrelationLag(series):
    # la funzione trova il lag dove c'è il picco massimo di autocorrelazione
    # viene usata per dare un suggerimento alla PSO per l'inizializzazione delle particelle
    # in modo che le particelle partono a fare differencing da questo lag
    result = autocorrelation_plot_Modified(series)

    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    lenSeries = len(series)

    threshold99 = z99 / np.sqrt(lenSeries)
    threshold95 = z95 / np.sqrt(lenSeries)
    # uso la treshold99 per discriminare i picchi sotto la soglia di rilevanza
    # uso la prominence settata da me a 0.01 per discriminare picchi "fittizi" cioè picchi locali molto piccoli e irrilevanti
    peaks = signal.find_peaks(result[1], height=threshold99, prominence=0.01)

    peakLags = peaks[0]
    peakHeights = peaks[1]["peak_heights"]
    # print('peakLags2AAAAAAAAAAAAAA', peakLags)
    # print('PeakHeights2AAAAAAAAAAAAAA' , peakHeights)

    if (len(peakHeights) > 0):
        FirstPeakLag = peakLags[0]

    else:
        FirstPeakLag = 0

    return FirstPeakLag

def FindAutocorrelationMaxLag2(series):
    peakLags = GetAutocorrelationLags(series)
    peakHeights = GetAutocorrelationLagsHeights(series)

    if (len(peakHeights) > 0):
        peakMaxValue = max(peakHeights)

        for i in range(0, len(peakHeights)):
            if (peakHeights[i] == peakMaxValue):
                positionMaxLag = i

        maxPeakLag = peakLags[positionMaxLag]

    else:
        maxPeakLag = 0

    return maxPeakLag

def GetAutocorrelationLags(series):
    #la funzione trova i vari lag dove c'è un picco di autocorrelazione
    #questi lag vengono usati come initializzazione della PSO
    #in modo che le particelle partono a fare differencing da questi lag
    result = autocorrelation_plot_Modified(series)

    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    lenSeries = len(series)

    threshold99 = z99 / np.sqrt(lenSeries)
    threshold95 = z95 / np.sqrt(lenSeries)
    # uso la treshold99 per discriminare i picchi sotto la soglia di rilevanza
    # uso la prominence settata da me a 0.01 per discriminare picchi "fittizi" cioè picchi locali molto piccoli e irrilevanti
    peaks = signal.find_peaks(result[1], height=threshold99, prominence=0.001)

    peakLags = peaks[0]
    peakHeights = peaks[1]["peak_heights"]

    if(len(peakHeights)>0):
        peakLags=peakLags

    else:
        peakLags=[]


    return peakLags

def GetAutocorrelationLagsHeights(series):
    #la funzione trova i vari lag dove c'è un picco di autocorrelazione
    #questi lag vengono usati come initializzazione della PSO
    #in modo che le particelle partono a fare differencing da questi lag
    result = autocorrelation_plot_Modified(series)

    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    lenSeries = len(series)

    threshold99 = z99 / np.sqrt(lenSeries)
    threshold95 = z95 / np.sqrt(lenSeries)
    # uso la treshold99 per discriminare i picchi sotto la soglia di rilevanza
    # uso la prominence settata da me a 0.01 per discriminare picchi "fittizi" cioè picchi locali molto piccoli e irrilevanti
    peaks = signal.find_peaks(result[1], height=threshold99, prominence=0.001)

    peakLags = peaks[0]
    peakHeights = peaks[1]["peak_heights"]

    if(len(peakHeights)>0):
        peakLags=peakLags
        peakHeights = peakHeights

    else:
        peakLags=[]
        peakHeights = []


    return peakHeights

def FindAutocorrelationPeaksSeason(series):
    #ho modificato leggermente  la soglia per individuare un picco
    #per meglio catturare un'effettiva stagionalità
    #questo perchè anche in serie stazionarie, qualche picco oltre la soglia dell'autocorrelazione viene restituito, quando è palese che la serie è non periodica
    #ma succede perchè la soglia dell'autocorrealione è troppo bassa
    #per riolvere questa situazione, metto una soglia che è max(0.3, soglia autocorr99)
    #perchè ho visto che 0.35 discrimina bene gli outliers

    result = autocorrelation_plot_Modified(series)


    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    lenSeries = len(series)

    threshold99 = z99 / np.sqrt(lenSeries)
    threshold95 = z95 / np.sqrt(lenSeries)
    # uso la treshold99 per discriminare i picchi sotto la soglia di rilevanza
    # uso la prominence settata da me a 0.01 per discriminare picchi "fittizi" cioè picchi locali molto piccoli e irrilevanti
    peaks = signal.find_peaks(result[1], height=max(threshold99,0.35), prominence=0.01)

    peakLags = peaks[0]

    return peakLags



#Scores

def TrendStationarityScore(seriesOriginal,series):
    # come computo il trendStationarityScore
    # piu il punteggio è basso (negativo) e piu la serie è stazionaria
    # partiamo dal valore ADF statistic (più è piccolo e piu la serie è stazionario) restituito da adfuller()
    # se il valore è sotto la soglia del 10%, gli diamo un bonus e sottrariamo al punteggio attuale il valore della soglia arrivando a un punteggio piu basso e quindi piu stazionario
    # se è sotto la soglia del 5% gli togliamo il valore della soglia *2
    # se è sotto l'1% ci togliamo il valore della soglia *3  (perchè la soglia dell'1% è la piu difficile da superare)
    # se invece il valore ADF statistic è maggiore del valore della soglia, glielo andiamo a sommare (cambiando il segno perchè le soglie sono sempre negative), facendo aumentare il punteggio e quindi aumentare la NON stazionarietà
    # con i fattori moltiplicativi inversi, perchè essere maggiore dell 10% è piu grave dell'essere maggiore del 1%
    # quindi se è sopra il 10%  aggiungo la soglia moltiplicata per 3, se è sopra il 5% aggiungo la soglia moltiplicata per 2, se è sopra 1% aggiungo la soglia

    # null ipotesis= series is non stationary  (fail to reject= non stationary)
    # rejecting the null ipotesis = la serie è stationary
    # the more negative is the ADF statistic, the more likely we reject the null ipotesi
    # cioè the more negative is the ADF, the more la serie è trend stazionaria
    # se l'ADF è inferiore al critical value all'1%, possiamo affermare che la serie è  trend stazionaria
    # con un livello di fiducia pari all 1%, cioè la probabilità che ci stiamo sbagiando è inferiore all' 1%
    result = adfuller(series)

    TrendStationarityScoreInit = result[0]
    TrendStationarityScoreFin = TrendStationarityScoreInit

    if (TrendStationarityScoreInit < result[4]['10%']):
        TrendStationarityScoreFin = TrendStationarityScoreFin + result[4]['10%'] * 0.4
    else:
        TrendStationarityScoreFin = TrendStationarityScoreFin - result[4]['10%'] * 0.6
    if (TrendStationarityScoreInit < result[4]['5%']):
        TrendStationarityScoreFin = TrendStationarityScoreFin + result[4]['5%'] * 0.5
    else:
        TrendStationarityScoreFin = TrendStationarityScoreFin - result[4]['5%'] * 0.5
    if (TrendStationarityScoreInit < result[4]['1%']):
        TrendStationarityScoreFin = TrendStationarityScoreFin + result[4]['1%'] * 0.6
    else:
        TrendStationarityScoreFin = TrendStationarityScoreFin - result[4]['1%'] * 0.4

    #calcolo range series
    serMax = seriesOriginal.max()
    serMin = seriesOriginal.min()
    serRange = serMax - serMin
    return TrendStationarityScoreFin/serRange

def SeasonStationarityScore(series):
    # per la season stationarity, posso usare i picchi che superano la soglia dell'autocorrelation
    # quindi quantifico facendo Valore= AltezzaPicco-Soglia99
    # più il picco supera la soglia e maggliore è il valore che mi restituisce, quindi maggiore è la non stazionarietà
    # in più, faccio una sommatoria di tale scostamento dalla soglia di tutti i picchi che la superano
    # così quanti piu picchi superano la soglia, tanto piu  è probabile che ci sia una sesonality
    # perchè magari può capitare che un picco da solo supera la soglia, ma è solo un outlier

    # come interpretare il valore di SeasonStationarity Score
    # piu è basso (prossimo allo 0) e piu la serie è stazionaria per le season
    # un valore prossimo a 1 (0.8,0.9) o superiore indica la presenza di una sesonality
    # valori intorno a 0.3,0.4 indicano assenza di seasonality
    result = autocorrelation_plot_Modified(series)

    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    lenSeries = len(series)

    threshold99 = z99 / np.sqrt(lenSeries)
    threshold95 = z95 / np.sqrt(lenSeries)
    # uso la treshold99 per discriminare i picchi sotto la soglia di rilevanza
    # uso la prominence settata DA ME  a 0.01 per discriminare picchi "fittizi" cioè picchi locali molto piccoli e irrilevanti
    peaks = signal.find_peaks(result[1], height=threshold99, prominence=0.01)

    peakLags = peaks[0]
    peakHeights = peaks[1]["peak_heights"]

    SeasonStationarityScore = 0

    for i in range(len(peakHeights)):
        SeasonStationarityScore = SeasonStationarityScore + (peakHeights[i] - threshold99)


    return SeasonStationarityScore

def StationarityScore(series):
    #più è basso il punteggio e piu la serie è stazionaria
    TrendScore = TrendStationarityScore(series)
    SeasonScore = SeasonStationarityScore(series)
    StationarityScore = TrendScore + SeasonScore


    return StationarityScore

def AR1Score(seriesOriginal, seriesTrasf1, seriesTrasf2, particle, lamb):
    #questa funzione prende la serie trasformata in input
    #effettua una predizione usando un modello arma che decide in automatico i valori migliori per p,q
    #applica alla predizione la trasfromazione inversa
    #confronta la serie trasformata-predetta-invertita con il test set della serie originale
    #calcola l'errore tra le due, questo errore è un indice della information loss causata dalle trasformazioni

    # calcolo range
    serMax = seriesOriginal.max()
    serMin = seriesOriginal.min()
    serRange = serMax - serMin


    # le trasformazioni mi fanno perdere gli indici sottoforma di data
    # quindi me li ricopio dalla serie originale
    seriesTrasf1.index = seriesOriginal.index
    seriesTrasf2.index = seriesOriginal.index

    # preparo i train e test sets
    size = len(seriesOriginal)
    test = int(max((size * 0.1), particle))
    train = size - test

    # questo mi serve per confrontare la serie predetta invertita con l'originale
    seriesTrainOriginal = seriesOriginal.iloc[:-test]
    seriesTestOriginal = seriesOriginal.iloc[train:]

    # questo mi serve per fare l'inversa della diff
    seriesTrainTrasf1 = seriesTrasf1.iloc[:-test]
    seriesTestTrasf1 = seriesTrasf1.iloc[train:]

    # questo mi serve per fare la predizione
    seriesTrainTrasf2 = seriesTrasf2.iloc[:-test]
    seriesTestTrasf2 = seriesTrasf2.iloc[train:]

    test_data = seriesTestTrasf2
    train_data = seriesTrainTrasf2


    # elimino i primi "particle" value, perchè la diff shifta in avanti la serie di un passo "particle" mettendo un numero di 0 pari a particle
    # per non falsare la predizione, bisogna togliere questi 0 che shiftano
    train_data = train_data.drop(train_data.index[0:particle])
    try:
        model = ARIMA(train_data, order=(1, 0, 0))

        model_fit = model.fit(disp=0)

        forecast = model_fit.predict(start=len(train_data), end=len(test_data) + len(train_data) - 1)

        seriesPredicted = Series(forecast)
        seriesPredicted.index = seriesTestTrasf2.index
        # seriesPredicted.plot(color='red')
        # seriesTestTrasf2.plot()
        # plt.show()

        # inverto la predizione
        seriesPredictedInv = InvDiffByParticlePredicted(seriesPredicted, seriesTrainTrasf1, particle)
        seriesPredictedInv = InverseYeojohnson(seriesPredictedInv,lamb)

        # calcolo l'errore tra test set originale e serie trasformata predetta invertita

        seriesTestOriginal = Series(seriesTestOriginal.values)
        rmse = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedInv))
        rmseRange = rmse / serRange

        # seriesPredictedInv.plot(color='red', label='Predicted')
        # seriesTestOriginal.plot(label='Original')
        # plt.legend()
        # plt.title("Particle = {}   lambda= {}   rmse/range={}".format(particle, lamb, rmseRange))
        # plt.show()


        return rmseRange

    except:
        #print('Errore AutoAR1 , particle= {}  lamb={}'.format(particle,lamb))
        return 9999999

def Quantif_Autocorr_Residual(seriesPredicted, seriesTest):
    resid = list()
    for i in range(0, len(seriesPredicted)):
        data = seriesTest[i] - seriesPredicted[i]
        resid.append(data)

    resid = Series(resid)
    resid.index = seriesPredicted.index

    z95 = 1.959963984540054
    threshold95 = z95 / np.sqrt(len(resid))

    result = autocorrelation_plot_Modified(resid)
    autocorrValues = result[1]

    autocorrQuantif = 0
    for i in range(0, len(autocorrValues)):
        if (autocorrValues[i] > threshold95):
            autocorrQuantif = autocorrQuantif + (autocorrValues[i] - threshold95)

    return autocorrQuantif


#stationarizaty functions

def StationarizeWithPSO_Original(series,i_test,i_ciclo):
    #è la funzione stationarize originale
    #è la PSO che viene usata sui pezzi di serie estratti


    #qui viene inizializzato l'intero swarm  per la diff al picco massimo di autocorrelazione della serie
    #la yeojohnson viene inizializzata ad 1

    #questa funzione viene usata in windowed statioanrize
    seriesOriginal = series

    # operazioni per PSO
    lenSeriesReduced = round(len(series) / 1)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    min_bound = np.array([0, 0])
    max_bound = np.array([1.0, lenSeriesReduced])
    bounds = (min_bound, max_bound)
    dimensions = 2

    # operazioni per inizializzare swarm
    swarm_size = 10
    num_iters = 10
    yeo_init = random.uniform(min_bound[0],max_bound[0])  # valora casuale per inizializzare le particle della yeotrasform

    init_lag = FindAutocorrelationMaxLag(series)
    print("INIT LAG = ", init_lag)
    auto_lags= GetAutocorrelationLags(series)
    init_value = [1.0,init_lag]  # metto 1.0 come inizializzazione della yeo, perchè voglio partire dal caso piu semplice, cioè non applica la yeo ma applica la diff di lag=maxAutocorrelationLag

    initialize = np.array([init_value for i in range(swarm_size)])

    #imposto questa seconda inizializzazione, per prendere i primi "swarm" valori dei lag di autocorrelation e non solo il Max
    if(len(auto_lags)==0):
        auto_lags=[0,1,7,12]
    initialize2 = list()
    k = 0
    initialize2.append([1.0, init_lag])
    initialize2.append([1.0, init_lag])
    for i in range(swarm_size-2):
        count = len(auto_lags)

        yeo = 1.0
        lag = auto_lags[k]

        k = k + 1
        if (k == count):
            k = 0

        initialize2.append([yeo, lag])

    initialize2 = np.array(initialize2)

    #anche se alla fine l'inizializzazione che prendo è quella originale (inizialize)
    #optimizer= ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dimensions, bounds=bounds, options=options)
    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dimensions, bounds=bounds, options=options, init_pos=initialize)
    cost, pos = optimizer.optimize(f, iters=num_iters,ser=seriesOriginal)



    # rileggo la serie per plottarla con PrintSeriesTrasform
    result = PrintSeriesTrasform(series, pos[0], pos[1])
    seriesTrasf2 = result[0]
    seriesTrasf1= result[1]
    pos[0] = result[2]
    pos[1] = result[3]

    print(f"Valore minimo: {cost}, Yeojohnson lambda={pos[0]}, DiffByParticle={round(pos[1])}")

    #elimino i primi particle value che sono messi a 0, mantenendo però gli indici
    #seriesTrasformed=seriesTrasformed.drop(seriesTrasformed.index[0:round(pos[1])])


    # StationarityScore= Funzioni.StationarityScore(seriesTrasformed)
    TrendScore = TrendStationarityScore(seriesOriginal, seriesTrasf2)
    SeasonScore = SeasonStationarityScore(seriesTrasf2)
    Ar1Score= cost - TrendScore - SeasonScore

    print("Ar1Score", cost - TrendScore - SeasonScore)
    print("TrendScore =", TrendScore)
    print("SeasonScore =", SeasonScore)
    # Funzioni.PlotZoomed(seriesOriginal,300,400)

    fil=open("D:/Universitaa/TESI/tests/immagini/ciclo_esterno" + str(i_ciclo) + "/test_"+str(i_test)+"/info.txt","a+")
    print(f"Valore minimo: {cost}, Yeojohnson lambda={pos[0]}, DiffByParticle={round(pos[1])}")
    fil.write('\nValore Minimo = '+str(cost)+' Yeojohnson lambda = '+ str(pos[0]) + ' DiffByParticle = '+str(round(pos[1]))+'\n')
    fil.write('Ar1Score = ' + str(Ar1Score) + ' TrendScore = ' + str(TrendScore)+ ' SeasonScore = '+ str(SeasonScore) +'\n\n')
    fil.close()

   #result2=[seriesTrasf2,seriesTrasf1,round(pos[1]),pos[0],TrendScore,SeasonScore,Ar1Score,cost]
    result2 = [seriesTrasf2, seriesTrasf1, round(pos[1]), pos[0], TrendScore, SeasonScore, Ar1Score, cost]
    return result2

def StationarizeWithPSO(series):
    #questa è la PSO che viene usata dalle finestre

    #qui invece lo swarm viene inizializzato con le prime 3  particelle pari al max_autocorr_lag, mentre le altre vengono inizializzate ad altri picchi di autocorrelazione
    #questo per fare "spaziare" di più la PSO
    #la yeojohnson viene sempre inizializzata ad 1
    seriesOriginal = series

    # operazioni per PSO
    lenSeriesReduced = round(len(series) / 1)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    min_bound = np.array([0, 0])
    max_bound = np.array([1.0, lenSeriesReduced])
    bounds = (min_bound, max_bound)
    dimensions = 2

    # operazioni per inizializzare swarm
    swarm_size = 10
    num_iters = 10
    yeo_init = random.uniform(min_bound[0],max_bound[0])  # valora casuale per inizializzare le particle della yeotrasform
    init_lag = FindAutocorrelationMaxLag(series)
    auto_lags= GetAutocorrelationLags(series)
    init_value = [1.0,init_lag]  # metto 1.0 come inizializzazione della yeo, perchè voglio partire dal caso piu semplice, cioè non applica la yeo ma applica la diff di lag=maxAutocorrelationLag
    initialize = np.array([init_value for i in range(swarm_size)])

    #uso questa seconda inizializzazione, per prendere i primi "swarm" valori dei lag di autocorrelation e non solo il Max
    if(len(auto_lags)==0):
        auto_lags=[0,1,7]
    initialize2 = list()
    k = 0
    initialize2.append([1.0, init_lag])
    initialize2.append([1.0, init_lag])
    for i in range(swarm_size-2):
        count = len(auto_lags)

        yeo = 1.0
        lag = auto_lags[k]

        k = k + 1
        if (k == count):
            k = 0

        initialize2.append([yeo, lag])

    initialize2 = np.array(initialize2)



    #optimizer= ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dimensions, bounds=bounds, options=options)
    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dimensions, bounds=bounds, options=options,init_pos=initialize2)
    cost, pos = optimizer.optimize(f, iters=num_iters,ser=seriesOriginal)



    # rileggo la serie per plottarla con PrintSeriesTrasform
    result = PrintSeriesTrasformWindow(series, pos[0], pos[1])
    seriesTrasf2 = result[0]
    seriesTrasf1= result[1]
    pos[0] = result[2]
    pos[1] = result[3]
    print(f"Valore minimo: {cost}, Yeojohnson lambda={pos[0]}, DiffByParticle={round(pos[1])}")

    #elimino i primi particle value che sono messi a 0, mantenendo però gli indici
    #seriesTrasformed=seriesTrasformed.drop(seriesTrasformed.index[0:round(pos[1])])


    # StationarityScore= Funzioni.StationarityScore(seriesTrasformed)
    TrendScore = TrendStationarityScoreADJ(seriesOriginal, seriesTrasf2)
    SeasonScore = SeasonStationarityScore(seriesTrasf2)
    Ar1Score= cost - TrendScore - SeasonScore
    print("Ar1Score", Ar1Score)
    print("TrendScore =", TrendScore)
    print("SeasonScore =", SeasonScore)
    # Funzioni.PlotZoomed(seriesOriginal,300,400)

   #result2=[seriesTrasf2,seriesTrasf1,round(pos[1]),pos[0],TrendScore,SeasonScore,Ar1Score,cost]
    result2 = [seriesTrasf2, seriesTrasf1, round(pos[1]), pos[0], TrendScore, SeasonScore, Ar1Score, cost]
    return result2

def Stationarize_PSO_Window4(series,counter_photo,period1,period2,period3,period4,i_test,i_ciclo):

    #questa è la versione per 2ndTestWindow (90)

    #la funzione prende in input una serie contenente diverse non stazionarietà
    #restituisce in output la serie scomposta in sottoserie in base alla non stazionarietà

    # la funzione crea delle finestre per studiare come varia la non stazionarietà della seire, andando ad analizzare come varia la trasformazione applicata dalla PSO nel tempo
    # per rendere le finestre piu generali possibili, ho scelto una grandezza di 5*maxAutocorrelationLag, in modo da essere sicuri di catturare eventuali periodicità
    # la dimensione della window cambia nel tempo, quando vengono individuati cambiamnti significativi della diff (e.g non multipli e non valori vicini)
    # e quando viene identificato anche un cambiamento  significaivo dell' maxAutocorrelation lag
    # a quel punto la serie analizzata fino a quel momento viene droppata, viene ricalcolato il maxAutocorrelationLag sulla serie rimanente e viene ricalcolata la window

    max_autocorrelation_lag = FindAutocorrelationMaxLag(series)
    #nel caso in cui non riesce a trovare un max_autocorr_lag all'inizio, a causa delle troppe non stazionarietà che confondono l'autocorrelazione
    #inizializzo max_auto_lag a 30, in modo da avere una generica finestra di 150, che poi si adatterà successivamente da sola
    if(max_autocorrelation_lag==0):
        max_autocorrelation_lag=30

    autocorrelation_peaks = GetAutocorrelationLags(series)
    autocorrelation_heights = GetAutocorrelationLagsHeights(series)
    list_par = list()
    list_lamb = list()
    list_score = list()
    list_window = list()
    list_series_extracted=list()
    list_autocorrelation_lags=list()

    i = 0  # mi fa muovere lungo la serie
    wind = 5 * max_autocorrelation_lag  # è l'ampiezza della finestra
    x = 0  # l'inizio della finestra
    y = wind  # la fine della finestra
    Count = 0  # mi serve come condizione per analizzare alla fine la serie completa
    lastLap = False  #serve per fare l'ultima analisi con windows=len(series)
    change = False # mi serve per fare la correzione dei lag nell'iterazione in cui c'è cambio di window
    change_station = False  # indica se c'è stato un cambio di stationarietà, serve per estrarre l'ultimo pezzo della serie con non-stazionarietà diversa
    oldCheckPoint = 0  # inizio di una porzione di serie con una certa non stazionarietà
    newCheckPoint = 0  # fine di una porizione di serie con una certa non stazionarietà
    num_nonStat_find = 1  #serve per tenere traccia di quanti "pezzi di non stazionarietà" sono contenuti nella serie
    counter_stationarity = 0 #mi serve per contare quante volte capita che la PSO applica diff=0 e lamb=1.0, cioè non applica trasformazioni , perchè se succede spesso allora non taglio la serie ma la considero nella sua interezza
    while (Count < 2):
        if (Count == 1):
            Count = 2
        batch = series.iloc[x:y]

        seriesOriginal = batch
        try:
            result = StationarizeWithPSO(batch)


        except:

            seriesExtracted=list()
            seriesExtracted.append(series)
            return seriesExtracted

        #se la PSO restituisce diff=0 e lamb=1 significa che la serie è stazionaria, o ci sono delle stazionarietà all'interno della serie, che manderebbero in errore il programma
        #per questo restituisco semplicemente la serie nella sua totalità

        if(result[2]==0 and result[3]==1.0):
            counter_stationarity = counter_stationarity+1
            if(counter_stationarity == 3):
                seriesExtracted = list()
                seriesExtracted.append(series)
                return seriesExtracted

        lagBatch= FindAutocorrelationMaxLag2(batch)
        #print('lagBatch   ', lagBatch)
        seriesTrasf2 = result[0]
        list_par.append(result[2])
        list_lamb.append(round(result[3], 2))
        list_score.append(round(result[7], 2))
        list_window.append((x, y, wind))


        plt.figure()
        plt.subplot(311)
        plt.title('window= {} Part= {} lamb= {} score={}'.format(list_window[i], list_par[i], list_lamb[i], list_score[i]))
        series.plot()
        batch.plot(color='red')
        plt.subplot(312)
        batch.plot(color='red')
        plt.subplot(313)
        seriesTrasf2.plot(color='green')
        pyplot.savefig('D:/Universitaa/TESI/tests/immagini/ciclo_esterno' + str(i_ciclo) + '/test_'+str(i_test)+'/Syn_'+str(period1)+'_'+str(period2)+'_'+str(period3)+'_'+ str(period4)+'_'+ str(counter_photo) + '_.png')
        counter_photo = counter_photo + 1
        plt.show()



        #questo if mi serve per aggiornare il max_autocorr_lag solo nel caso in cui il max_lag visto nella finestra è cambiato in modo significativo
        if(lagBatch!=0 and (lagBatch<max_autocorrelation_lag-2 or lagBatch>max_autocorrelation_lag+2)):
           max_autocorrelation_lag=lagBatch

        #questo if mi serve per aggiornare il max_autocorr_lag solo nel caso in cui ci sono stati cambiamenti significativi della diff
        if ((list_par[i] > list_par[i - 1] + 3 or list_par[i] < list_par[i - 1] - 3) ):
            max_autocorrelation_lag=lagBatch

        # quando c'è un cambiamento nella diff applicata, allora potrebbe significare che c'è un cambiamento di non stazionarietà
        # visto che a volte la diff scelta dalla PSO si confonde con la diff giusta e i suoi multipli, faccio un check per controllare se c'è stato un effettivo cambiamento significativo (la diff che si  muove da un multiplo all'altro non è significativo)
        print('Autocorrelation lag =', max_autocorrelation_lag)
        list_autocorrelation_lags.append(max_autocorrelation_lag)

        #questo if serve ad accorgersi del cambio di non stazionarietà, andando a confrontare gli ultimi 2 valori di max_autocorr_lag registrati
        #se i due valori si discostano in modo significativo, allora la non stazionarietà potrebbe essere cambiata

        if ((list_autocorrelation_lags[i] > list_autocorrelation_lags[i - 1] + 2 or list_autocorrelation_lags[i] < list_autocorrelation_lags[i - 1] - 2) and lastLap==False):
            #quindi vado a ricalcolare il max_autocorrelation_lag con ciò che rimane della serie, droppando la parte analizzata fin ora

            # rimuovo la serie analizzata fin ora
            seriesHalf = series.drop(series.index[0:y])
            # ricalcolo il maxAutocorrelationLag con la serie rimanente
            New_max_autocorrelation_lag = FindAutocorrelationMaxLag2(seriesHalf)

            max_autocorrelation_lag = New_max_autocorrelation_lag

            change=True
            change_station = True
            num_nonStat_find=num_nonStat_find+1


            # estraggo la porzione di serie vista fino ad ora, che avrà una sua non stazionarietà, diversa dalle altre porzioni di serie
            #sottraggo (wind/2) per essere sicuro di non prendere i valori transitori tra una serie e l'altra
            newCheckPoint = x
            #print('********************************')
            #print(newCheckPoint)
            seriesExtracted = series[oldCheckPoint:newCheckPoint]
            list_series_extracted.append(seriesExtracted)
            oldCheckPoint = y

        # una volta ricalcolato il max_autocorrelation lag, ricalcolo la dimensione della window

        wind = 5 * max_autocorrelation_lag
        x = y
        y = min(len(series), y+wind )

        #questo if serve per aggiornare la finestra a seguito di un cambio di non-stazionarietà
        if(change==True):
            batch = series.iloc[x:y]
            lagBatch = FindAutocorrelationMaxLag2(batch)
            if (lagBatch != 0):
                max_autocorrelation_lag = lagBatch
                list_autocorrelation_lags[i] = max_autocorrelation_lag
            change=False

        # se la window arriva all'ultimo valore della serie
        # fa un'ultima analisi con una window pari alla dimensione della serie
        # così da fare un'analisi della serie nella sua interezza
        if (y == len(series) and Count == 0):
            Count = 1
            x = 0
            oldwind=wind
            wind = len(series)
            lastLap=True

            if (change_station == True):
                seriesExtracted = series[oldCheckPoint:y]
                list_series_extracted.append(seriesExtracted)
            else:
                list_series_extracted.append(series)
        i = i + 1

    print('list_par: ', list_par)
    print('list_lamb: ', list_lamb)
    print('list_score', list_score)
    print('list_window', list_window)
    print('num_nonStat_find ', num_nonStat_find)

    fil=open("D:/Universitaa/TESI/tests/immagini/ciclo_esterno" + str(i_ciclo) + "/test_"+str(i_test)+"/info.txt","a+")
    #fil.write("Period 1 = "+str(period1)+" Period 2 = "+str(period2)+" Period3 = "+str(period3)+ " Period4 =" +str(period4)+"\n")
    fil.write('Differencing applied :  '+str(list_par)+' \n')
    fil.write('Y.J. Lambda applied : ' + str(list_lamb)+' \n')
    fil.write('Scores : '+ str(list_score)+' \n')
    fil.write('Windows : '+ str(list_window)+ ' \n')
    fil.write('Numero non stazionarietà trovate : ' + str(num_nonStat_find) + '\n')
    fil.close()

    #k=1
   # for ser in list_series_extracted:
        #ser.plot(color='red')
        #pyplot.savefig('D:/Universitaa/TESI/tests/immagini/series_extracted_' + str(k) + '_.png')
       # plt.show()
       # k=k+1

    return list_series_extracted



#predictions

def ProphetPredictSeries(series,size,train,test):
    # costruisco il dataframe ad hoc per prophet, a partire dalla serie originale
    df = DataFrame()
    df["ds"] = [series.index[i] for i in range(len(series))]
    df["y"] = [series[i] for i in range(len(series))]


    # divido il dataset in train e test

    dfTrain = df.iloc[:-test]
    dfTest = df.iloc[train:]

    seriesTest = Series(dfTest["y"])
    seriesTrain = Series(dfTrain["y"])

    # indico gli step futuri di cui fare la predizione
    future = DataFrame(dfTest["ds"])

    m = Prophet()
    m.fit(df)

    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    # trasformo le predizioni in serie, seguendo l'index di dfTest
    forecastValues = forecast['yhat']
    seriesForecast = Series(forecastValues)
    seriesForecast.index = dfTest.index

    rmse=sqrt(mean_squared_error(seriesTest, seriesForecast))

    result=[seriesForecast,rmse,seriesTest,seriesTrain]

    return result

def ProphetPredictSeries_Window90(train_set,test_set):
    # costruisco il dataframe ad hoc per prophet, a partire dalla serie originale
    dfTrain = DataFrame()
    dfTrain["ds"] = [train_set.index[i] for i in range(len(train_set))]
    dfTrain["y"] = [train_set[i] for i in range(len(train_set))]

    dfTest = DataFrame()
    dfTest["ds"] = [test_set.index[i] for i in range(len(test_set))]
    dfTest["y"] = [test_set.index[i] for i in range(len(test_set))]

    seriesTest = Series(dfTest["y"])
    seriesTrain = Series(dfTrain["y"])

    # indico gli step futuri di cui fare la predizione
    future = DataFrame(dfTest["ds"])

    m = Prophet()
    plt.title('TrainSet prophet')
    train_set.plot(color='yellow')
    plt.show()
    m.fit(dfTrain)

    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    # trasformo le predizioni in serie, seguendo l'index di dfTest
    forecastValues = forecast['yhat']
    seriesForecast = Series(forecastValues)
    seriesForecast.index = dfTest.index

    result=[seriesForecast]

    return result

def LSTM_Prediction2_Window90_Ale(training_set, test_set):
    #modello LSTM di alesasandro
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.models import RNNModel

    train, val = TimeSeries.from_series(training_set), TimeSeries.from_series(test_set,freq='D')

    plt.title('TrainSet LSTM')
    train.plot(color='yellow')
    plt.show()

    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    val_transformed = transformer.transform(val)

    #input_chunk = round(len(train) / 8)
    input_chunk= 8

    my_model = RNNModel(
        model='LSTM',
        input_chunk_length=input_chunk,
        output_chunk_length=1, model_name='My_LSTM'
    )
    plt.title('TrainSetTrasformed LSTM')
    train_transformed.plot(color='yellow')
    plt.show()

    my_model.fit(train_transformed, val_series=val_transformed, verbose=False)
    #best_model = RNNModel.load_from_checkpoint(model_name='My_LSTM', best=True)
    #forecast = best_model.predict(len(val))
    forecast = my_model.predict(len(val))

    forecast = transformer.inverse_transform(forecast)


    forecast = forecast.pd_series()

    return forecast

def TestPrediction_AutoArima_Prophet_LSTM_Window90_Ale(seriesOriginal,seriesTrasf1,seriesTrasf2,particle,lamb,counter_photo,train_set,test_set,scaler2,i_test,i_ciclo,workbook,worksheet,list_diff,list_lamb):
    #questa versione è diversa semplicemente perchè applica il modello LSTM di alessandro

    #questa è la versione per fare i test con la trasformazione del primo 90% usando le finestre (scomponendo le non stazionarietà e trasformandole), ricollegando le trasformate e usando la serie trasformata così fatta per la predizione
    # preparo i train e test sets
    size = len(seriesOriginal)
    test = len(test_set)
    train = len(train_set)

    seriesTrainOriginal = train_set
    seriesTestOriginal = test_set

    seriesTrainTrasf1 = seriesTrasf1


    seriesTrainTrasf2 = seriesTrasf2
    #seriesTestTrasf2 = seriesTrasf2.iloc[train:]

    train_dataOriginal = seriesTrainOriginal
    test_dataOriginal = seriesTestOriginal

    train_dataTrasf2 = seriesTrainTrasf2
    #test_dataTrasf2 = seriesTestTrasf2

    # facciamo la predizione con auto ARIMA della serie originale e calcoliamo rmse

    modelOriginal = pm.auto_arima(train_dataOriginal, error_action='ignore', trace=True, suppress_warnings=True)
    forecastOriginal = modelOriginal.predict(test_dataOriginal.shape[0])

    seriesPredictedOriginal = Series(forecastOriginal)
    seriesPredictedOriginal.index = seriesTestOriginal.index

    rmseOriginal = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedOriginal))
    maeOriginal = mean_absolute_error(seriesTestOriginal, seriesPredictedOriginal)
    autocorr_residOriginal = Quantif_Autocorr_Residual(seriesPredictedOriginal, seriesTestOriginal)

    # facciamo la predizione con auto ARIMA della serie trasformata e calcoliamo rmse
    # elimino i primi particle value settati a 0 che mi sballano il training
    #train_dataTrasf2 = train_dataTrasf2.drop(train_dataTrasf2.index[0:particle])
    #plt.title('TrainSet autoarima')
    #seriesTrasf2.plot(color='yellow')
    #plt.show()
    modelTrasf2 = pm.auto_arima(seriesTrasf2, error_action='ignore', trace=True, suppress_warnings=True)
    forecastTrasf2 = modelTrasf2.predict(test_dataOriginal.shape[0])

    seriesPredictedTrasf2 = Series(forecastTrasf2)

    #de-normalizzo la predizione
    seriesPredictedTrasf2 = Invert_Normalize_Series(seriesPredictedTrasf2,scaler2)

    #plt.title('AutoArima')
    #seriesPredictedTrasf2.plot(color='violet')
    #plt.show()
    # inverto la predizione
    seriesPredictedInv = InvDiffByParticlePredicted(seriesPredictedTrasf2, seriesTrainTrasf1, particle)
    seriesPredictedInv = InverseYeojohnson(seriesPredictedInv, seriesPredictedInv, lamb)

    seriesPredictedInv.index = seriesTestOriginal.index

    rmseTrasf2 = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedInv))
    maeTrasf2 = mean_absolute_error(seriesTestOriginal, seriesPredictedInv)
    autocorr_residTrasf2 = Quantif_Autocorr_Residual(seriesPredictedInv, seriesTestOriginal)


    # plottiamo le predizioni
    seriesPredictedTrasf2.index = seriesTestOriginal.index
    pyplot.title("AutoARIMA prediction series trasformed before inverting")
    #seriesPredictedTrasf2.plot(color='violet')
    #base.index = seriesPredictedTrasf2.index
    #base.plot(color='orange')
    #seriesTestTrasf2.plot()
    #pyplot.show()

    pyplot.figure()
    pyplot.subplot(211)
    train_set.plot(color='blue', label='Original')
    pyplot.legend()
    pyplot.title('Last Transformation Applied  Particle={}  lambda={:.2f}'.format(particle,lamb))
    pyplot.subplot(212)
    seriesTrasf2.plot(color='green', label='Trasformed')
    pyplot.legend()
    pyplot.savefig('D:/Universitaa/TESI/tests/immagini/ciclo_esterno'+ str(i_ciclo) + '/test_'+str(i_test)+'/Syn_' + str(counter_photo) + '.png')
    counter_photo = counter_photo + 1
    pyplot.show()

    pyplot.figure()
    pyplot.subplot(311)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedOriginal.plot(color='orange', label='PredictedOriginal')
    pyplot.legend()
    pyplot.title("AutoARIMAOrigin  rmse={:.2f}  mae={:.2f}  autocorrRes={:.2f}".format(rmseOriginal, maeOriginal, autocorr_residOriginal))
    pyplot.subplot(313)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedInv.plot(color='red', label='PredictedTrasf')
    pyplot.legend()
    pyplot.title("AutoARIMATrasf rmse={:.2f}   mae={:.2f}  autcorrRes={:.2f}".format(rmseTrasf2,maeTrasf2, autocorr_residTrasf2))

    pyplot.savefig('D:/Universitaa/TESI/tests/immagini/ciclo_esterno' + str(i_ciclo) + '/test_'+str(i_test)+'/Syn_'+ str(counter_photo) +'.png')
    counter_photo = counter_photo + 1
    pyplot.show()

    # facciamo le predizioni usando prophet
    # facciamo predizione usando serie originale

    result1 = ProphetPredictSeries_Window90(train_dataOriginal, test_dataOriginal)
    forecastOriginalProp = result1[0]

    seriesPredictedOriginalProp = Series(forecastOriginalProp)
    seriesPredictedOriginalProp.index = seriesTestOriginal.index

    rmseOriginalProp = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedOriginalProp))
    maeOriginalProp = mean_absolute_error(seriesTestOriginal, seriesPredictedOriginalProp)
    autocorr_residOriginal_Prop = Quantif_Autocorr_Residual(seriesPredictedOriginalProp, seriesTestOriginal)
    # facciamo la predizione usando la serie trasformata
    result = ProphetPredictSeries_Window90(seriesTrasf2, test_dataOriginal)
    forecastTrasf2Prop = result[0]

    seriesPredictedTrasf2Prop = Series(forecastTrasf2Prop.values)

    # de-normalizzo la predizione
    seriesPredictedTrasf2Prop = Invert_Normalize_Series(seriesPredictedTrasf2Prop, scaler2)

    #plt.title('Prophet')
    #seriesPredictedTrasf2Prop.plot(color='violet')
    #plt.show()
    seriesTrainTrasf1Val = Series(seriesTrainTrasf1.values)
    # inverto la predizione
    seriesPredictedInvProp = InvDiffByParticlePredicted(seriesPredictedTrasf2Prop, seriesTrainTrasf1Val,particle)
    seriesPredictedInvProp = InverseYeojohnson(seriesPredictedInvProp, seriesPredictedInvProp, lamb)

    seriesPredictedInvProp.index = seriesTestOriginal.index

    rmseTrasf2Prop = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedInvProp))
    maeTrasf2Prop = mean_absolute_error(seriesTestOriginal, seriesPredictedInvProp)
    autocorr_residTrasf2_Prop = Quantif_Autocorr_Residual(seriesPredictedInvProp, seriesTestOriginal)


    #plottiamo le predizioni
    seriesPredictedTrasf2Prop.index = seriesTestOriginal.index
    #pyplot.title("Prophet prediction series trasformed before inverting")
    #seriesPredictedTrasf2Prop.plot(color='violet')
    #seriesTestTrasf2.plot()
    #base.plot(color='orange')
    #pyplot.show()

    pyplot.figure()
    pyplot.subplot(311)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedOriginalProp.plot(color='orange', label='PredictedOriginal')
    pyplot.legend()
    pyplot.title("ProphetOrigin rmse={:.2f}  mae={:.2f} autocorrRes={:.2f}  ".format(rmseOriginalProp, maeOriginalProp,autocorr_residOriginal_Prop))
    pyplot.subplot(313)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedInvProp.plot(color='red', label='PredictedTrasf')
    pyplot.legend()
    pyplot.title("ProphetTrasf rmse={:.2f}  mae={:.2f}, autocorrRes={:.2f}".format(rmseTrasf2Prop,maeTrasf2Prop,autocorr_residTrasf2_Prop))

    pyplot.savefig('D:/Universitaa/TESI/tests/immagini/ciclo_esterno' + str(i_ciclo) + '/test_'+str(i_test)+'/Syn_'+ str(counter_photo) +'.png')
    counter_photo = counter_photo + 1
    pyplot.show()

    # facciamo la predizione con LSTM
    # iniziamo con la serie originale
    seriesPredictedOriginalLSTM = LSTM_Prediction2_Window90_Ale(train_dataOriginal, test_dataOriginal)
    #calcoliamo rmse,mae e quantifichiamo l'autocorrelazione del residuo




    rmseOriginalLSTM = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedOriginalLSTM))
    maeOriginalLSTM = mean_absolute_error(seriesTestOriginal, seriesPredictedOriginalLSTM)
    autocorr_residOriginal_LSTM= Quantif_Autocorr_Residual(seriesPredictedOriginalLSTM,seriesTestOriginal)

    # facciamo la predizione usando la serie trasformata
    seriesPredictedTrasf2LSTM = LSTM_Prediction2_Window90_Ale(seriesTrasf2, test_dataOriginal)




    # de-normalizzo la predizione
    seriesPredictedTrasf2LSTM = Invert_Normalize_Series(seriesPredictedTrasf2LSTM, scaler2)

    #plt.title('LSTM, before inverting')
    #seriesPredictedTrasf2LSTM.plot(color='violet')
    #base.plot(color='orange')
    #plt.show()

    # inverto la predizione
    seriesPredictedInvLSTM = InvDiffByParticlePredicted(seriesPredictedTrasf2LSTM, seriesTrainTrasf1, particle)
    seriesPredictedInvLSTM = InverseYeojohnson(seriesPredictedInvLSTM, seriesPredictedInvLSTM, lamb)
    seriesPredictedInvLSTM.index = seriesTestOriginal.index
    # calcoliamo rmse,mae e quantifichiamo l'autocorrelazione del residuo
    rmseTrasf2LSTM = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedInvLSTM))
    maeTrasf2LSTM = mean_absolute_error(seriesTestOriginal, seriesPredictedInvLSTM)
    autocorr_residTrasf2_LSTM = Quantif_Autocorr_Residual(seriesPredictedInvLSTM, seriesTestOriginal)

    #plottiamo la predizione di LSTM
    pyplot.figure()
    pyplot.subplot(311)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedOriginalLSTM.plot(color='orange', label='PredictedOriginal')
    pyplot.legend()
    pyplot.title("LSTMOrigin rmse={:.2f}  mae={:.2f}  autocorrRes={:.2f}".format(rmseOriginalLSTM, maeOriginalLSTM,autocorr_residOriginal_LSTM))
    pyplot.subplot(313)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedInvLSTM.plot(color='red', label='PredictedTrasf')
    pyplot.legend()
    pyplot.title("LSTMTrasf rmse={:.2f} mae={:.2f}   autocorrRes={:.2f}".format(rmseTrasf2LSTM,maeTrasf2LSTM,autocorr_residTrasf2_LSTM))
    pyplot.savefig('D:/Universitaa/TESI/tests/immagini/ciclo_esterno'+ str(i_ciclo) + '/test_'+str(i_test)+'/Syn_'+ str(counter_photo) +'.png')
    counter_photo = counter_photo + 1
    pyplot.show()

    #scriviamo i valori di rmse,mae,autocorr_res su file
    fil = open("D:/Universitaa/TESI/tests/immagini/ciclo_esterno" + str(i_ciclo) + "/test_"+str(i_test)+"/info.txt", "a+")

    fil.write('\n\nAuto_Arima_Original :  Rmse =' + str(rmseOriginal) + ' Mae = '+ str(maeOriginal) + ' Autocorr_resid = '+ str(autocorr_residOriginal) + ' \n')
    fil.write('Prophet_Original :  Rmse =' + str(rmseOriginalProp) + ' Mae = ' + str(maeOriginalProp) + ' Autocorr_resid = ' + str( autocorr_residOriginal_Prop) + ' \n')
    fil.write('LSTM_Original :  Rmse =' + str(rmseOriginalLSTM) + ' Mae = ' + str(maeOriginalLSTM) + ' Autocorr_resid = ' + str(autocorr_residOriginal_LSTM) + ' \n')

    fil.write('\n\nAuto_Arima_Trasf :  Rmse =' + str(rmseTrasf2) + ' Mae = ' + str(maeTrasf2) + ' Autocorr_resid = ' + str(autocorr_residTrasf2) + ' \n')
    fil.write('Proohet_Trasf :  Rmse =' + str(rmseTrasf2Prop) + ' Mae = ' + str(maeTrasf2Prop) + ' Autocorr_resid = ' + str(autocorr_residTrasf2_Prop) + ' \n')
    fil.write('LSTM_Trasf :  Rmse =' + str(rmseTrasf2LSTM) + ' Mae = ' + str(maeTrasf2LSTM) + ' Autocorr_resid = ' + str(autocorr_residTrasf2_LSTM) + ' \n')
    fil.close()

    Create_Excel_File(i_test, rmseOriginal, maeOriginal, autocorr_residOriginal,rmseOriginalProp, maeOriginalProp,autocorr_residOriginal_Prop,rmseOriginalLSTM, maeOriginalLSTM, autocorr_residOriginal_LSTM, rmseTrasf2,maeTrasf2, autocorr_residTrasf2,  rmseTrasf2Prop, maeTrasf2Prop, autocorr_residTrasf2_Prop, rmseTrasf2LSTM, maeTrasf2LSTM,autocorr_residTrasf2_LSTM,i_ciclo)

    WriteTabellaRiassuntiva(workbook,worksheet,i_test,rmseOriginal, maeOriginal, autocorr_residOriginal,rmseOriginalProp, maeOriginalProp,autocorr_residOriginal_Prop,rmseOriginalLSTM, maeOriginalLSTM, autocorr_residOriginal_LSTM, rmseTrasf2,maeTrasf2, autocorr_residTrasf2,  rmseTrasf2Prop, maeTrasf2Prop, autocorr_residTrasf2_Prop, rmseTrasf2LSTM, maeTrasf2LSTM,autocorr_residTrasf2_LSTM,list_diff,list_lamb)

def TestPrediction_AutoArima_Prophet_LSTM_Window90_Pulita(seriesOriginal,seriesTrasf1,seriesTrasf2,particle,lamb,counter_photo,train_set,test_set):
    #questa è la versione per fare i test con la trasformazione del primo 90% usando le finestre (scomponendo le non stazionarietà e trasformandole), ricollegando le trasformate e usando la serie trasformata così fatta per la predizione
    # preparo i train e test sets

    train_dataTrasf1 = seriesTrasf1
    train_dataTrasf2 = seriesTrasf2


    # facciamo la predizione con auto ARIMA della serie originale e calcoliamo rmse
    modelOriginal = pm.auto_arima(train_set, error_action='ignore', trace=True, suppress_warnings=True)
    forecastOriginal = modelOriginal.predict(test_set.shape[0])

    seriesPredictedOriginal = Series(forecastOriginal)
    seriesPredictedOriginal.index = test_set.index

    rO= Compute_Rmse_Mae_AutocorrRes(test_set,seriesPredictedOriginal)
    rmseOriginal = rO[0]
    maeOriginal = rO[1]
    autocorr_residOriginal = rO[2]

    # facciamo la predizione con auto ARIMA della serie trasformata e calcoliamo rmse
    # elimino i primi particle value settati a 0 che mi sballano il training
    #train_dataTrasf2 = train_dataTrasf2.drop(train_dataTrasf2.index[0:particle])

    modelTrasf2 = pm.auto_arima(train_dataTrasf2, error_action='ignore', trace=True, suppress_warnings=True)
    forecastTrasf2 = modelTrasf2.predict(test_set.shape[0])

    seriesPredictedTrasf2 = Series(forecastTrasf2)

    # inverto la predizione
    seriesPredictedInv = InvDiffByParticlePredicted(seriesPredictedTrasf2, train_dataTrasf1, particle)
    seriesPredictedInv = InverseYeojohnson(seriesPredictedInv, seriesPredictedInv, lamb)

    seriesPredictedInv.index = test_set.index

    rt2= Compute_Rmse_Mae_AutocorrRes(test_set,seriesPredictedInv)
    rmseTrasf2 = rt2[0]
    maeTrasf2 = rt2[0]
    autocorr_residTrasf2 = rt2[0]


    # plottiamo le predizioni
    seriesPredictedTrasf2.index = test_set.index


    pyplot.figure()
    pyplot.subplot(211)
    seriesOriginal.plot(color='blue', label='Original')
    pyplot.legend()
    pyplot.title('Transformation Applied  Particle={}  lambda={:.2f}'.format(particle,lamb))
    pyplot.subplot(212)
    seriesTrasf2.plot(color='green', label='Trasformed')
    pyplot.legend()
    pyplot.savefig('D:/Universitaa/TESI/tests/immagini/Syn_' + str(counter_photo) + '.png')
    counter_photo = counter_photo + 1
    pyplot.show()

    pyplot.figure()
    pyplot.subplot(311)
    test_set.plot(color='blue', label='Original')
    seriesPredictedOriginal.plot(color='orange', label='PredictedOriginal')
    pyplot.legend()
    pyplot.title("AutoARIMAOrigin  rmse={:.2f}  mae={:.2f}  autocorrRes={:.2f}".format(rmseOriginal, maeOriginal, autocorr_residOriginal))
    pyplot.subplot(313)
    test_set.plot(color='blue', label='Original')
    seriesPredictedInv.plot(color='red', label='PredictedTrasf')
    pyplot.legend()
    pyplot.title("AutoARIMATrasf rmse={:.2f}   mae={:.2f}  autcorrRes={:.2f}".format(rmseTrasf2,maeTrasf2, autocorr_residTrasf2))

    pyplot.savefig('D:/Universitaa/TESI/tests/immagini/Syn_'+ str(counter_photo) +'.png')
    counter_photo = counter_photo + 1
    pyplot.show()

    # facciamo le predizioni usando prophet
    # facciamo predizione usando serie originale

    result1 = ProphetPredictSeries_Window90(train_set, test_set)
    forecastOriginalProp = result1[0]

    seriesPredictedOriginalProp = Series(forecastOriginalProp)
    seriesPredictedOriginalProp.index = test_set.index

    rOP= Compute_Rmse_Mae_AutocorrRes(test_set,seriesPredictedOriginalProp)
    rmseOriginalProp = rOP[0]
    maeOriginalProp = rOP[1]
    autocorr_residOriginal_Prop = rOP[2]
    # facciamo la predizione usando la serie trasformata
    result = ProphetPredictSeries_Window90(seriesTrasf2, test_set)
    forecastTrasf2Prop = result[0]

    seriesPredictedTrasf2Prop = Series(forecastTrasf2Prop.values)
    seriesTrainTrasf1Val = Series(train_dataTrasf1.values)
    # inverto la predizione
    seriesPredictedInvProp = InvDiffByParticlePredicted(seriesPredictedTrasf2Prop, seriesTrainTrasf1Val,particle)
    seriesPredictedInvProp = InverseYeojohnson(seriesPredictedInvProp, seriesPredictedInvProp, lamb)

    seriesPredictedInvProp.index = test_set.index

    rT2P = Compute_Rmse_Mae_AutocorrRes(test_set, seriesPredictedInvProp)
    rmseTrasf2Prop = rT2P[0]
    maeTrasf2Prop = rT2P[1]
    autocorr_residTrasf2_Prop = rT2P[2]


    # plottiamo le predizioni
    seriesPredictedTrasf2Prop.index = test_set.index


    pyplot.figure()
    pyplot.subplot(311)
    test_set.plot(color='blue', label='Original')
    seriesPredictedOriginalProp.plot(color='orange', label='PredictedOriginal')
    pyplot.legend()
    pyplot.title("ProphetOrigin rmse={:.2f}  mae={:.2f} autocorrRes={:.2f}  ".format(rmseOriginalProp, maeOriginalProp,autocorr_residOriginal_Prop))
    pyplot.subplot(313)
    test_set.plot(color='blue', label='Original')
    seriesPredictedInvProp.plot(color='red', label='PredictedTrasf')
    pyplot.legend()
    pyplot.title("ProphetTrasf rmse={:.2f}  mae={:.2f}, autocorrRes={:.2f}".format(rmseTrasf2Prop,maeTrasf2Prop,autocorr_residTrasf2_Prop))

    pyplot.savefig('D:/Universitaa/TESI/tests/immagini/Syn_'+ str(counter_photo) +'.png')
    counter_photo = counter_photo + 1
    pyplot.show()

    # facciamo la predizione con LSTM
    # iniziamo con la serie originale
    seriesPredictedOriginalLSTM = LSTM_Prediction2_Window90(train_set, test_set)
    #calcoliamo rmse,mae e quantifichiamo l'autocorrelazione del residuo
    rOL= Compute_Rmse_Mae_AutocorrRes(test_set,seriesPredictedOriginalLSTM)
    rmseOriginalLSTM = rOL[0]
    maeOriginalLSTM = rOL[1]
    autocorr_residOriginal_LSTM = rOL[2]

    # facciamo la predizione usando la serie trasformata
    seriesPredictedTrasf2LSTM = LSTM_Prediction2_Window90(seriesTrasf2, test_set)

    # inverto la predizione
    seriesPredictedInvLSTM = InvDiffByParticlePredicted(seriesPredictedTrasf2LSTM, train_dataTrasf1, particle)
    seriesPredictedInvLSTM = InverseYeojohnson(seriesPredictedInvLSTM, seriesPredictedInvLSTM, lamb)
    seriesPredictedInvLSTM.index = test_set.index
    # calcoliamo rmse,mae e quantifichiamo l'autocorrelazione del residuo
    rT2L= Compute_Rmse_Mae_AutocorrRes(test_set,seriesPredictedInvLSTM)
    rmseTrasf2LSTM = rT2L[0]
    maeTrasf2LSTM = rT2L[1]
    autocorr_residTrasf2_LSTM = rT2L[2]

    #plottiamo la predizione di LSTM
    pyplot.figure()
    pyplot.subplot(311)
    test_set.plot(color='blue', label='Original')
    seriesPredictedOriginalLSTM.plot(color='orange', label='PredictedOriginal')
    pyplot.legend()
    pyplot.title("LSTMOrigin rmse={:.2f}  mae={:.2f}  autocorrRes={:.2f}".format(rmseOriginalLSTM, maeOriginalLSTM,autocorr_residOriginal_LSTM))
    pyplot.subplot(313)
    test_set.plot(color='blue', label='Original')
    seriesPredictedInvLSTM.plot(color='red', label='PredictedTrasf')
    pyplot.legend()
    pyplot.title("LSTMTrasf rmse={:.2f} mae={:.2f}   autocorrRes={:.2f}".format(rmseTrasf2LSTM,maeTrasf2LSTM,autocorr_residTrasf2_LSTM))
    pyplot.savefig('D:/Universitaa/TESI/tests/immagini/Syn_'+ str(counter_photo) +'.png')
    counter_photo = counter_photo + 1
    pyplot.show()

#syntetic series generator

def GenerateSynSeries(length,RangeNoise,orderTrend,SinAmpl,fs,period,i_test,i_ciclo):
    #creiamo la sine wave

    linearTrend = list()
    for i in range(0, length):
        data =i *orderTrend
        linearTrend.append(data)
    linearTrend = Series(linearTrend)

    #random.seed()
    x = [random.randint(-RangeNoise, RangeNoise) for i in range(0, length)]
    x = Series(x)
    #fil = open("D:/Universitaa/TESI/tests/immagini/ciclo_esterno" + str(i_ciclo) + "/test_"+str(i_test)+"/info.txt", "a+")
    #rand_values=[(x.head(10).values)]
    #fil.write("First 10 Random values = " + str(x.head(10).values) + "\n" )
    #fil.close()

    if (period!=0):
        f = 1 / period
        t = length
        samples = np.linspace(0, t, int(fs * t), endpoint=False)
        signal = np.sin(2 * np.pi * f * samples) * SinAmpl

        sineWave = Series(signal)

        synSeries1 = list()
        for i in range(0, length):
            #data = x[i] + sineWave[i] + linearTrend[i]
            data= sineWave[i] + linearTrend[i]
            synSeries1.append(data)

        synSeries1 = Series(synSeries1)


    else:
        synSeries1 = list()
        for i in range(0, length):
            #data = x[i]  + linearTrend[i]
            data = linearTrend[i]
            synSeries1.append(data)

        synSeries1 = Series(synSeries1)

    return synSeries1

def GenerateSynSeriesNonRandom(length,base,orderTrend,SinAmpl,fs,period,i_test,i_ciclo):
    #creiamo la sine wave

    linearTrend = list()
    for i in range(0, length):
        data =i *orderTrend
        linearTrend.append(data)
    linearTrend = Series(linearTrend)



    if (period!=0):
        f = 1 / period
        t = length
        samples = np.linspace(0, t, int(fs * t), endpoint=False)
        signal = np.sin(2 * np.pi * f * samples) * SinAmpl
        sineWave = Series(signal)





        synSeries1 = list()
        for i in range(0, length):
            data = sineWave[i] + linearTrend[i] + base[i] +200
            synSeries1.append(data)

        synSeries1 = Series(synSeries1)


    else:
        synSeries1 = list()
        for i in range(0, length):
            data =linearTrend[i] + base[i] + 100
            synSeries1.append(data)

        synSeries1 = Series(synSeries1)

    return synSeries1

def GenerateBase(length):
    # creiamo la base della serie che porta informazioni
    # tale base  sarà composta da del rumore
    # random.seed()
    x = [random.randint(-50,50) for i in range(0, length)]
    x = Series(x)
    # fil = open("D:/Universitaa/TESI/tests/immagini/ciclo_esterno" + str(i_ciclo) + "/test_"+str(i_test)+"/info.txt", "a+")
    # rand_values=[(x.head(10).values)]
    # fil.write("First 10 Random values = " + str(x.head(10).values) + "\n" )
    # fil.close()

    fs=1
    f2 = 1 / (1500)
    t = length
    samples = np.linspace(0, t, int(fs * t), endpoint=False)
    signal2 = np.sin(2 * np.pi * f2 * samples) * (30)
    sineWave2 = Series(signal2)

    base = list()
    for i in range(0, length):
        data = sineWave2[i] + x[i]
        base.append(data)
    base = Series(base)

    plt.title("Base della serie con le informazioni")
    base.plot(color='green')

    # divido la base in test e train giusto per plottarla di colori diversi
    train_lenght = int(length * 0.9)
    base_test = base[train_lenght:]
    base_test.plot(color='red')
    plt.show()

    return [base,base_test]

def GenerateSynSeriesCorrelation(length,base,orderTrend,SinAmpl,fs,period,i_test,i_ciclo):
    #creiamo la sine wave

    linearTrend = list()
    for i in range(0, length):
        data =i *orderTrend
        linearTrend.append(data)
    linearTrend = Series(linearTrend)

    if (period!=0):
        f = 1 / period
        t = length
        samples = np.linspace(0, t, int(fs * t), endpoint=False)
        signal = np.sin(2 * np.pi * f * samples) * SinAmpl

        sineWave = Series(signal)

        synSeries1 = list()
        for i in range(0, length):
            data = base[i] + sineWave[i] + linearTrend[i]
            synSeries1.append(data)

        synSeries1 = Series(synSeries1)




    else:
        synSeries1 = list()
        for i in range(0, length):
            data = base[i]  + linearTrend[i]
            synSeries1.append(data)

        synSeries1 = Series(synSeries1)


    dti1 = pd.date_range("2018-01-01", periods=len(synSeries1), freq="D")
    synSeries1.index= dti1

    return synSeries1

def GenerateArmaSignal(length):
    # Set coefficients
    ar_coefs = [0.5, 0.5, 0.2]
    ma_coefs = [0.5, -0.3, 0.2]


    # Generate data
    y = arma_generate_sample(ar_coefs, ma_coefs, nsample=length, scale=1)
    #y = y + 200

    dti = pd.date_range("2018-01-01", periods=length, freq="D")
    s = pd.Series(y, index=dti, name="Value")
    s = s.apply(lambda x: round(x, 3))

    s.plot(color='orange')
    plt.show()

    return s

def AddArmaSignal_toSeries(armaSignal,series):
    s = list()
    for i in range(len(series)):
        data=series[i]+armaSignal[i]
        s.append(data)
    s=Series(s)
    s.index=series.index

    return s

def concatSeries(series1,series2):

    concatenatedSeries=list()

    for i in range(0,len(series1)):
        data=series1[i]
        concatenatedSeries.append(data)

    lastValue=series1[len(series1)-1]
    avgValue= mean(series1[len(series1)-10:len(series1)])


    for i in range(0,len(series2)):
        data=series2[i]+avgValue
        concatenatedSeries.append(data)

    concatenatedSeries=Series(concatenatedSeries)
    return concatenatedSeries

def concatSeries2(series1,series2):

    concatenatedSeries=list()

    for i in range(0,len(series1)):
        data=series1[i]
        concatenatedSeries.append(data)

    lastValue=series1[len(series1)-1]
    avgValue= mean(series1)


    for i in range(0,len(series2)):
        data=series2[i]
        concatenatedSeries.append(data)

    concatenatedSeries=Series(concatenatedSeries)
    return concatenatedSeries






def alarm():
    duration = 1500  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)


def RoundLambda(lamb):
    if(lamb>=0 and lamb <=0.25):
        lamb=0
    if(lamb>=0.25 and lamb <= 0.75):
        lamb=0.5
    if(lamb>=0.75 and lamb <=1):
        lamb=1
    return lamb






def Compute_Rmse_Mae_AutocorrRes(test_set,seriesPredicted):
        rmse = sqrt(mean_squared_error(test_set, seriesPredicted))
        mae = mean_absolute_error(test_set, seriesPredicted)
        autocorr_resid = Quantif_Autocorr_Residual(seriesPredicted, test_set)
        result = [rmse,mae,autocorr_resid]
        return result





def Create_Excel_File(i_test,rmseOriginalArima,maeOriginalArima,autocorrOriginalArima,rmseOriginalProphet,maeOriginalProphet,autocorrOriginalProphet,rmseOriginalLSTM,maeOriginalLSTM,autocorrOriginalLSTM,rmseTrasf2Arima,maeTrasf2Arima,autocorrTrasf2Arima,rmseTrasf2Prophet,maeTrasf2Prophet,autocorrTrasf2Prophet,rmseTrasf2LSTM,maeTrasf2LSTM,autocorrTrasf2LSTM,i_ciclo):
    workbook = xlsxwriter.Workbook('D:/Universitaa/TESI/tests/immagini/ciclo_esterno' + str(i_ciclo) + '/test_' + str(i_test) + '/risultati'+str(i_test)+'.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write('B2', 'Senza Trasformazioni')
    worksheet.write('K2', 'Con Trasformazioni')
    worksheet.write('B3', 'AutoArima')
    worksheet.write('E3', 'Prophet')
    worksheet.write('H3', 'LSTM')
    worksheet.write('K3', 'AutoArima')
    worksheet.write('N3', 'Prophet')
    worksheet.write('Q3', 'LSTM')

    worksheet.write('B4', 'rmse')
    worksheet.write('C4', 'mae')
    worksheet.write('D4', 'autocorR')
    worksheet.write('E4', 'rmse')
    worksheet.write('F4', 'mae')
    worksheet.write('G4', 'autocorR')
    worksheet.write('H4', 'rmse')
    worksheet.write('I4', 'mae')
    worksheet.write('J4', 'autocoR')

    worksheet.write('K4', 'rmse')
    worksheet.write('L4', 'mae')
    worksheet.write('M4', 'autocorR')
    worksheet.write('N4', 'rmse')
    worksheet.write('O4', 'mae')
    worksheet.write('P4', 'autocorR')
    worksheet.write('Q4', 'rmse')
    worksheet.write('R4', 'mae')
    worksheet.write('S4', 'autocoR')

    #Scrivo i valori rmse,mae,autocorr
    worksheet.write('B5', round(rmseOriginalArima,2))
    worksheet.write('C5', round(maeOriginalArima,2))
    worksheet.write('D5', round(autocorrOriginalArima,2))
    worksheet.write('E5', round(rmseOriginalProphet, 2))
    worksheet.write('F5', round(maeOriginalProphet, 2))
    worksheet.write('G5', round(autocorrOriginalProphet, 2))
    worksheet.write('H5', round(rmseOriginalLSTM, 2))
    worksheet.write('I5', round(maeOriginalLSTM, 2))
    worksheet.write('J5', round(autocorrOriginalLSTM, 2))
    worksheet.write('K5', round(rmseTrasf2Arima,2))
    worksheet.write('L5', round(maeTrasf2Arima,2))
    worksheet.write('M5', round(autocorrTrasf2Arima,2))
    worksheet.write('N5', round(rmseTrasf2Prophet,2))
    worksheet.write('O5', round(maeTrasf2Prophet,2))
    worksheet.write('P5', round(autocorrTrasf2Prophet,2))
    worksheet.write('Q5', round(rmseTrasf2LSTM,2))
    worksheet.write('R5', round(maeTrasf2LSTM,2))
    worksheet.write('S5', round(autocorrTrasf2LSTM,2))

    workbook.close()






    '''
    diff_senza_yeo = [6, 5, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 5, 6, 6, 6, 6, 1, 17, 18, 18, 18, 18, 17, 13, 13, 13, 13,
                      12]

    diff_con_yeo = [12, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 17, 18, 18, 19, 18, 17, 13, 13, 13, 13,
                    12]
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(diff_senza_yeo, color='red', label='diff scelta con lambda da Yeojonson')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(diff_con_yeo, color='blue', label='diff scelta con lambda da PSO')
    plt.legend()
    plt.savefig('D:/Universitaa/TESI/tests/immagini/test_'+str(i)+'/my_plot')
    plt.show()
    '''





def FillStockSeries(series):
    series = TimeSeries.from_series(series, freq='D')
    series = series.pd_series()
    series = series.interpolate()
    return series

def NormalizeScore(score,seriesTrasf):
    max= seriesTrasf.max()
    min= seriesTrasf.min()

    score = (score - min) / (max - min)
    return score

def Create_Nan_Series(series):
    #crea una serie di NaN seguendo la serie che gli viene data in input
    seriesNan=list()

    for i in range(0, len(series)):
        seriesNan.append(np.nan)

    seriesNan = Series(seriesNan)
    seriesNan.index = series.index

    return seriesNan



def SeparateNonStat_Stationarize_ConcatenateTrasf(series, counter_photo, period1, period2, period3,period4, i_test,i_ciclo):
    #questa funzione prende in input una serie con diverse non stazionarietà
    #scompone la serie seguendo le diverse non stazioanrietà
    #rende stazionari i vari segmenti estratti trasformandoli
    #ricongiunge i segmenti trasformati

    #restituisce in output la serie trasformata 1 e 2  ricongiunta e l'ultima lamb e diff applicata

    # costruiamo le nan series per ricongiungere le serie estratte trasformate
    seriesRecostructed = Create_Nan_Series(series)
    seriesTrasf1Recostructed = Create_Nan_Series(series)
    seriesTrasf2Recostructed = Create_Nan_Series(series)

    # vado a estrarre le non stazionarità delle serie
    seriesExtracted = Stationarize_PSO_Window4(series, counter_photo, period1, period2, period3,period4, i_test,i_ciclo)


    # vado a trasformare una alla volta le non stazionarietà

    for ser in seriesExtracted:
        result = StationarizeWithPSO_Original(ser, i_test, i_ciclo)
        seriesTrasf1 = result[1]
        seriesTrasf2 = result[0]
        particle = result[2]
        lamb = result[3]

        # normalizziamo  serTrasf2
        seriesTrasf2, scaler2 = Normalize_Series(seriesTrasf2)

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

    result = [seriesTrasf2Recostructed, seriesTrasf1Recostructed, particle, lamb]

    return result

def CreateTabellaRiassuntiva(i_ciclo,testo):
    os.makedirs(r'D:/Universitaa/TESI/tests/immagini/ciclo_esterno' + str(i_ciclo))
    workbook = xlsxwriter.Workbook(
        'D:/Universitaa/TESI/tests/immagini/ciclo_esterno' + str(i_ciclo) + '/tabella_riassuntiva.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', str(testo))
    worksheet.write('C1', 'AARmse')
    worksheet.write('D1', 'AAMae')
    worksheet.write('E1', 'AAAutocorr')
    worksheet.write('F1', 'PRmse')
    worksheet.write('G1', 'PMae')
    worksheet.write('H1', 'PAutocorr')
    worksheet.write('I1', 'LRmse')
    worksheet.write('J1', 'LMae')
    worksheet.write('K1', 'Lauto')
    worksheet.write('L1', 'TAARmse')
    worksheet.write('M1', 'TAAMae')
    worksheet.write('N1', 'TAAAutocor')
    worksheet.write('O1', 'TPRmse')
    worksheet.write('P1', 'TPMae')
    worksheet.write('Q1', 'TPAutocorr')
    worksheet.write('R1', 'TLRmse')
    worksheet.write('S1', 'TLMae')
    worksheet.write('T1', 'TLAutocorr')
    worksheet.write('U1', 'Diff1')
    worksheet.write('V1', 'YJ1')
    worksheet.write('W1', 'Diff2')
    worksheet.write('X1', 'YJ2')
    worksheet.write('Y1', 'Diff3')
    worksheet.write('Z1', 'YJ3')
    worksheet.write('AA1', 'Diff4')
    worksheet.write('AB1', 'YJ4')


    worksheet.write('B2', 'Media')
    worksheet.write('C2', '=MEDIA(C4:C14)')
    worksheet.write('D2', '=MEDIA(D4:D14)')
    worksheet.write('E2', '=MEDIA(E4:E14)')
    worksheet.write('F2', '=MEDIA(F4:F14)')
    worksheet.write('G2', '=MEDIA(G4:G14)')
    worksheet.write('H2', '=MEDIA(H4:H14)')
    worksheet.write('I2', '=MEDIA(I4:I14)')
    worksheet.write('J2', '=MEDIA(J4:J14)')
    worksheet.write('K2', '=MEDIA(K4:K14)')
    worksheet.write('L2', '=MEDIA(L4:L14)')
    worksheet.write('M2', '=MEDIA(M4:M14)')
    worksheet.write('N2', '=MEDIA(N4:N14)')
    worksheet.write('O2', '=MEDIA(O4:O14)')
    worksheet.write('P2', '=MEDIA(P4:P14)')
    worksheet.write('Q2', '=MEDIA(Q4:Q14)')
    worksheet.write('R2', '=MEDIA(R4:R14)')
    worksheet.write('S2', '=MEDIA(S4:S14)')
    worksheet.write('T2', '=MEDIA(T4:T14)')
    worksheet.write('U2', '=MEDIA(U4:U14)')
    worksheet.write('V2', '=MEDIA(V4:V14)')
    worksheet.write('W2', '=MEDIA(W4:W14)')
    worksheet.write('X2', '=MEDIA(X4:X14)')
    worksheet.write('Y2', '=MEDIA(Y4:Y14)')
    worksheet.write('Z2', '=MEDIA(Z4:Z14)')
    worksheet.write('AA2','=MEDIA(AA4:AA14)')
    worksheet.write('AB2', '=MEDIA(AB4:AB14)')


    worksheet.write('B3', 'Dev std')
    worksheet.write('C3', '=DEV.ST.C(C4:C14)')
    worksheet.write('D3', '=DEV.ST.C(D4:D14)')
    worksheet.write('E3', '=DEV.ST.C(E4:E14)')
    worksheet.write('F3', '=DEV.ST.C(F4:F14)')
    worksheet.write('G3', '=DEV.ST.C(G4:G14)')
    worksheet.write('H3', '=DEV.ST.C(H4:H14)')
    worksheet.write('I3', '=DEV.ST.C(I4:I14)')
    worksheet.write('J3', '=DEV.ST.C(J4:J14)')
    worksheet.write('K3', '=DEV.ST.C(K4:K14)')
    worksheet.write('L3', '=DEV.ST.C(L4:L14)')
    worksheet.write('M3', '=DEV.ST.C(M4:M14)')
    worksheet.write('N3', '=DEV.ST.C(N4:N14)')
    worksheet.write('O3', '=DEV.ST.C(O4:O14)')
    worksheet.write('P3', '=DEV.ST.C(P4:P14)')
    worksheet.write('Q3', '=DEV.ST.C(Q4:Q14)')
    worksheet.write('R3', '=DEV.ST.C(R4:R14)')
    worksheet.write('S3', '=DEV.ST.C(S4:S14)')
    worksheet.write('T3', '=DEV.ST.C(T4:T14)')
    worksheet.write('U3', '=DEV.ST.C(U4:U14)')
    worksheet.write('V3', '=DEV.ST.C(V4:V14)')
    worksheet.write('W3', '=DEV.ST.C(W4:W14)')
    worksheet.write('X3', '=DEV.ST.C(X4:X14)')
    worksheet.write('Y3', '=DEV.ST.C(Y4:Y14)')
    worksheet.write('Z3', '=DEV.ST.C(Z4:Z14)')
    worksheet.write('AA3','=DEV.ST.C(AA4:AA14)')
    worksheet.write('AB3', '=DEV.ST.C(AB4:AB14)')

    return [workbook,worksheet]

def WriteTabellaRiassuntiva(workbook,worksheet,i_test,rmseOriginal, maeOriginal, autocorr_residOriginal,rmseOriginalProp, maeOriginalProp,autocorr_residOriginal_Prop,rmseOriginalLSTM, maeOriginalLSTM, autocorr_residOriginal_LSTM, rmseTrasf2,maeTrasf2, autocorr_residTrasf2,  rmseTrasf2Prop, maeTrasf2Prop, autocorr_residTrasf2_Prop, rmseTrasf2LSTM, maeTrasf2LSTM,autocorr_residTrasf2_LSTM,list_diff,list_lamb):
    # scriviamo su file excel i risultati
    cell_index = 4 + i_test
    worksheet.write('B' + str(cell_index), 'test' + str(i_test))
    worksheet.write('C' + str(cell_index), rmseOriginal)
    worksheet.write('D' + str(cell_index), maeOriginal)
    worksheet.write('E' + str(cell_index), autocorr_residOriginal)
    worksheet.write('F' + str(cell_index), rmseOriginalProp)
    worksheet.write('G' + str(cell_index), maeOriginalProp)
    worksheet.write('H' + str(cell_index), autocorr_residOriginal_Prop)
    worksheet.write('I' + str(cell_index), rmseOriginalLSTM)
    worksheet.write('J' + str(cell_index), maeOriginalLSTM)
    worksheet.write('K' + str(cell_index), autocorr_residOriginal_LSTM)
    worksheet.write('L' + str(cell_index), rmseTrasf2)
    worksheet.write('M' + str(cell_index), maeTrasf2)
    worksheet.write('N' + str(cell_index), autocorr_residTrasf2)
    worksheet.write('O' + str(cell_index), rmseTrasf2Prop)
    worksheet.write('P' + str(cell_index), maeTrasf2Prop)
    worksheet.write('Q' + str(cell_index), autocorr_residTrasf2_Prop)
    worksheet.write('R' + str(cell_index), rmseTrasf2LSTM)
    worksheet.write('S' + str(cell_index), maeTrasf2LSTM)
    worksheet.write('T' + str(cell_index), autocorr_residTrasf2_LSTM)

    if(len(list_diff)>0):
        worksheet.write('U' + str(cell_index), list_diff[0])
        worksheet.write('V' + str(cell_index), list_lamb[0])

    if(len(list_diff)>1):
        worksheet.write('W' + str(cell_index), list_diff[1])
        worksheet.write('X' + str(cell_index), list_lamb[1])

    if (len(list_diff) > 2):
        worksheet.write('Y' + str(cell_index), list_diff[2])
        worksheet.write('Z' + str(cell_index), list_lamb[2])

    if (len(list_diff) > 3):
        worksheet.write('AA' + str(cell_index), list_diff[3])
        worksheet.write('AB' + str(cell_index), list_lamb[3])



#stationarization

def StationarityFunction(params,ser):
    series=ser
    seriesOriginal = series

    p0 = params[0]
    p1 = params[1]

    # le due istruzioni commentate sotto servono per disattivare le trasformazioni
    # p0=1
    # p1=0
    seriesTrasf1 = YeojohnsonTrasform(series, p0)
    seriesTrasf2 = DifferencingByParticleValue(seriesTrasf1, round(p1))

    return (TrendStationarityScore(seriesOriginal, seriesTrasf2) + SeasonStationarityScore(seriesTrasf2) + AR1Score(seriesOriginal, seriesTrasf1, seriesTrasf2, round(p1), p0))

def f(x,ser):
    n_particles=x.shape[0]

    j=[StationarityFunction(x[i],ser) for i in range(n_particles)]
    return np.array(j)

def ATSS_Apply_Transformation(series,p0,p1):
    "ripete le trasformazioni fatte dallo swarm una volta che ha trovato i parametri migliori"
    "questo è per ottenere la serie trasformata"

    seriesTrasf1 = YeojohnsonTrasform(series, p0)
    seriesTrasf2 = DifferencingByParticleValue(seriesTrasf1,round(p1))

    seriesTrasf1.index=series.index
    seriesTrasf2.index=series.index

    return [seriesTrasf2,seriesTrasf1,p0,p1]

def ATSS_Extract_Subseries(series):
    #la funzione prende in input una serie contenente diverse non stazionarietà
    #restituisce in output la serie scomposta in sottoserie in base alla non stazionarietà

    # la funzione crea delle finestre per studiare come varia la non stazionarietà della seire, andando ad analizzare come varia la trasformazione applicata dalla PSO nel tempo
    # per rendere le finestre piu generali possibili, ho scelto una grandezza di 5*maxAutocorrelationLag, in modo da essere sicuri di catturare eventuali periodicità
    # la dimensione della window cambia nel tempo, quando vengono individuati cambiamnti significativi della diff (e.g non multipli e non valori vicini)
    # e quando viene identificato anche un cambiamento  significaivo dell' maxAutocorrelation lag
    # a quel punto la serie analizzata fino a quel momento viene droppata, viene ricalcolato il maxAutocorrelationLag sulla serie rimanente e viene ricalcolata la window

    max_autocorrelation_lag = FindAutocorrelationMaxLag(series)
    #nel caso in cui non riesce a trovare un max_autocorr_lag all'inizio, a causa delle troppe non stazionarietà che confondono l'autocorrelazione
    #inizializzo max_auto_lag a 30, in modo da avere una generica finestra di 150, che poi si adatterà successivamente da sola
    if(max_autocorrelation_lag==0):
        max_autocorrelation_lag=30


    list_par = list()
    list_lamb = list()
    list_score = list()
    list_window = list()
    list_series_extracted=list()
    list_autocorrelation_lags=list()

    i = 0  # mi fa muovere lungo la serie
    wind = 5 * max_autocorrelation_lag  # è l'ampiezza della finestra
    x = 0  # l'inizio della finestra
    y = wind  # la fine della finestra
    Count = 0  # mi serve come condizione per analizzare alla fine la serie completa
    lastLap = False  #serve per fare l'ultima analisi con windows=len(series)
    change = False # mi serve per fare la correzione dei lag nell'iterazione in cui c'è cambio di window
    change_station = False  # indica se c'è stato un cambio di stationarietà, serve per estrarre l'ultimo pezzo della serie con non-stazionarietà diversa
    oldCheckPoint = 0  # inizio di una porzione di serie con una certa non stazionarietà
    num_nonStat_find = 1  #serve per tenere traccia di quanti segmenti di serie con diverse non-stazionarietà sono contenuti nella serie
    counter_stationarity = 0 #mi serve per contare quante volte capita che la PSO applica diff=0 e lamb=1.0, cioè non applica trasformazioni , perchè se succede spesso allora non taglio la serie ma la considero nella sua interezza
    while (Count < 2):
        if (Count == 1):
            Count = 2
        batch = series.iloc[x:y]

        try:
            result = ATSS_Stationarize_Window(batch)

        except:
            #se la trasformazione applicata da una finestra va in errore, significa che la finestra non riesce a catturare l'eventuale non stazionaerietà
            #per questo motivo restituisco direttamente la serie nella sua interezza
            seriesExtracted=list()
            seriesExtracted.append(series)
            return seriesExtracted

        #se la PSO restituisce diff=0 e lamb=1 significa che la porzione di window è stazionaria, o ci sono delle stazionarietà all'interno della serie che la window non riesce a vedere
        #per questo dopo 3 volte che la finestra restituisce una prozione stazionaria
        #restituisco semplicemente la serie nella sua totalità senza scomporla, per analizzarla nella sua interezza

        if(result[2]==0 and result[3]==1.0):
            counter_stationarity = counter_stationarity+1
            if(counter_stationarity == 3):
                seriesExtracted = list()
                seriesExtracted.append(series)
                return seriesExtracted

        #calcolo il maxAutocorrelation lag all'interno della finestra, per capire che periodicità c'è nella finestra
        lagBatch = FindAutocorrelationMaxLag2(batch)


        list_par.append(result[2])
        list_lamb.append(round(result[3], 2))
        list_score.append(round(result[4], 2))
        list_window.append((x, y, wind))

        #questo if mi serve per aggiornare il max_autocorr_lag solo nel caso in cui il max_lag visto nella finestra è cambiato in modo significativo
        if(lagBatch!=0 and (lagBatch<max_autocorrelation_lag-2 or lagBatch>max_autocorrelation_lag+2)):
           max_autocorrelation_lag=lagBatch
           print('AAAAAA')


        #questo if mi serve per aggiornare il max_autocorr_lag solo nel caso in cui ci sono stati cambiamenti significativi della diff
        if ((list_par[i] > list_par[i - 1] + 3 or list_par[i] < list_par[i - 1] - 3) ):
            max_autocorrelation_lag=lagBatch
            print('BBBBBB')

        list_autocorrelation_lags.append(max_autocorrelation_lag)

        #questo if serve ad accorgersi del cambio di non stazionarietà, andando a confrontare gli ultimi 2 valori di max_autocorr_lag registrati
        #se i due valori si discostano in modo significativo, allora la non stazionarietà potrebbe essere cambiata
        if ((list_autocorrelation_lags[i] > list_autocorrelation_lags[i - 1] + 2 or list_autocorrelation_lags[i] < list_autocorrelation_lags[i - 1] - 2) and lastLap==False):
            #quindi vado a ricalcolare il max_autocorrelation_lag con ciò che rimane della serie, droppando la parte analizzata fin ora

            # rimuovo la serie analizzata fin ora
            seriesHalf = series.drop(series.index[0:y])
            # ricalcolo il maxAutocorrelationLag con la serie rimanente
            New_max_autocorrelation_lag = FindAutocorrelationMaxLag2(seriesHalf)

            max_autocorrelation_lag = New_max_autocorrelation_lag

            change=True
            change_station = True
            num_nonStat_find=num_nonStat_find+1


            # estraggo la porzione di serie vista fino ad ora, che avrà una sua non stazionarietà, diversa dalle altre porzioni di serie
            newCheckPoint = x
            seriesExtracted = series[oldCheckPoint:newCheckPoint]
            list_series_extracted.append(seriesExtracted)
            oldCheckPoint = y

        # una volta ricalcolato il max_autocorrelation lag, ricalcolo la dimensione della window

        wind = 5 * max_autocorrelation_lag
        x = y
        y = min(len(series), y+wind )

        #questo if serve per aggiornare la finestra a seguito di un cambio di non-stazionarietà
        if(change==True):
            batch = series.iloc[x:y]
            lagBatch = FindAutocorrelationMaxLag2(batch)
            if (lagBatch != 0):
                max_autocorrelation_lag = lagBatch
                list_autocorrelation_lags[i] = max_autocorrelation_lag
            change=False

        # se la window arriva all'ultimo valore della serie
        # fa un'ultima analisi con una window pari alla dimensione della serie
        # così da fare un'analisi della serie nella sua interezza
        if (y == len(series) and Count == 0):
            Count = 1
            x = 0
            wind = len(series)
            lastLap=True

            if (change_station == True):
                seriesExtracted = series[oldCheckPoint:y]
                list_series_extracted.append(seriesExtracted)
            else:
                list_series_extracted.append(series)
        i = i + 1

    print(list_autocorrelation_lags)
    return list_series_extracted

def ATSS_Stationarize_Series(series):
    #è la funzione stationarize originale
    #è la PSO che viene usata sui pezzi di serie estratti

    #qui viene inizializzato l'intero swarm  per la diff al picco massimo di autocorrelazione della serie
    #la yeojohnson viene inizializzata ad 1

    # operazioni per PSO
    lenSeriesReduced = round(len(series) / 1)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    min_bound = np.array([0, 0])
    max_bound = np.array([1.0, lenSeriesReduced])
    bounds = (min_bound, max_bound)
    dimensions = 2

    # operazioni per inizializzare swarm
    swarm_size = 10
    num_iters = 10

    init_lag = FindAutocorrelationMaxLag(series)
    init_value = [1.0,init_lag]  # metto 1.0 come inizializzazione della yeo, perchè voglio partire dal caso piu semplice, cioè non applica la yeo ma applica la diff di lag=maxAutocorrelationLag

    initialize = np.array([init_value for i in range(swarm_size)])

    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dimensions, bounds=bounds, options=options, init_pos=initialize)
    cost, pos = optimizer.optimize(f, iters=num_iters,ser=series)

    # applico la best trasformation torvata dalla PSO
    result=ATSS_Apply_Transformation(series,pos[0],pos[1])
    seriesTrasf2 = result[0]
    seriesTrasf1 = result[1]
    pos[0] = result[2]
    pos[1] = result[3]


    return [seriesTrasf2, seriesTrasf1, round(pos[1]), pos[0],cost]

def ATSS_Stationarize_Window(series):
    #questa è la PSO che viene usata dalle finestre
    #controlliamo se la prozione di serie vista dalla finestra è stazionaria o meno
    #perchè se è stazionaria allora significa che la finestra non riesce a catturare la non stazionarietà, quindi è meglio valutare la serie nella sua interezza
    #quindi non applico la trasfomrazione ma restituisco semplciemente diff=0 e YJ=1
    if (CheckStationarity(series) == True):
        return [series, series, 0, 1.0, 0]

    else:
        # qui invece lo swarm viene inizializzato con le prime 3  particelle pari al max_autocorr_lag, mentre le altre vengono inizializzate ad altri picchi di autocorrelazione
        # questo per fare "spaziare" di più la PSO
        # la yeojohnson viene sempre inizializzata ad 1
        seriesOriginal = series

        # operazioni per PSO
        lenSeriesReduced = round(len(series) / 1)
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        min_bound = np.array([0, 0])
        max_bound = np.array([1.0, lenSeriesReduced])
        bounds = (min_bound, max_bound)
        dimensions = 2

        # operazioni per inizializzare swarm
        swarm_size = 10
        num_iters = 10
        init_lag = FindAutocorrelationMaxLag(series)
        auto_lags = GetAutocorrelationLags(series)

        # uso questa seconda inizializzazione, per prendere i primi "swarm" valori dei lag di autocorrelation e non solo il Max
        if (len(auto_lags) == 0):
            auto_lags = [0, 1, 7]
        initialize2 = list()
        k = 0
        initialize2.append([1.0, init_lag])
        initialize2.append([1.0, init_lag])
        for i in range(swarm_size - 2):
            count = len(auto_lags)

            yeo = 1.0
            lag = auto_lags[k]

            k = k + 1
            if (k == count):
                k = 0

            initialize2.append([yeo, lag])

        initialize2 = np.array(initialize2)

        optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dimensions, bounds=bounds,
                                            options=options, init_pos=initialize2)
        cost, pos = optimizer.optimize(f, iters=num_iters, ser=seriesOriginal)

        # applico la best trasformation scelta da PSO
        result = ATSS_Apply_Transformation(series, pos[0], pos[1])

        seriesTrasf2 = result[0]
        seriesTrasf1 = result[1]
        pos[0] = result[2]
        pos[1] = result[3]

        return [seriesTrasf2, seriesTrasf1, round(pos[1]), pos[0], cost]




# FUNZIONI PER FARE TEST

def ATSS(series):
    #questo è l'algoritmo finale
    #questa funzione prende in input una serie con diverse non stazionarietà
    #scompone la serie seguendo le diverse eventuali non stazioanrietà
    #rende stazionari i vari segmenti estratti trasformandoli applicando la best diff e YJ scelta dalla PSO
    #normalizza i segmenti trasformati
    #ricongiunge i segmenti normalizzati

    #restituisce in output la serie trasformata 1 e 2 e le lamb e diff applicate per i vari segmenti

    #controlliamo prima di tutto se la serie è stazionaria o meno
    #se è stazionaria, non facciamo nessuna trasformazione e semplicemente restutiamo diff=0 e YJ=1.0
    if (CheckStationarity(series) == True):
        return [series, series, [0], [1.0]]

    #se è non stazionaria, invece, cerchiamo la trasformazione migliore
    else:
                # costruiamo le nan series per ricongiungere le serie estratte trasformate
        seriesTrasf1Recostructed = Create_Nan_Series(series)
        seriesTrasf2Recostructed = Create_Nan_Series(series)

        # vado a estrarre le non stazionarità delle serie
        seriesExtracted = ATSS_Extract_Subseries(series)

        list_diff = list()
        list_lamb = list()
        # vado a trasformare una alla volta le non stazionarietà
        for ser in seriesExtracted:
            seriesTrasf2, seriesTrasf1, diff, lamb, cost = ATSS_Stationarize_Series(ser)
            list_diff.append(diff)
            list_lamb.append(lamb)

            # normalizziamo  serTrasf2
            seriesTrasf2, scaler2 = Normalize_Series(seriesTrasf2)

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

        return [seriesTrasf2Recostructed, seriesTrasf1Recostructed, list_diff, list_lamb,scaler2]

def ATSS_Invert_Transformation(seriesTrasf2,seriesTrasf1,diff,lamb,scaler):
    #inverto le trasformazioni seguendo l'ordine inverso di applicazione
    seriesInverted = Invert_Normalize_Series(seriesTrasf2,scaler)
    seriesInverted = InvertDiffByParticleValue(seriesTrasf1,seriesInverted,diff)
    seriesInverted = InverseYeojohnson(seriesInverted,lamb)

    seriesInverted.index = seriesTrasf2[diff:].index

    return seriesInverted

def ATSS_Invert_Prediction(seriesPredicted, seriesTrasf1, diff, lamb, scaler):
    # applico le trasformazioni in ordine inverso alla predizione
    seriesPredictedInverted = Invert_Normalize_Series(seriesPredicted, scaler)
    seriesPredictedInverted = InvDiffByParticlePredicted(seriesPredictedInverted, seriesTrasf1, diff)
    seriesPredictedInverted = InverseYeojohnson(seriesPredictedInverted, lamb)

    # la predizione invertita che viene ritornata non ha l'index originale
    # l'index deve essere copiato dal test set
    return seriesPredictedInverted

def ATSS_Change_Point(series):

    #questa è la versione per 2ndTestWindow (90)

    #la funzione prende in input una serie contenente diverse non stazionarietà
    #restituisce in output la serie scomposta in sottoserie in base alla non stazionarietà

    # la funzione crea delle finestre per studiare come varia la non stazionarietà della seire, andando ad analizzare come varia la trasformazione applicata dalla PSO nel tempo
    # per rendere le finestre piu generali possibili, ho scelto una grandezza di 5*maxAutocorrelationLag, in modo da essere sicuri di catturare eventuali periodicità
    # la dimensione della window cambia nel tempo, quando vengono individuati cambiamnti significativi della diff (e.g non multipli e non valori vicini)
    # e quando viene identificato anche un cambiamento  significaivo dell' maxAutocorrelation lag
    # a quel punto la serie analizzata fino a quel momento viene droppata, viene ricalcolato il maxAutocorrelationLag sulla serie rimanente e viene ricalcolata la window

    max_autocorrelation_lag = FindAutocorrelationMaxLag(series)
    #nel caso in cui non riesce a trovare un max_autocorr_lag all'inizio, a causa delle troppe non stazionarietà che confondono l'autocorrelazione
    #inizializzo max_auto_lag a 30, in modo da avere una generica finestra di 150, che poi si adatterà successivamente da sola
    if(max_autocorrelation_lag==0):
        max_autocorrelation_lag=30

    list_par = list()
    list_lamb = list()
    list_score = list()
    list_window = list()
    list_series_extracted=list()
    list_autocorrelation_lags=list()

    i = 0  # mi fa muovere lungo la serie
    wind = 5 * max_autocorrelation_lag  # è l'ampiezza della finestra
    x = 0  # l'inizio della finestra
    y = wind  # la fine della finestra
    Count = 0  # mi serve come condizione per analizzare alla fine la serie completa
    lastLap = False  #serve per fare l'ultima analisi con windows=len(series)
    change = False # mi serve per fare la correzione dei lag nell'iterazione in cui c'è cambio di window
    change_station = False  # indica se c'è stato un cambio di stationarietà, serve per estrarre l'ultimo pezzo della serie con non-stazionarietà diversa
    oldCheckPoint = 0  # inizio di una porzione di serie con una certa non stazionarietà
    num_nonStat_find = 1  #serve per tenere traccia di quanti "pezzi di non stazionarietà" sono contenuti nella serie
    counter_stationarity = 0 #mi serve per contare quante volte capita che la PSO applica diff=0 e lamb=1.0, cioè non applica trasformazioni , perchè se succede spesso allora non taglio la serie ma la considero nella sua interezza
    list_window_change_index = list()
    while (Count < 2):
        if (Count == 1):
            Count = 2
        batch = series.iloc[x:y]

        try:
            result = ATSS_Stationarize_Window(batch)

        except:

            seriesExtracted=list()
            seriesExtracted.append(series)
            return seriesExtracted

        #se la PSO restituisce diff=0 e lamb=1 significa che la porzione di window è stazionaria, o ci sono delle stazionarietà all'interno della serie che la window non riesce a vedere
        #per questo dopo 3 volte che la finestra restituisce una prozione stazionaria
        #restituisco semplicemente la serie nella sua totalità senza scomporla

        if(result[2]==0 and result[3]==1.0):
            counter_stationarity = counter_stationarity+1
            if(counter_stationarity == 3):
                seriesExtracted = list()
                seriesExtracted.append(series)
                return seriesExtracted

        lagBatch= FindAutocorrelationMaxLag2(batch)


        list_par.append(result[2])
        list_lamb.append(round(result[3], 2))
        list_score.append(round(result[4], 2))
        list_window.append((x, y, wind))

        #questo if mi serve per aggiornare il max_autocorr_lag solo nel caso in cui il max_lag visto nella finestra è cambiato in modo significativo
        if(lagBatch!=0 and (lagBatch<max_autocorrelation_lag-2 or lagBatch>max_autocorrelation_lag+2)):
           max_autocorrelation_lag=lagBatch

        #questo if mi serve per aggiornare il max_autocorr_lag solo nel caso in cui ci sono stati cambiamenti significativi della diff
        if ((list_par[i] > list_par[i - 1] + 3 or list_par[i] < list_par[i - 1] - 3) ):
            max_autocorrelation_lag=lagBatch

        # quando c'è un cambiamento nella diff applicata, allora potrebbe significare che c'è un cambiamento di non stazionarietà
        # visto che a volte la diff scelta dalla PSO si confonde con la diff giusta e i suoi multipli, faccio un check per controllare se c'è stato un effettivo cambiamento significativo (la diff che si  muove da un multiplo all'altro non è significativo)

        list_autocorrelation_lags.append(max_autocorrelation_lag)

        #questo if serve ad accorgersi del cambio di non stazionarietà, andando a confrontare gli ultimi 2 valori di max_autocorr_lag registrati
        #se i due valori si discostano in modo significativo, allora la non stazionarietà potrebbe essere cambiata

        if ((list_autocorrelation_lags[i] > list_autocorrelation_lags[i - 1] + 2 or list_autocorrelation_lags[i] < list_autocorrelation_lags[i - 1] - 2) and lastLap==False):
            #quindi vado a ricalcolare il max_autocorrelation_lag con ciò che rimane della serie, droppando la parte analizzata fin ora

            # rimuovo la serie analizzata fin ora
            seriesHalf = series.drop(series.index[0:y])
            # ricalcolo il maxAutocorrelationLag con la serie rimanente
            New_max_autocorrelation_lag = FindAutocorrelationMaxLag2(seriesHalf)

            max_autocorrelation_lag = New_max_autocorrelation_lag

            change=True
            change_station = True
            num_nonStat_find=num_nonStat_find+1

            list_window_change_index.append([x,y])

            # estraggo la porzione di serie vista fino ad ora, che avrà una sua non stazionarietà, diversa dalle altre porzioni di serie
            #sottraggo (wind/2) per essere sicuro di non prendere i valori transitori tra una serie e l'altra
            newCheckPoint = x
            #print('********************************')
            #print(newCheckPoint)
            seriesExtracted = series[oldCheckPoint:newCheckPoint]
            list_series_extracted.append(seriesExtracted)
            oldCheckPoint = y

        # una volta ricalcolato il max_autocorrelation lag, ricalcolo la dimensione della window

        wind = 5 * max_autocorrelation_lag
        x = y
        y = min(len(series), y+wind )

        #questo if serve per aggiornare la finestra a seguito di un cambio di non-stazionarietà
        if(change==True):
            batch = series.iloc[x:y]
            lagBatch = FindAutocorrelationMaxLag2(batch)
            if (lagBatch != 0):
                max_autocorrelation_lag = lagBatch
                list_autocorrelation_lags[i] = max_autocorrelation_lag
            change=False

        # se la window arriva all'ultimo valore della serie
        # fa un'ultima analisi con una window pari alla dimensione della serie
        # così da fare un'analisi della serie nella sua interezza
        if (y == len(series) and Count == 0):
            Count = 1
            x = 0
            wind = len(series)
            lastLap=True

            if (change_station == True):
                seriesExtracted = series[oldCheckPoint:y]
                list_series_extracted.append(seriesExtracted)
            else:
                list_series_extracted.append(series)
        i = i + 1

    return [list_series_extracted,list_window_change_index]





def ATSS_Extract_Subseries2(series):
    #la funzione prende in input una serie contenente diverse non stazionarietà
    #restituisce in output la serie scomposta in sottoserie in base alla non stazionarietà

    # la funzione crea delle finestre per studiare come varia la non stazionarietà della seire, andando ad analizzare come varia la trasformazione applicata dalla PSO nel tempo
    # per rendere le finestre piu generali possibili, ho scelto una grandezza di 5*maxAutocorrelationLag, in modo da essere sicuri di catturare eventuali periodicità
    # la dimensione della window cambia nel tempo, quando vengono individuati cambiamnti significativi della diff (e.g non multipli e non valori vicini)
    # e quando viene identificato anche un cambiamento  significaivo dell' maxAutocorrelation lag
    # a quel punto la serie analizzata fino a quel momento viene droppata, viene ricalcolato il maxAutocorrelationLag sulla serie rimanente e viene ricalcolata la window

    max_autocorrelation_lag = FindAutocorrelationMaxLag(series)
    #nel caso in cui non riesce a trovare un max_autocorr_lag all'inizio, a causa delle troppe non stazionarietà che confondono l'autocorrelazione
    #inizializzo max_auto_lag a 30, in modo da avere una generica finestra di 150, che poi si adatterà successivamente da sola
    if(max_autocorrelation_lag==0):
        max_autocorrelation_lag=30

    list_par = list()
    list_lamb = list()
    list_score = list()
    list_window = list()
    list_series_extracted=list()
    list_autocorrelation_lags=list()

    i = 0  # mi fa muovere lungo la serie
    wind = 5 * max_autocorrelation_lag  # è l'ampiezza della finestra
    x = 0  # l'inizio della finestra
    y = wind  # la fine della finestra
    Count = 0  # mi serve come condizione per analizzare alla fine la serie completa
    lastLap = False  #serve per fare l'ultima analisi con windows=len(series)
    change = False # mi serve per fare la correzione dei lag nell'iterazione in cui c'è cambio di window
    change_station = False  # indica se c'è stato un cambio di stationarietà, serve per estrarre l'ultimo pezzo della serie con non-stazionarietà diversa
    oldCheckPoint = 0  # inizio di una porzione di serie con una certa non stazionarietà
    num_nonStat_find = 1  #serve per tenere traccia di quanti segmenti di serie con diverse non-stazionarietà sono contenuti nella serie
    counter_stationarity = 0 #mi serve per contare quante volte capita che la PSO applica diff=0 e lamb=1.0, cioè non applica trasformazioni , perchè se succede spesso allora non taglio la serie ma la considero nella sua interezza
    while (Count < 2):
        if (Count == 1):
            Count = 2
        batch = series.iloc[x:y]

        try:
            result = ATSS_Stationarize_Window(batch)

        except:
            #se la trasformazione applicata da una finestra va in errore, significa che la finestra non riesce a catturare l'eventuale non stazionaerietà
            #per questo motivo restituisco direttamente la serie nella sua interezza
            seriesExtracted=list()
            seriesExtracted.append(series)
            return seriesExtracted

        #se la PSO restituisce diff=0 e lamb=1 significa che la porzione di window è stazionaria, o ci sono delle stazionarietà all'interno della serie che la window non riesce a vedere
        #per questo dopo 3 volte che la finestra restituisce una prozione stazionaria
        #restituisco semplicemente la serie nella sua totalità senza scomporla, per analizzarla nella sua interezza

        if(result[2]==0 and result[3]==1.0):
            counter_stationarity = counter_stationarity+1
            if(counter_stationarity == 3):
                seriesExtracted = list()
                seriesExtracted.append(series)
                return seriesExtracted

        #calcolo il maxAutocorrelation lag all'interno della finestra, per capire che periodicità c'è nella finestra
        lagBatch = FindAutocorrelationMaxLag2(batch)


        list_par.append(result[2])
        list_lamb.append(round(result[3], 2))
        list_score.append(round(result[4], 2))
        list_window.append((x, y, wind))

        #questo if mi serve per aggiornare il max_autocorr_lag solo nel caso in cui il max_lag visto nella finestra è cambiato in modo significativo
        # questo if mi serve per aggiornare il max_autocorr_lag solo nel caso in cui ci sono stati cambiamenti significativi della diff
        if((lagBatch!=0 and (lagBatch<max_autocorrelation_lag-2 or lagBatch>max_autocorrelation_lag+2)) or (list_par[i] > list_par[i - 1] + 3 or list_par[i] < list_par[i - 1] - 3)  ):
           max_autocorrelation_lag=lagBatch
           print('AAAAAAAAAAAA')

        list_autocorrelation_lags.append(max_autocorrelation_lag)

        #questo if serve ad accorgersi del cambio di non stazionarietà, andando a confrontare gli ultimi 2 valori di max_autocorr_lag registrati
        #se i due valori si discostano in modo significativo, allora la non stazionarietà potrebbe essere cambiata
        if ((list_autocorrelation_lags[i] > list_autocorrelation_lags[i - 1] + 2 or list_autocorrelation_lags[i] < list_autocorrelation_lags[i - 1] - 2) and lastLap==False):
            #quindi vado a ricalcolare il max_autocorrelation_lag con ciò che rimane della serie, droppando la parte analizzata fin ora

            # rimuovo la serie analizzata fin ora
            seriesHalf = series.drop(series.index[0:y])
            seriesHalf.plot(color='black')
            plt.show()
            # ricalcolo il maxAutocorrelationLag con la serie rimanente
            New_max_autocorrelation_lag = FindAutocorrelationMaxLag2(seriesHalf)

            max_autocorrelation_lag = New_max_autocorrelation_lag

            change=True
            change_station = True
            num_nonStat_find=num_nonStat_find+1


            # estraggo la porzione di serie vista fino ad ora, che avrà una sua non stazionarietà, diversa dalle altre porzioni di serie
            newCheckPoint = x
            seriesExtracted = series[oldCheckPoint:newCheckPoint]
            list_series_extracted.append(seriesExtracted)
            oldCheckPoint = y

        # una volta ricalcolato il max_autocorrelation lag, ricalcolo la dimensione della window

        wind = 5 * max_autocorrelation_lag
        x = y
        y = min(len(series), y+wind )

        #questo if serve per aggiornare la finestra a seguito di un cambio di non-stazionarietà
        if(change==True):
            batch = series.iloc[x:y]
            lagBatch = FindAutocorrelationMaxLag2(batch)
            if (lagBatch != 0):
                max_autocorrelation_lag = lagBatch
                list_autocorrelation_lags[i] = max_autocorrelation_lag
            change=False

        # se la window arriva all'ultimo valore della serie
        # fa un'ultima analisi con una window pari alla dimensione della serie
        # così da fare un'analisi della serie nella sua interezza
        if (y == len(series) and Count == 0):
            Count = 1
            x = 0
            wind = len(series)
            lastLap=True

            if (change_station == True):
                seriesExtracted = series[oldCheckPoint:y]
                list_series_extracted.append(seriesExtracted)
            else:
                list_series_extracted.append(series)
        i = i + 1

    return list_series_extracted

def AR1Score2(seriesOriginal, seriesTrasf1, seriesTrasf2, particle, lamb):
    #questa funzione prende la serie trasformata in input
    #effettua una predizione usando un modello arma che decide in automatico i valori migliori per p,q
    #applica alla predizione la trasfromazione inversa
    #confronta la serie trasformata-predetta-invertita con il test set della serie originale
    #calcola l'errore tra le due, questo errore è un indice della information loss causata dalle trasformazioni

    # calcolo range
    serMax = seriesOriginal.max()
    serMin = seriesOriginal.min()
    serRange = serMax - serMin


    # le trasformazioni mi fanno perdere gli indici sottoforma di data
    # quindi me li ricopio dalla serie originale
    seriesTrasf1.index = seriesOriginal.index
    seriesTrasf2.index = seriesOriginal.index

    # preparo i train e test sets
    size = len(seriesOriginal)
    test = int(max((size * 0.1), particle))
    train = size - test

    # questo mi serve per confrontare la serie predetta invertita con l'originale
    seriesTrainOriginal = seriesOriginal.iloc[:-test]
    seriesTestOriginal = seriesOriginal.iloc[train:]

    # questo mi serve per fare l'inversa della diff
    seriesTrainTrasf1 = seriesTrasf1.iloc[:-test]
    seriesTestTrasf1 = seriesTrasf1.iloc[train:]

    # questo mi serve per fare la predizione
    seriesTrainTrasf2 = seriesTrasf2.iloc[:-test]
    seriesTestTrasf2 = seriesTrasf2.iloc[train:]

    test_data = seriesTestTrasf2
    train_data = seriesTrainTrasf2


    # elimino i primi "particle" value, perchè la diff shifta in avanti la serie di un passo "particle" mettendo un numero di 0 pari a particle
    # per non falsare la predizione, bisogna togliere questi 0 che shiftano
    train_data = train_data.drop(train_data.index[0:particle])
    model = ARIMA(train_data, order=(1, 0, 0))

    model_fit = model.fit(disp=0)

    forecast = model_fit.predict(start=len(train_data), end=len(test_data) + len(train_data) - 1)

    seriesPredicted = Series(forecast)
    seriesPredicted.index = seriesTestTrasf2.index
    # seriesPredicted.plot(color='red')
    # seriesTestTrasf2.plot()
    # plt.show()

    # inverto la predizione
    seriesPredictedInv = InvDiffByParticlePredicted(seriesPredicted, seriesTrainTrasf1, particle)
    seriesPredictedInv = InverseYeojohnson(seriesPredictedInv, seriesPredictedInv, lamb)

    # calcolo l'errore tra test set originale e serie trasformata predetta invertita

    seriesTestOriginal = Series(seriesTestOriginal.values)
    rmse = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedInv))
    rmseRange = rmse / serRange

    # seriesPredictedInv.plot(color='red', label='Predicted')
    # seriesTestOriginal.plot(label='Original')
    # plt.legend()
    # plt.title("Particle = {}   lambda= {}   rmse/range={}".format(particle, lamb, rmseRange))
    # plt.show()



    return rmseRange

