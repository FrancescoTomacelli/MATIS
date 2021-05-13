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


        #pyplot.title('Serie originale')
        #seriesOriginal.plot()
        #pyplot.show()

    result=[seriesTrasf2,seriesTrasf1,p0,p1]
    return result

def PlotZoomed(series,start,end):
    series[start:end].plot(color='green')
    pyplot.show()


#Checks

def CheckStationarity(series):
    #la funzione ritorna True se la serie è stazionaria, false altrimenti
    X=series.values
    result=adfuller(X)
    isStat=False

    if(result[1]>0.05):
        isStat=False

    else:
        #ADF trova solo stationarity ai trend, ma non identifica la seasonality
        #Quindi se siamo in questo step significa che la serie è trend-stationaria,
        #quindi rispettiamo le ipotesi della funzione CheckSeasonality
        #Dobbiamo verificare che non abbia seasonality.

        thereIsSeasonality= CheckSeasonality3(series)
        if(thereIsSeasonality==True):
            isStat=False
        else:
            isStat=True

    return isStat

def CheckTrend(series):
    #la funzione ritorna True se la serie contiene un trend, false altrimenti
    X=series.values
    result=adfuller(X)
    thereIsTrend=False

    if(result[1]>0.05):
        thereIsTrend=True

    else:
        thereIsTrend=False

    return thereIsTrend

def CheckSeasonality3(series):
    # la funzione restituisce True se la serie contiene seasonality, altrimenti false
    # lavora con i picchi dell'autocorrelazione
    # da trial and error, ho visto che una soglia ottimale per la presenza di seasonality
    # è data da un numero superiore a 2 picchi, che superano il max tra la soglia dell'autocorrelazione
    # e la soglia 0.3 imposta da me per discriminare outliers

    num_peaks=FindAutocorrelationPeaksSeason(series)
    if(len(num_peaks)>=1):
        thereIsSeason=True
    else:
        thereIsSeason=False

    return thereIsSeason





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

def InverseYeojohnson(seriesOriginal,sertrasf,lamb):
    #in realtà seriesOriginal deve essere sempre la serieTrasf
    inv = list()
    X = seriesOriginal.values
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
        # (perchè per invertire un il valore della predizione allo step [i] mi serve il valore della serie originale allo steo [i-particle]
        # che sono cioè gli ultimi "particle" step del train set
        seriesPredictedInv = list()
        for i in range(0, particle):
            value = seriesPredicted[i] + trainSetModified[len(trainSetModified) - particle + i]
            seriesPredictedInv.append(value)

        # una volta invertiti i primi "particle" step, per predirre i successivi non posso usare piu il train set
        # ma dovrei usare dei valori che rientrano nel test set, ma visto che in teoria noi i valori reali del test set non li conosciamo
        # andiamo ad utilizzare i primi valori della predizione inverita che ci siamo calcolati prima
        for i in range(particle, len(seriesPredicted)):
            value = seriesPredicted[i] + seriesPredictedInv[i - particle]
            seriesPredictedInv.append(value)

    else:
        seriesPredictedInv = seriesPredicted

    seriesPredictedInv = Series(seriesPredictedInv)

    return seriesPredictedInv


#Utilities per check

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
    #print('peakLags2AAAAAAAAAAAAAA', peakLags)
    #print('PeakHeights2AAAAAAAAAAAAAA' , peakHeights)




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
    # per la season stationality, posso usare i picchi che superano la soglia dell'autocorrelation
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

    except:
        #print('Errore AutoAR1 , particle= {}  lamb={}'.format(particle,lamb))
        return 9999999

def AR1ScoreRetrained(seriesOriginal, seriesTrasf1, seriesTrasf2, particle, lamb):
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

    history = [x for x in train_data]
    # elimino i primi "particle" value, perchè la diff shifta in avanti la serie di un passo "particle" mettendo un numero di 0 pari a particle
    # per non falsare la predizione, bisogna togliere questi 0 che shiftano
    train_data = train_data.drop(train_data.index[0:particle])
    predictions=list()
    try:
        for t in range(len(test_data)):
            model = ARIMA(history, order=(1,0,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat[0])
            obs = test_data[t]
            history.append(obs)



        seriesPredicted = Series(predictions)
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

def StationarizeWithPSO_Original(series):
    #questa funzione differisce dall'altra semplicemente nell'inizializzazione dello swarm
    #qui viene inizializzato l'intero swarm  per la diff al picco massimo di autocorrelazione della serie
    #la yeojohnson viene inizializzata ad 1


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

    #optimizer= ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dimensions, bounds=bounds, options=options)
    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dimensions, bounds=bounds, options=options,init_pos=initialize)
    cost, pos = optimizer.optimize(f, iters=num_iters,ser=seriesOriginal)

    print(f"Valore minimo: {cost}, Yeojohnson lambda={pos[0]}, DiffByParticle={round(pos[1])}")

    # rileggo la serie per plottarla con PrintSeriesTrasform
    result = PrintSeriesTrasform(series, pos[0], pos[1])
    seriesTrasf2 = result[0]
    seriesTrasf1= result[1]

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

    fil=open("D:/Universitaa/TESI/tests/immagini/info.txt","a+")
    fil.write('\n\n')
    print(f"Valore minimo: {cost}, Yeojohnson lambda={pos[0]}, DiffByParticle={round(pos[1])}")
    fil.write('Valore Minimo = '+str(cost)+' Yeojohnson lambda = '+ str(pos[0]) + ' DiffByParticle = '+str(round(pos[1]))+'\n')
    fil.write('Ar1Score = ' + str(Ar1Score) + ' TrendScore = ' + str(TrendScore)+ ' SeasonScore = '+ str(SeasonScore) +'\n')
    fil.close()

   #result2=[seriesTrasf2,seriesTrasf1,round(pos[1]),pos[0],TrendScore,SeasonScore,Ar1Score,cost]
    result2 = [seriesTrasf2, seriesTrasf1, round(pos[1]), pos[0], TrendScore, SeasonScore, Ar1Score, cost]
    return result2

def StationarizeWithPSO(series):
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

    print(f"Valore minimo: {cost}, Yeojohnson lambda={pos[0]}, DiffByParticle={round(pos[1])}")

    # rileggo la serie per plottarla con PrintSeriesTrasform
    result = PrintSeriesTrasform(series, pos[0], pos[1])
    seriesTrasf2 = result[0]
    seriesTrasf1= result[1]

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

   #result2=[seriesTrasf2,seriesTrasf1,round(pos[1]),pos[0],TrendScore,SeasonScore,Ar1Score,cost]
    result2 = [seriesTrasf2, seriesTrasf1, round(pos[1]), pos[0], TrendScore, SeasonScore, Ar1Score, cost]
    return result2





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

def LSTM_Prediction(series, size,train,test,particle):
    # restituisce la serie predetta, seguendo il train e test set fornito
    # reference:https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f
    df = DataFrame()
    df["ds"] = [series.index[i] for i in range(len(series))]
    df["y"] = [series[i] for i in range(len(series))]

    #size=len(series)
    #train=int(max(size*0.1,particle))
    #test=size-train
    train_index = series.iloc[:train].index
    test_index = series.iloc[train:].index
    train_series = series.iloc[:train]
    test_series = series.iloc[train:]

    training_set = df.iloc[:train, 1:2].values
    test_set = df.iloc[train:, 1:2].values

    plt.plot(training_set)
    plt.show()
    td = df.iloc[train:, 1:2].values


    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    for i in range(60, train - 60):
        print(i)
        X_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units=1))

    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs=30, batch_size=32)

    # Getting the predicted stock price of 2017
    dataset_train = df.iloc[:train, 1:2]
    dataset_test = df.iloc[train:, 1:2]
    dataset_total = pd.concat((dataset_train, dataset_test), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, test + 60):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    prediction = model.predict(X_test)
    prediction = sc.inverse_transform(prediction)

    forecast = list()
    for i in range(0, len(prediction)):
        yhat = prediction[i][0]
        forecast.append(yhat)
    forecast = Series(forecast)
    forecast.index = test_index

    return forecast

def LSTM_Prediction2(series, size,train,test):
    # restituisce la serie predetta, seguendo il train e test set fornito
    # reference:https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f
    df = DataFrame()
    df["ds"] = [series.index[i] for i in range(len(series))]
    df["y"] = [series[i] for i in range(len(series))]

    #size=len(series)
    #test=int(max(size*0.1,particle))
    #train=size-test

    train_index = series.iloc[:train].index
    test_index = series.iloc[train:].index
    train_series = series.iloc[:train]
    test_series = series.iloc[train:]

    training_set = df.iloc[:train, 1:2].values
    test_set = df.iloc[train:, 1:2].values


    td = df.iloc[train:, 1:2].values


    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    #original val=60
    val=1
    for i in range(val, train - val):
        X_train.append(training_set_scaled[i - val:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units=1))

    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Getting the predicted stock price of 2017
    dataset_train = df.iloc[:train, 1:2]
    dataset_test = df.iloc[train:, 1:2]
    dataset_total = pd.concat((dataset_train, dataset_test), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(val, test + val):
        X_test.append(inputs[i - val:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    prediction = model.predict(X_test)
    prediction = sc.inverse_transform(prediction)

    forecast = list()
    for i in range(0, len(prediction)):
        yhat = prediction[i][0]
        forecast.append(yhat)
    forecast = Series(forecast)
    forecast.index = test_index

    return forecast

def TestPrediction_AutoArima_Prophet(seriesOriginal,seriesTrasf1,seriesTrasf2,particle,lamb):

    # preparo i train e test sets
    size = len(seriesOriginal)
    test = int(max(size*0.1, particle))
    train = size - test

    seriesTrainOriginal = seriesOriginal.iloc[:-test]
    seriesTestOriginal = seriesOriginal.iloc[train:]

    seriesTrainTrasf1 = seriesTrasf1.iloc[:-test]
    seriesTestTrasf1 = seriesTrasf1.iloc[train:]

    seriesTrainTrasf2 = seriesTrasf2.iloc[:-test]
    seriesTestTrasf2 = seriesTrasf2.iloc[train:]

    train_dataOriginal = seriesTrainOriginal
    test_dataOriginal = seriesTestOriginal

    train_dataTrasf2 = seriesTrainTrasf2
    test_dataTrasf2 = seriesTestTrasf2

    # facciamo la predizione con auto ARIMA della serie originale e calcoliamo rmse
    modelOriginal = pm.auto_arima(train_dataOriginal, error_action='ignore', trace=True, suppress_warnings=True)
    forecastOriginal = modelOriginal.predict(test_dataOriginal.shape[0])

    seriesPredictedOriginal = Series(forecastOriginal)
    seriesPredictedOriginal.index = seriesTestOriginal.index

    rmseOriginal = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedOriginal))
    maeOriginal = mean_absolute_error(seriesTestOriginal, seriesPredictedOriginal)

    # facciamo la predizione con auto ARIMA della serie trasformata e calcoliamo rmse

    # elimino i primi particle value settati a 0 che mi sballano il training
    train_dataTrasf2 = train_dataTrasf2.drop(train_dataTrasf2.index[0:particle])


    modelTrasf2 = pm.auto_arima(train_dataTrasf2, error_action='ignore', trace=True, suppress_warnings=True)
    forecastTrasf2 = modelTrasf2.predict(test_dataTrasf2.shape[0])

    seriesPredictedTrasf2 = Series(forecastTrasf2)

    # inverto la predizione
    seriesPredictedInv = InvDiffByParticlePredicted(seriesPredictedTrasf2, seriesTrainTrasf1, particle)
    seriesPredictedInv = InverseYeojohnson(seriesPredictedInv, seriesPredictedInv, lamb)

    seriesPredictedInv.index = seriesTestOriginal.index

    rmseTrasf2 = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedInv))
    maeTrasf2 = mean_absolute_error(seriesTestOriginal, seriesPredictedInv)
    print('*************************************')
    print('MAETrasf2=', maeTrasf2)

    # plottiamo le predizioni
    seriesPredictedTrasf2.index = seriesTestTrasf2.index
    pyplot.title("AutoARIMA prediction series trasformed before inverting")
    seriesPredictedTrasf2.plot(color='red')
    seriesTestTrasf2.plot()
    pyplot.show()

    pyplot.figure()
    pyplot.subplot(211)
    seriesOriginal.plot(color='blue', label='Original')
    pyplot.legend()
    pyplot.title("SeriesOriginal vs SeriesTrasformed")
    pyplot.subplot(212)
    seriesTrasf2.plot(color='green', label='Trasformed')
    pyplot.legend()
    pyplot.show()

    pyplot.figure()
    pyplot.subplot(311)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedOriginal.plot(color='orange', label='PredictedOriginal')
    pyplot.legend()
    pyplot.title("AutoARIMA  rmseOrigin={:.2f}  maeO={:.2f}".format(rmseOriginal, maeOriginal))
    pyplot.subplot(313)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedInv.plot(color='red', label='PredictedTrasf')
    pyplot.legend()
    pyplot.title("AutoARIMA Particle = {}    lambda= {:.2f}    rmseT={:.2f}   maeT={:.2f}".format(particle, lamb, rmseTrasf2,
                                                                                         maeTrasf2))

    pyplot.show()

    # facciamo le predizioni usando prophet

    # facciamo predizione usando serie originale

    result1 = ProphetPredictSeries(seriesOriginal, size, train, test)
    forecastOriginalProp = result1[0]

    seriesPredictedOriginalProp = Series(forecastOriginalProp)
    seriesPredictedOriginalProp.index = seriesTestOriginal.index

    rmseOriginalProp = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedOriginalProp))
    maeOriginalProp = mean_absolute_error(seriesTestOriginal, seriesPredictedOriginalProp)
    # facciamo la predizione usando la serie trasformata
    result = ProphetPredictSeries(seriesTrasf2, size, train, test)
    forecastTrasf2Prop = result[0]

    seriesPredictedTrasf2Prop = Series(forecastTrasf2Prop.values)
    seriesTrainTrasf1Val = Series(seriesTrainTrasf1.values)
    # inverto la predizione
    seriesPredictedInvProp = InvDiffByParticlePredicted(seriesPredictedTrasf2Prop, seriesTrainTrasf1Val,particle)
    seriesPredictedInvProp = InverseYeojohnson(seriesPredictedInvProp, seriesPredictedInvProp, lamb)

    seriesPredictedInvProp.index = seriesTestOriginal.index

    rmseTrasf2Prop = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedInvProp))
    maeTrasf2Prop = mean_absolute_error(seriesTestOriginal, seriesPredictedInvProp)
    print('---------------------------------------')
    print('MAETrasf2Prop=', maeTrasf2Prop)

    # plottiamo le predizioni
    seriesPredictedTrasf2Prop.index = seriesTestTrasf2.index
    #pyplot.title("Prophet prediction series trasformed before inverting")
    #seriesPredictedTrasf2Prop.plot(color='red')
    #seriesTestTrasf2.plot()
    #pyplot.show()

    pyplot.figure()
    pyplot.subplot(311)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedOriginalProp.plot(color='orange', label='PredictedOriginal')
    pyplot.legend()
    pyplot.title("Prophet rmseOrigin={:.2f}  maeOrigin={:.2f}".format(rmseOriginalProp, maeOriginalProp))
    pyplot.subplot(313)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedInvProp.plot(color='red', label='PredictedTrasf')
    pyplot.legend()
    pyplot.title("Prophet Particle = {}    lambda= {:.2f}    rmseTrasf={:.2f}  maeTrasf={:.2f}".format(particle, lamb,
                                                                                                    rmseTrasf2Prop,
                                                                                                    maeTrasf2Prop))

    pyplot.show()

def TestPrediction_AutoArimaRetrain_Prophet(seriesOriginal,seriesTrasf1,seriesTrasf2,particle,lamb):

    # preparo i train e test sets
    size = len(seriesOriginal)
    test = int(max(size*0.1, particle))
    train = size - test

    seriesTrainOriginal = seriesOriginal.iloc[:-test]
    seriesTestOriginal = seriesOriginal.iloc[train:]

    seriesTrainTrasf1 = seriesTrasf1.iloc[:-test]
    seriesTestTrasf1 = seriesTrasf1.iloc[train:]

    seriesTrainTrasf2 = seriesTrasf2.iloc[:-test]
    seriesTestTrasf2 = seriesTrasf2.iloc[train:]

    train_dataOriginal = seriesTrainOriginal
    test_dataOriginal = seriesTestOriginal

    train_dataTrasf2 = seriesTrainTrasf2
    test_dataTrasf2 = seriesTestTrasf2

    PassoPrediction = train_dataOriginal.drop(train_dataOriginal.index[1:len(train_dataOriginal)])
    historyOriginal = [x for x in train_dataOriginal]
    predictionsOriginal = list()

    modelOriginal = pm.auto_arima(historyOriginal, error_action='ignore', trace=True, suppress_warnings=True)
    orderAutoArima = modelOriginal.to_dict()['order']
    '''
    for t in range(len(test_dataOriginal)):
        model = ARIMA(historyOriginal, order=orderAutoArima)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictionsOriginal.append(yhat[0])
        obs = test_dataOriginal[t]
        historyOriginal.append(obs)
    '''

    #seriesPredictedOriginal = Series(predictionsOriginal)
    seriesPredictedOriginal=test_dataOriginal
    seriesPredictedOriginal.index = seriesTestOriginal.index



    rmseOriginal = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedOriginal))
    maeOriginal = mean_absolute_error(seriesTestOriginal, seriesPredictedOriginal)

    # facciamo la predizione con auto ARIMA della serie trasformata e calcoliamo rmse

    # elimino i primi particle value settati a 0 che mi sballano il training
    train_dataTrasf2 = train_dataTrasf2.drop(train_dataTrasf2.index[0:particle])

    PassoPrediction= train_dataTrasf2.drop(train_dataTrasf2.index[1:len(train_dataTrasf2)])
    historyTrasf2= [x for x in train_dataTrasf2]
    predictionsTrasf2=list()

    modelTrasf2 = pm.auto_arima(historyTrasf2, error_action='ignore', trace=True, suppress_warnings=True)
    orderAutoArimaTrasf2 = modelTrasf2.to_dict()['order']
    for t in range(len(test_dataTrasf2)):
        model = ARIMA(historyTrasf2, order=orderAutoArimaTrasf2)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat=output[0]
        predictionsTrasf2.append(yhat[0])
        obs = test_dataTrasf2[t]
        historyTrasf2.append(obs)


    seriesPredictedTrasf2 = Series(predictionsTrasf2)

    # inverto la predizione
    seriesPredictedInv = InvDiffByParticlePredicted(seriesPredictedTrasf2, seriesTrainTrasf1, particle)
    seriesPredictedInv = InverseYeojohnson(seriesPredictedInv, seriesPredictedInv, lamb)

    seriesPredictedInv.index = seriesTestOriginal.index

    rmseTrasf2 = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedInv))
    maeTrasf2 = mean_absolute_error(seriesTestOriginal, seriesPredictedInv)
    print('*************************************')
    print('MAETrasf2=', maeTrasf2)

    # plottiamo le predizioni
    seriesPredictedTrasf2.index = seriesTestTrasf2.index
    #pyplot.title("AutoARIMA prediction series trasformed before inverting")
    #seriesPredictedTrasf2.plot(color='red')
    #seriesTestTrasf2.plot()
    #pyplot.show()

    pyplot.figure()
    pyplot.subplot(211)
    seriesOriginal.plot(color='blue', label='Original')
    pyplot.legend()
    pyplot.title("SeriesOriginal vs SeriesTrasformed")
    pyplot.subplot(212)
    seriesTrasf2.plot(color='green', label='Trasformed')
    pyplot.legend()
    pyplot.show()

    pyplot.figure()
    pyplot.subplot(311)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedOriginal.plot(color='orange', label='PredictedOriginal')
    pyplot.legend()
    pyplot.title("AutoARIMA  rmseOrigin={:.2f}  maeO={:.2f}".format(rmseOriginal, maeOriginal))
    pyplot.subplot(313)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedInv.plot(color='red', label='PredictedTrasf')
    pyplot.legend()
    pyplot.title(
        "AutoARIMA Particle = {}    lambda= {:.2f}    rmseT={:.2f}   maeT={:.2f}".format(particle, lamb, rmseTrasf2,
                                                                                         maeTrasf2))

    pyplot.show()

    # facciamo le predizioni usando prophet

    # facciamo predizione usando serie originale

    result1 = ProphetPredictSeries(seriesOriginal, size, train, test)
    forecastOriginalProp = result1[0]

    seriesPredictedOriginalProp = Series(forecastOriginalProp)
    seriesPredictedOriginalProp.index = seriesTestOriginal.index

    rmseOriginalProp = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedOriginalProp))
    maeOriginalProp = mean_absolute_error(seriesTestOriginal, seriesPredictedOriginalProp)
    # facciamo la predizione usando la serie trasformata
    result = ProphetPredictSeries(seriesTrasf2, size, train, test)
    forecastTrasf2Prop = result[0]

    seriesPredictedTrasf2Prop = Series(forecastTrasf2Prop.values)
    seriesTrainTrasf1Val = Series(seriesTrainTrasf1.values)
    # inverto la predizione
    seriesPredictedInvProp = InvDiffByParticlePredicted(seriesPredictedTrasf2Prop, seriesTrainTrasf1Val,particle)
    seriesPredictedInvProp = InverseYeojohnson(seriesPredictedInvProp, seriesPredictedInvProp, lamb)

    seriesPredictedInvProp.index = seriesTestOriginal.index

    rmseTrasf2Prop = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedInvProp))
    maeTrasf2Prop = mean_absolute_error(seriesTestOriginal, seriesPredictedInvProp)
    print('---------------------------------------')
    print('MAETrasf2Prop=', maeTrasf2Prop)

    # plottiamo le predizioni
    seriesPredictedTrasf2Prop.index = seriesTestTrasf2.index
    pyplot.title("Prophet prediction series trasformed before inverting")
    seriesPredictedTrasf2Prop.plot(color='red')
    seriesTestTrasf2.plot()
    pyplot.show()

    pyplot.figure()
    pyplot.subplot(311)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedOriginalProp.plot(color='orange', label='PredictedOriginal')
    pyplot.legend()
    pyplot.title("Prophet rmseOrigin={:.2f}  maeOrigin={:.2f}".format(rmseOriginalProp, maeOriginalProp))
    pyplot.subplot(313)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedInvProp.plot(color='red', label='PredictedTrasf')
    pyplot.legend()
    pyplot.title("Prophet Particle = {}    lambda= {:.2f}    rmseTrasf={:.2f}  maeTrasf={:.2f}".format(particle, lamb,
                                                                                                    rmseTrasf2Prop,
                                                                                                    maeTrasf2Prop))

    pyplot.show()

def TestPrediction_AutoArima_Prophet_LSTM(seriesOriginal,seriesTrasf1,seriesTrasf2,particle,lamb,counter_photo):

    # preparo i train e test sets
    size = len(seriesOriginal)
    test = int(max(size*0.1, particle))
    train = size - test

    seriesTrainOriginal = seriesOriginal.iloc[:-test]
    seriesTestOriginal = seriesOriginal.iloc[train:]

    seriesTrainTrasf1 = seriesTrasf1.iloc[:-test]
    seriesTestTrasf1 = seriesTrasf1.iloc[train:]

    seriesTrainTrasf2 = seriesTrasf2.iloc[:-test]
    seriesTestTrasf2 = seriesTrasf2.iloc[train:]

    train_dataOriginal = seriesTrainOriginal
    test_dataOriginal = seriesTestOriginal

    train_dataTrasf2 = seriesTrainTrasf2
    test_dataTrasf2 = seriesTestTrasf2

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


    modelTrasf2 = pm.auto_arima(train_dataTrasf2, error_action='ignore', trace=True, suppress_warnings=True)
    forecastTrasf2 = modelTrasf2.predict(test_dataTrasf2.shape[0])

    seriesPredictedTrasf2 = Series(forecastTrasf2)

    # inverto la predizione
    seriesPredictedInv = InvDiffByParticlePredicted(seriesPredictedTrasf2, seriesTrainTrasf1, particle)
    seriesPredictedInv = InverseYeojohnson(seriesPredictedInv, seriesPredictedInv, lamb)

    seriesPredictedInv.index = seriesTestOriginal.index

    rmseTrasf2 = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedInv))
    maeTrasf2 = mean_absolute_error(seriesTestOriginal, seriesPredictedInv)
    autocorr_residTrasf2 = Quantif_Autocorr_Residual(seriesPredictedInv, seriesTestOriginal)


    # plottiamo le predizioni
    seriesPredictedTrasf2.index = seriesTestTrasf2.index
    #pyplot.title("AutoARIMA prediction series trasformed before inverting")
    #seriesPredictedTrasf2.plot(color='red')
    #seriesTestTrasf2.plot()
    #pyplot.show()

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
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedOriginal.plot(color='orange', label='PredictedOriginal')
    pyplot.legend()
    pyplot.title("AutoARIMAOrigin  rmse={:.2f}  mae={:.2f}  autocorrRes={:.2f}".format(rmseOriginal, maeOriginal, autocorr_residOriginal))
    pyplot.subplot(313)
    seriesTestOriginal.plot(color='blue', label='Original')
    seriesPredictedInv.plot(color='red', label='PredictedTrasf')
    pyplot.legend()
    pyplot.title("AutoARIMATrasf rmse={:.2f}   mae={:.2f}  autcorrRes={:.2f}".format(rmseTrasf2,maeTrasf2, autocorr_residTrasf2))

    pyplot.savefig('D:/Universitaa/TESI/tests/immagini/Syn_'+ str(counter_photo) +'.png')
    counter_photo = counter_photo + 1
    pyplot.show()

    # facciamo le predizioni usando prophet
    # facciamo predizione usando serie originale

    result1 = ProphetPredictSeries(seriesOriginal, size, train, test)
    forecastOriginalProp = result1[0]

    seriesPredictedOriginalProp = Series(forecastOriginalProp)
    seriesPredictedOriginalProp.index = seriesTestOriginal.index

    rmseOriginalProp = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedOriginalProp))
    maeOriginalProp = mean_absolute_error(seriesTestOriginal, seriesPredictedOriginalProp)
    autocorr_residOriginal_Prop = Quantif_Autocorr_Residual(seriesPredictedOriginalProp, seriesTestOriginal)
    # facciamo la predizione usando la serie trasformata
    result = ProphetPredictSeries(seriesTrasf2, size, train, test)
    forecastTrasf2Prop = result[0]

    seriesPredictedTrasf2Prop = Series(forecastTrasf2Prop.values)
    seriesTrainTrasf1Val = Series(seriesTrainTrasf1.values)
    # inverto la predizione
    seriesPredictedInvProp = InvDiffByParticlePredicted(seriesPredictedTrasf2Prop, seriesTrainTrasf1Val,particle)
    seriesPredictedInvProp = InverseYeojohnson(seriesPredictedInvProp, seriesPredictedInvProp, lamb)

    seriesPredictedInvProp.index = seriesTestOriginal.index

    rmseTrasf2Prop = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedInvProp))
    maeTrasf2Prop = mean_absolute_error(seriesTestOriginal, seriesPredictedInvProp)
    autocorr_residTrasf2_Prop = Quantif_Autocorr_Residual(seriesPredictedInvProp, seriesTestOriginal)


    # plottiamo le predizioni
    seriesPredictedTrasf2Prop.index = seriesTestTrasf2.index
    #pyplot.title("Prophet prediction series trasformed before inverting")
    #seriesPredictedTrasf2Prop.plot(color='red')
    #seriesTestTrasf2.plot()
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

    pyplot.savefig('D:/Universitaa/TESI/tests/immagini/Syn_'+ str(counter_photo) +'.png')
    counter_photo = counter_photo + 1
    pyplot.show()

    # facciamo la predizione con LSTM
    # iniziamo con la serie originale
    seriesPredictedOriginalLSTM = LSTM_Prediction2(seriesOriginal, size, train, test)
    #calcoliamo rmse,mae e quantifichiamo l'autocorrelazione del residuo
    rmseOriginalLSTM = sqrt(mean_squared_error(seriesTestOriginal, seriesPredictedOriginalLSTM))
    maeOriginalLSTM = mean_absolute_error(seriesTestOriginal, seriesPredictedOriginalLSTM)
    autocorr_residOriginal_LSTM= Quantif_Autocorr_Residual(seriesPredictedOriginalLSTM,seriesTestOriginal)

    # facciamo la predizione usando la serie trasformata
    seriesPredictedTrasf2LSTM = LSTM_Prediction2(seriesTrasf2, size, train, test)

    # inverto la predizione
    seriesPredictedInvLSTM = InvDiffByParticlePredicted(seriesPredictedTrasf2LSTM, seriesTrainTrasf1, particle)
    seriesPredictedInvLSTM = InverseYeojohnson(seriesPredictedInvLSTM, seriesPredictedInvLSTM, lamb)
    seriesPredictedInvLSTM.index = seriesTestTrasf2.index
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
    pyplot.savefig('D:/Universitaa/TESI/tests/immagini/Syn_'+ str(counter_photo) +'.png')
    counter_photo = counter_photo + 1
    pyplot.show()


#syntetic series generator

def GenerateSynSeries(length,RangeNoise,orderTrend,SinAmpl,fs,f):
    #creiamo la sine wave

    t = length
    samples = np.linspace(0, t, int(fs * t), endpoint=False)
    signal = np.sin(2 * np.pi * f * samples) * SinAmpl

    sineWave = Series(signal)

    linearTrend = list()
    for i in range(0,length):
        data = i * orderTrend
        linearTrend.append(data)
    linearTrend = Series(linearTrend)

    random.seed(9001)
    x = [random.randint(-RangeNoise, RangeNoise) for i in range(0, length)]
    x = Series(x)

    synSeries1 = list()
    for i in range(0, length):
        data = x[i] + sineWave[i]+linearTrend[i]
        synSeries1.append(data)

    synSeries1 = Series(synSeries1)

    return synSeries1

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



def Stationarize_PSO_Window(series,counter_photo):
    # la funzione applica crea delle finestre per studiare come varia la trasformazione applicata dalla PSO nel tempo
    # quindi per studiare come la non stazionarietà varia nel tempo
    # per rendere le finestre piu generali possibili, ho scelto una grandezza di 5*maxAutocorrelationLag, in modo da essere sicuri di catturare eventuali periodicità
    # la dimensione della window cambia nel tempo, quando vengono individuati cambiamnti significativi della diff (e.g non multipli e non valori vicini)
    # e quando viene identificato anche un cambiamento  significaivo dell' maxAutocorrelation lag
    # a quel punto la serie analizzata fino a quel momento viene droppata, viene ricalcolato il maxAutocorrelationLag sulla serie rimanente e viene ricalcolata la window

    max_autocorrelation_lag = FindAutocorrelationMaxLag2(series)
    autocorrelation_peaks = GetAutocorrelationLags(series)
    autocorrelation_heights = GetAutocorrelationLagsHeights(series)
    list_par = list()
    list_lamb = list()
    list_score = list()
    list_window = list()
    list_series_extracted=list()

    i = 0  # mi fa muovere lungo la serie
    wind = 5 * max_autocorrelation_lag  # è l'ampiezza della finestra
    x = 0  # l'inizio della finestra
    y = wind  # la fine della finestra
    Count = 0  # mi serve come condizione per analizzare alla fine la serie completa
    change_station = False  # indica se c'è stato un cambio di stationarietà, serve per estrarre l'ultimo pezzo della serie con non-stazionarietà diversa
    oldCheckPoint = 0  # inizio di una porzione di serie con una certa non stazionarietà
    newCheckPoint = 0  # fine di una porizione di serie con una certa non stazionarietà
    num_nonStat_find = 1
    while (Count < 2):
        if (Count == 1):
            Count = 2
        batch = series.iloc[x:y]
        # print('x = {}  y = {} maxlag = {} window = {} '.format(x, y, max_autocorrelation_lag, len(batch)))

        seriesOriginal = batch
        result = StationarizeWithPSO(batch)

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
        pyplot.savefig('D:/Universitaa/TESI/tests/immagini/Syn' + str(counter_photo) + '_.png')
        counter_photo = counter_photo + 1
        plt.show()

        print('Autocorrelation lag =', max_autocorrelation_lag)
        print('Autocorrelation peaks= ', autocorrelation_peaks)
        print('Autocorrelation Heights =', autocorrelation_heights)
        # print("list_par[i]= {}  list_par[i-1]= {} , i={}".format(list_par[i], list_par[i - 1], i))

        # quando c'è un cambiamento nella diff applicata, allora potrebbe significare che c'è un cambiamento di non stazionarietà
        # visto che a volte la diff scelta dalla PSO si confonde con la diff giusta e i suoi multipli, faccio un check per controllare se c'è stato un effettivo cambiamento significativo (la diff che si  muove da un multiplo all'altro non è significativo)
        multiplo = False
        if (not (list_par[i] != list_par[i - 1] * 2 and list_par[i] != list_par[i - 1] * 3 and  list_par[i] != list_par[i-1] * 4 and list_par[i - 1] !=list_par[i] * 2 and list_par[i - 1] != list_par[i] * 3 and list_par[i - 1] != list_par[i] * 4)):
            multiplo = True

        # queste condizioni servono per accorgersi del cambio di stazionarietà.  Visto che a volte la diff si confonde con i multipli della periodicità, ho messo questa condizione con i multipli , in modo da non cambiare finestra in modo sbagliato
        if ((list_par[i] > list_par[i - 1] + 3 or list_par[i] < list_par[i - 1] - 3) and multiplo == False):
            print("list_par[i]  ",list_par[i])
            print("list_par[i-1]  ",list_par[i-1])
            # rimuovo la serie analizzata fin ora
            seriesHalf = series.drop(series.index[0:y])
            # ricalcolo il maxAutocorrelationLag con la serie rimanente
            New_max_autocorrelation_lag = FindAutocorrelationMaxLag2(seriesHalf)
            New_peaks = GetAutocorrelationLags(seriesHalf)
            New_heights = GetAutocorrelationLagsHeights(seriesHalf)
            # se il nuovo MaxAutocorrelationLag è cambiato in modo significativo ed è !=0, allora cambio la dimensione della window
            if (New_max_autocorrelation_lag != 0 and (
                    New_max_autocorrelation_lag > max_autocorrelation_lag + 1 or New_max_autocorrelation_lag < max_autocorrelation_lag - 1)):
                max_autocorrelation_lag = New_max_autocorrelation_lag
                autocorrelation_peaks=New_peaks
                autocorrelation_heights = New_heights

                change_station = True
                num_nonStat_find=num_nonStat_find+1


                # estraggo la porzione di serie vista fino ad ora, che avrà una sua non stazionarietà, diversa dalle altre porzioni di serie
                newCheckPoint = x
                #nell'estrazione sommo e tolgo wind, così da estrarre solo la dinamica predominante della sottoserie
                #evitando di prendere i valori di transitori tra una sottoserie e l'altra
                seriesRemaning = series.drop(series.index[(oldCheckPoint+wind):(newCheckPoint-wind)])

                #seriesRemaning.plot(color='violet')
                #plt.show()

                seriesExtracted = series.drop(seriesRemaning.index)
                seriesExtracted.plot(color='orange')
                plt.show()
                list_series_extracted.append(seriesExtracted)
                oldCheckPoint = newCheckPoint

        # una volta ricalcolato il max_autocorrelation lag, ricalcolo la dimensione della window

        wind = 5 * max_autocorrelation_lag
        x = y
        y = min(len(series), y + wind)

        # se la window arriva all'ultimo valore della serie
        # fa un'ultima analisi con una window pari alla dimensione della serie
        # così da fare un'analisi della serie nella sua interezza
        if (y == len(series) and Count == 0):
            Count = 1
            x = 0
            wind = len(series)

            if (change_station == True):
                seriesExtracted = series.drop(series.index[0:(newCheckPoint)])
                #seriesExtracted.plot(color='orange')
                list_series_extracted.append(seriesExtracted)
                #plt.show()

            else:
                list_series_extracted.append(series)
        i = i + 1

    print('list_par: ', list_par)
    print('list_lamb: ', list_lamb)
    print('list_score', list_score)
    print('list_window', list_window)
    print('num_nonStat_find ', num_nonStat_find)

    fil=open("D:/Universitaa/TESI/tests/immagini/info.txt","a+")
    fil.write('Differencing applied :  '+str(list_par)+' \n')
    fil.write('Y.J. Lambda applied : ' + str(list_lamb)+' \n')
    fil.write('Scores : '+ str(list_score)+' \n')
    fil.write('Windows : '+ str(list_window)+ ' \n')
    fil.write('Numero non stazionarietà trovate : ' + str(num_nonStat_find) + '\n')
    fil.close()

    return list_series_extracted

def Stationarize_PSO_Window2(series,counter_photo,period1,period2,period3):
    # la funzione applica crea delle finestre per studiare come varia la trasformazione applicata dalla PSO nel tempo
    # quindi per studiare come la non stazionarietà varia nel tempo
    # per rendere le finestre piu generali possibili, ho scelto una grandezza di 5*maxAutocorrelationLag, in modo da essere sicuri di catturare eventuali periodicità
    # la dimensione della window cambia nel tempo, quando vengono individuati cambiamnti significativi della diff (e.g non multipli e non valori vicini)
    # e quando viene identificato anche un cambiamento  significaivo dell' maxAutocorrelation lag
    # a quel punto la serie analizzata fino a quel momento viene droppata, viene ricalcolato il maxAutocorrelationLag sulla serie rimanente e viene ricalcolata la window

    max_autocorrelation_lag = FindAutocorrelationMaxLag2(series)
    print('MAXXXXXXXXXXX', max_autocorrelation_lag)
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
    while (Count < 2):
        if (Count == 1):
            Count = 2
        batch = series.iloc[x:y]
        # print('x = {}  y = {} maxlag = {} window = {} '.format(x, y, max_autocorrelation_lag, len(batch)))

        seriesOriginal = batch
        result = StationarizeWithPSO(batch)
        lagBatch= FindAutocorrelationMaxLag2(batch)
        print('lagBatch   ', lagBatch)
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
        pyplot.savefig('D:/Universitaa/TESI/tests/immagini/Syn_'+str(period1)+'_'+str(period2)+'_'+str(period3)+'_' + str(counter_photo) + '_.png')
        counter_photo = counter_photo + 1
        plt.show()

        if(lagBatch!=0 and (lagBatch<max_autocorrelation_lag-2 or lagBatch>max_autocorrelation_lag+2)):
         max_autocorrelation_lag=lagBatch
         print('111111111111111')



        if ((list_par[i] > list_par[i - 1] + 3 or list_par[i] < list_par[i - 1] - 3) ):
            max_autocorrelation_lag=lagBatch
            print('22222222222222222222')
        # quando c'è un cambiamento nella diff applicata, allora potrebbe significare che c'è un cambiamento di non stazionarietà
        # visto che a volte la diff scelta dalla PSO si confonde con la diff giusta e i suoi multipli, faccio un check per controllare se c'è stato un effettivo cambiamento significativo (la diff che si  muove da un multiplo all'altro non è significativo)
        print('Autocorrelation lag =', max_autocorrelation_lag)
        list_autocorrelation_lags.append(max_autocorrelation_lag)

        # queste condizioni servono per accorgersi del cambio di stazionarietà.  Visto che a volte la diff si confonde con i multipli della periodicità, ho messo questa condizione con i multipli , in modo da non cambiare finestra in modo sbagliato
        if ((list_autocorrelation_lags[i] > list_autocorrelation_lags[i - 1] + 2 or list_autocorrelation_lags[i] < list_autocorrelation_lags[i - 1] - 2) and lastLap==False):
            print('3333333333333333')
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
            #nell'estrazione sommo e tolgo wind, così da estrarre solo la dinamica predominante della sottoserie
            #evitando di prendere i valori di transitori tra una sottoserie e l'altra
            seriesRemaning = series.drop(series.index[(oldCheckPoint+wind):(newCheckPoint-wind)])

            #seriesRemaning.plot(color='violet')
            #plt.show()

            seriesExtracted = series.drop(seriesRemaning.index)
            list_series_extracted.append(seriesExtracted)
            oldCheckPoint = newCheckPoint

        # una volta ricalcolato il max_autocorrelation lag, ricalcolo la dimensione della window

        wind = 5 * max_autocorrelation_lag
        x = y
        y = min(len(series), y + wind)

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
                seriesExtracted = series.drop(series.index[0:(newCheckPoint+oldwind)])
                #seriesExtracted.plot(color='orange')
                list_series_extracted.append(seriesExtracted)
                #plt.show()

            else:
                list_series_extracted.append(series)
        i = i + 1

    print('list_par: ', list_par)
    print('list_lamb: ', list_lamb)
    print('list_score', list_score)
    print('list_window', list_window)
    print('num_nonStat_find ', num_nonStat_find)

    fil=open("D:/Universitaa/TESI/tests/immagini/info.txt","a+")
    fil.write("Period 1 = "+str(period1)+" Period 2 = "+str(period2)+" Period3 = "+str(period3)+"\n")
    fil.write('Differencing applied :  '+str(list_par)+' \n')
    fil.write('Y.J. Lambda applied : ' + str(list_lamb)+' \n')
    fil.write('Scores : '+ str(list_score)+' \n')
    fil.write('Windows : '+ str(list_window)+ ' \n')
    fil.write('Numero non stazionarietà trovate : ' + str(num_nonStat_find) + '\n')
    fil.close()

    k=1
    for ser in list_series_extracted:
        ser.plot(color='red')
        pyplot.savefig('D:/Universitaa/TESI/tests/immagini/series_extracted_' + str(k) + '_.png')
        plt.show()
        k=k+1

    return list_series_extracted


def Stationarize_PSO_Window3(series,counter_photo,period1,period2,period3,period4):
    #in questa versione cambia il modo di estrarre le serie

    # la funzione applica crea delle finestre per studiare come varia la trasformazione applicata dalla PSO nel tempo
    # quindi per studiare come la non stazionarietà varia nel tempo
    # per rendere le finestre piu generali possibili, ho scelto una grandezza di 5*maxAutocorrelationLag, in modo da essere sicuri di catturare eventuali periodicità
    # la dimensione della window cambia nel tempo, quando vengono individuati cambiamnti significativi della diff (e.g non multipli e non valori vicini)
    # e quando viene identificato anche un cambiamento  significaivo dell' maxAutocorrelation lag
    # a quel punto la serie analizzata fino a quel momento viene droppata, viene ricalcolato il maxAutocorrelationLag sulla serie rimanente e viene ricalcolata la window

    max_autocorrelation_lag = FindAutocorrelationMaxLag(series)
    #nel caso in cui non riesce a trovare un max_autocorr_lag all'inizio, a causa delle troppe non stazionarietà che confondono l'autocorrelazione
    #inizializzo max_auto_lag a 30, in modo da avere una generica finestra di 150, che poi si adatterà successivamente da sola
    if(max_autocorrelation_lag==0):
        max_autocorrelation_lag=30
    print('MAXXXXXXXXXXX', max_autocorrelation_lag)
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
    while (Count < 2):
        if (Count == 1):
            Count = 2
        batch = series.iloc[x:y]
        # print('x = {}  y = {} maxlag = {} window = {} '.format(x, y, max_autocorrelation_lag, len(batch)))

        seriesOriginal = batch
        result = StationarizeWithPSO(batch)
        lagBatch= FindAutocorrelationMaxLag2(batch)
        print('lagBatch   ', lagBatch)
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
        pyplot.savefig('D:/Universitaa/TESI/tests/immagini/Syn_'+str(period1)+'_'+str(period2)+'_'+str(period3)+'_'+ str(period4)+'_'+ str(counter_photo) + '_.png')
        counter_photo = counter_photo + 1
        plt.show()

        if(lagBatch!=0 and (lagBatch<max_autocorrelation_lag-2 or lagBatch>max_autocorrelation_lag+2)):
         max_autocorrelation_lag=lagBatch
         print('111111111111111')



        if ((list_par[i] > list_par[i - 1] + 3 or list_par[i] < list_par[i - 1] - 3) ):
            max_autocorrelation_lag=lagBatch
            print('22222222222222222222')
        # quando c'è un cambiamento nella diff applicata, allora potrebbe significare che c'è un cambiamento di non stazionarietà
        # visto che a volte la diff scelta dalla PSO si confonde con la diff giusta e i suoi multipli, faccio un check per controllare se c'è stato un effettivo cambiamento significativo (la diff che si  muove da un multiplo all'altro non è significativo)
        print('Autocorrelation lag =', max_autocorrelation_lag)
        list_autocorrelation_lags.append(max_autocorrelation_lag)

        # queste condizioni servono per accorgersi del cambio di stazionarietà.  Visto che a volte la diff si confonde con i multipli della periodicità, ho messo questa condizione con i multipli , in modo da non cambiare finestra in modo sbagliato
        if ((list_autocorrelation_lags[i] > list_autocorrelation_lags[i - 1] + 2 or list_autocorrelation_lags[i] < list_autocorrelation_lags[i - 1] - 2) and lastLap==False):
            print('3333333333333333')
            # rimuovo la serie analizzata fin ora
            seriesHalf = series.drop(series.index[0:y])
            # ricalcolo il maxAutocorrelationLag con la serie rimanente
            New_max_autocorrelation_lag = FindAutocorrelationMaxLag2(seriesHalf)

            max_autocorrelation_lag = New_max_autocorrelation_lag


            change=True
            change_station = True
            num_nonStat_find=num_nonStat_find+1


            # estraggo la porzione di serie vista fino ad ora, che avrà una sua non stazionarietà, diversa dalle altre porzioni di serie

            newCheckPoint = x-int(wind/2)




            seriesExtracted = series[oldCheckPoint:newCheckPoint]
            list_series_extracted.append(seriesExtracted)
            oldCheckPoint = y-int(wind/2)

        # una volta ricalcolato il max_autocorrelation lag, ricalcolo la dimensione della window

        wind = 5 * max_autocorrelation_lag
        x = y
        y = min(len(series), y + wind)

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

    fil=open("D:/Universitaa/TESI/tests/immagini/info.txt","a+")
    fil.write("Period 1 = "+str(period1)+" Period 2 = "+str(period2)+" Period3 = "+str(period3)+ " Period4 =" +str(period4)+"\n")
    fil.write('Differencing applied :  '+str(list_par)+' \n')
    fil.write('Y.J. Lambda applied : ' + str(list_lamb)+' \n')
    fil.write('Scores : '+ str(list_score)+' \n')
    fil.write('Windows : '+ str(list_window)+ ' \n')
    fil.write('Numero non stazionarietà trovate : ' + str(num_nonStat_find) + '\n')
    fil.close()

    k=1
    for ser in list_series_extracted:
        ser.plot(color='red')
        pyplot.savefig('D:/Universitaa/TESI/tests/immagini/series_extracted_' + str(k) + '_.png')
        plt.show()
        k=k+1

    return list_series_extracted







