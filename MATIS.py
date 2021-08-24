import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot
from pandas import Series
from scipy import signal
from scipy.stats import yeojohnson
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
import warnings
import math
from pandas import DataFrame
from fbprophet import Prophet
import pmdarima as pm
from numpy import random
import pyswarms as ps
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima_process import arma_generate_sample


#Stationarity Checks

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


#Transformations

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

#Inverse Transformations

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

#PSO stationarization

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


#MATIS

def MATIS(series):
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
        return [series, series, [0], [1.0],0]

    #se è non stazionaria, invece, cerchiamo la trasformazione migliore
    else:
                # costruiamo le nan series per ricongiungere le serie estratte trasformate
        seriesTrasf1Recostructed = Create_Nan_Series(series)
        seriesTrasf2Recostructed = Create_Nan_Series(series)

        # vado a estrarre le non stazionarità delle serie
        seriesExtracted = MATIS_Extract_Subseries(series)

        list_diff = list()
        list_lamb = list()
        # vado a trasformare una alla volta le non stazionarietà
        for ser in seriesExtracted:
            seriesTrasf2, seriesTrasf1, diff, lamb, cost = MATIS_Stationarize_Series(ser)
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

def MATIS_Extract_Subseries(series):
    # La funzione prende in input una serie contenente diverse non stazionarietà
    # Restituisce in output la serie scomposta in sottoserie in base alla non stazionarietà
    # La funzione crea delle finestre per studiare come varia la non stazionarietà della seire, andando ad analizzare come varia la trasformazione applicata dalla PSO nel tempo
    # Per rendere le finestre piu generali possibili, ho scelto una grandezza di 5*maxAutocorrelationLag, in modo da essere sicuri di catturare eventuali periodicità
    # La dimensione della window cambia nel tempo, quando vengono individuati cambiamnti significativi della diff  e dell'autocorrelation lag calcolato dalla finestra
    # A quel punto la serie analizzata fino a quel momento viene droppata, viene ricalcolato il maxAutocorrelationLag sulla serie rimanente e viene ricalcolata la window

    # list_par serve per tenere traccia delle diff applicate dalle finestre
    # list_series_extracted serve per raccogliere le subseries individuate
    list_par = list()
    list_series_extracted = list()

    # Calcolo il max autocorrelation lag su tutta la serie, per avere una prima indicazione sulla dimensione della finestra
    max_autocorrelation_lag = FindAutocorrelationMaxLag2(series)

    #dopo aver avuto l'inicazione sulla prima dimensione della finestra
    #vado a raffinare la sua dimensione andando a cacolare il max autocorrelation lag all'interno della finestra (non più su tutta la serie)
    #quello che otteniamo è il window lag, che viene poi usato per dare dimensione alla window
    window_lag = FindAutocorrelationMaxLag2(series[0:max_autocorrelation_lag*5])

    # Variabili per il funzionamento dell'algoritmo
    # i serve per accedere alle varie liste
    # wind è l'ampiezza della finestra
    # x e y sono l'inzio e la fine della finestra
    # End serve come condizione per terminare il ciclo quando ho analizzato tutta la serie
    # change_station serve per tenere traccia dell'avvenimento di un cambio di non stazionarietà
    i = 0
    wind = 5 * window_lag
    x = 0
    y = wind
    End = False  #
    oldCheckPoint = 0
    change_station = False

    while (End == False):

        batch = series.iloc[x:y]


        try:
            result = MATIS_Stationarize_Window(batch)

        except:
            #se la trasformazione applicata da una finestra va in errore, significa che la finestra non riesce a catturare l'eventuale non stazionaerietà
            #per questo motivo restituisco direttamente la serie nella sua interezza
            seriesExtracted=list()
            seriesExtracted.append(series)
            return seriesExtracted


        #calcolo ad ogni iterazione il maxAutocorrelation lag all'interno della finestra, per capire che periodicità c'è nella finestra
        tmp_window_lag = FindAutocorrelationMaxLag2(batch)
        #salvo il diff applicato
        list_par.append(result[2])

        # Questo if serve per accorgersi del cambio di non stazionarietà
        # Quando il correlation lag dell'attuale finestra è diverso  di un fattore 2 dal window_lag computato all'inizio. (il fattore 2 serve perchè lagBatch a volte oscilla di +/- 1 anche quando non ci sono cambiamenti. Quindi il fattore 2 serve per evitare falsi positivi)
        # E quando il differencing si discosta dall'ultimo differencing applicato
        # Allora c'è un cambio di non stazionarietà

        # Una volta entrato nell'if
        # Estraggo la serie vista fin ora (sottoserie con una specifica non stazionarietà)
        # Per individuare la nuova stazionarietà, ripeto i passaggi effettuati all'inzio, quindi:
        # Calcolo il maxAutocorrealtionLag su tutta la serie rimanente per avere una prima indicazione sulla dimensione della finestra
        # Vado a raffinare tale maxAutocorrelationLag calcolando il window_lag a solo sulla prima finestra della serie 0:5*max
        # Ottenuto il nuovo window_lag, aggiorno la dimensione della finestra e continuo l'analisi

        # if (((lagBatch<max_autocorrelation_lag-2 or lagBatch>max_autocorrelation_lag+2)) and (list_par[i] > list_par[i - 1] + 3 or list_par[i] < list_par[i - 1] - 3) ):
        if((tmp_window_lag<window_lag-2 or tmp_window_lag>window_lag+2) and list_par[i]!=list_par[i-1]):
            change_station = True
            seriesHalf = series.drop(series.index[0:y])

            #ricalcolo max_autocorr_lag e window_lag per ricalcolare dimensioni window
            max_autocorrelation_lag= FindAutocorrelationMaxLag2(seriesHalf)
            window_lag = FindAutocorrelationMaxLag2(seriesHalf[0:5*max_autocorrelation_lag])

            # estraggo la porzione di serie vista fino ad ora, che avrà una sua non stazionarietà, diversa dalle altre porzioni di serie
            newCheckPoint = x
            seriesExtracted = series[oldCheckPoint:newCheckPoint]
            list_series_extracted.append(seriesExtracted)
            oldCheckPoint = y

            #ricalcolo la dimensione della finestra con il nuovo max_autocorr_lag trovato
            wind = 5 * window_lag

        #faccio avanzare la finestra
        x = y
        y = min(len(series), y+wind )

        # se la window arriva all'ultimo valore della serie, termino il ciclo mettendo End=True ed estraggo l'eventuale ultimo pezzo analizzato
        if (y == len(series)):
            End = True
            if (change_station == True):
                seriesExtracted = series[oldCheckPoint:y]
                list_series_extracted.append(seriesExtracted)
            else:
                list_series_extracted.append(series)
        #ogni iterazione faccio aumentare i per accedere alle liste dei parametri
        i = i + 1






    return list_series_extracted

def MATIS_Stationarize_Series(series):
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
    result=MATIS_Apply_Transformation(series,pos[0],pos[1])
    seriesTrasf2 = result[0]
    seriesTrasf1 = result[1]
    pos[0] = result[2]
    pos[1] = result[3]


    return [seriesTrasf2, seriesTrasf1, round(pos[1]), pos[0],cost]

def MATIS_Stationarize_Window(series):
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
        result = MATIS_Apply_Transformation(series, pos[0], pos[1])

        seriesTrasf2 = result[0]
        seriesTrasf1 = result[1]
        pos[0] = result[2]
        pos[1] = result[3]

        return [seriesTrasf2, seriesTrasf1, round(pos[1]), pos[0], cost]

def MATIS_Apply_Transformation(series,p0,p1):
    "ripete le trasformazioni fatte dallo swarm una volta che ha trovato i parametri migliori"
    "questo è per ottenere la serie trasformata"

    seriesTrasf1 = YeojohnsonTrasform(series, p0)
    seriesTrasf2 = DifferencingByParticleValue(seriesTrasf1,round(p1))

    seriesTrasf1.index=series.index
    seriesTrasf2.index=series.index

    return [seriesTrasf2,seriesTrasf1,p0,p1]

def MATIS_Invert_Transformation(seriesTrasf2,seriesTrasf1,diff,lamb,scaler):
    #inverto le trasformazioni seguendo l'ordine inverso di applicazione
    seriesInverted = Invert_Normalize_Series(seriesTrasf2,scaler)
    seriesInverted = InvertDiffByParticleValue(seriesTrasf1,seriesInverted,diff)
    seriesInverted = InverseYeojohnson(seriesInverted,lamb)

    seriesInverted = Series(seriesInverted)


    return seriesInverted

def MATIS_Invert_Prediction(seriesPredicted, seriesTrasf1, diff, lamb, scaler):
    # applico le trasformazioni in ordine inverso alla predizione
    seriesPredictedInverted = Invert_Normalize_Series(seriesPredicted, scaler)
    seriesPredictedInverted = InvDiffByParticlePredicted(seriesPredictedInverted, seriesTrasf1, diff)
    seriesPredictedInverted = InverseYeojohnson(seriesPredictedInverted, lamb)

    # la predizione invertita che viene ritornata non ha l'index originale
    # l'index deve essere copiato dal test set
    return seriesPredictedInverted


#utilities MATIS

def Create_Nan_Series(series):
    #crea una serie di NaN seguendo la serie che gli viene data in input
    seriesNan=list()

    for i in range(0, len(series)):
        seriesNan.append(np.nan)

    seriesNan = Series(seriesNan)
    seriesNan.index = series.index

    return seriesNan

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

def FindAutocorrelationMaxLag2(series):
    #Trova i picchi di autocorrelazione per lo sliding window change detection
    #cambia dagli altri per la sensibilità della soglia imposta

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
    #Trova i picchi di autocorrelazione per CheckSeasonality
    # cambia dagli altri per la sensibilità della soglia imposta

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

def FindAutocorrelationMaxLag(series):
    #Trova i picchi di autocorrelazione per Inizializzazione PSO
    # cambia dagli altri per la sensibilità della soglia imposta

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













