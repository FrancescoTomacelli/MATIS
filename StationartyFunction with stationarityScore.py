from pandas import read_csv
import numpy as np
import pyswarms as ps
import Funzioni
from matplotlib import pyplot
import pandas as pd
from pandas import Series
from pandas import DataFrame
from numpy import random
import math




def StationarityFunction(params,ser):
    #series = read_csv('Datasets/Airline_Passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    series=ser
    seriesOriginal=series
    
    p0 = params[0]
    p1 = params[1]


    #le due istruzioni commentate sotto servono per disattivare le trasformazioni
    #p0=1
    #p1=0
    seriesTrasf1 = Funzioni.YeojohnsonTrasform(series, p0)
    seriesTrasf2 = Funzioni.DifferencingByParticleValue(seriesTrasf1, round(p1))




    return (Funzioni.TrendStationarityScore(seriesOriginal,seriesTrasf2) + Funzioni.SeasonStationarityScore(seriesTrasf2)+ Funzioni.AR1Score(seriesOriginal,seriesTrasf1,seriesTrasf2,round(p1),p0))

def f(x,ser):
    n_particles=x.shape[0]

    j=[StationarityFunction(x[i],series) for i in range(n_particles)]
    return np.array(j)

#main

series = read_csv('Datasets/Airline_Passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
seriesOriginal=series

#operazioni per PSO
lenSeriesReduced= round(len(series)/3)
options={'c1':0.5,'c2':0.3, 'w':0.9}
min_bound = np.array([0,0])
max_bound = np.array([1.0,lenSeriesReduced])
bounds = (min_bound,max_bound)
dimensions=2

#operazioni per inizializzare swarm
swarm_size=10
num_iters=10
yeo_init=random.uniform(min_bound[0],max_bound[0])   #valora casuale per inizializzare le particle della yeotrasform
init_lag=Funzioni.FindAutocorrelationMaxLag(series)
init_value=[1.0,init_lag] #metto 1.0 come inizializzazione della yeo, perchè voglio partire dal caso piu semplice, cioè non applica la yeo ma applica la diff di lag=maxAutocorrelationLag
initialize=np.array([init_value for i in range(swarm_size)])

#optimizer= ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dimensions, bounds=bounds, options=options)
optimizer= ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dimensions, bounds=bounds, options=options,init_pos=initialize)
cost,pos = optimizer.optimize(f,iters=num_iters,ser=seriesOriginal)

print(f"Valore minimo: {cost}, Yeojohnson lambda={pos[0]}, DiffByParticle={round(pos[1])}")

#rileggo la serie per plottarla con PrintSeriesTrasform
seriesTrasformed=Funzioni.PrintSeriesTrasform(series, pos[0],pos[1])
#StationarityScore= Funzioni.StationarityScore(seriesTrasformed)
TrendScore=Funzioni.TrendStationarityScore(seriesOriginal,seriesTrasformed)
SeasonScore=Funzioni.SeasonStationarityScore(seriesTrasformed)
print("Ar1Score",cost-TrendScore-SeasonScore)
print("TrendScore =", TrendScore)
print("SeasonScore =",SeasonScore)
#Funzioni.PlotZoomed(seriesOriginal,300,400)


def PSO(series):
    seriesOriginal = series

    # operazioni per PSO
    lenSeriesReduced = round(len(series) / 3)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    min_bound = np.array([0, 0])
    max_bound = np.array([1.0, lenSeriesReduced])
    bounds = (min_bound, max_bound)
    dimensions = 2

    # operazioni per inizializzare swarm
    swarm_size = 10
    num_iters = 10
    yeo_init = random.uniform(min_bound[0],max_bound[0])  # valora casuale per inizializzare le particle della yeotrasform
    init_lag = Funzioni.FindAutocorrelationMaxLag(series)
    init_value = [1.0,init_lag]  # metto 1.0 come inizializzazione della yeo, perchè voglio partire dal caso piu semplice, cioè non applica la yeo ma applica la diff di lag=maxAutocorrelationLag
    initialize = np.array([init_value for i in range(swarm_size)])

    # optimizer= ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dimensions, bounds=bounds, options=options)
    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dimensions, bounds=bounds, options=options,init_pos=initialize)
    cost, pos = optimizer.optimize(f, iters=num_iters)

    print(f"Valore minimo: {cost}, Yeojohnson lambda={pos[0]}, DiffByParticle={round(pos[1])}")

    # rileggo la serie per plottarla con PrintSeriesTrasform
    seriesTrasformed = Funzioni.PrintSeriesTrasform(series, pos[0], pos[1])
    # StationarityScore= Funzioni.StationarityScore(seriesTrasformed)
    TrendScore = Funzioni.TrendStationarityScore(seriesOriginal, seriesTrasformed)
    SeasonScore = Funzioni.SeasonStationarityScore(seriesTrasformed)
    print("Ar1Score", cost - TrendScore - SeasonScore)
    print("TrendScore =", TrendScore)
    print("SeasonScore =", SeasonScore)
    # Funzioni.PlotZoomed(seriesOriginal,300,400)
