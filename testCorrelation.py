import Funzioni
from pandas import read_csv
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from numpy import random
from pandas import Series
import numpy as np
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame


seriesIndex=read_csv('Correlation Datasets/covid - Copia.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

fs = 1
f=1/12
t=len(seriesIndex)
samples = np.linspace(0, t, int(fs*t), endpoint=False)
ampl=300
signal = np.sin(2 * np.pi * f * samples)*ampl

sineWave=Series(signal)
#sineWave.index=seriesIndex.index

#plt.title("Sine Wave Ampl[-70,70] [T=12]")
#sineWave.plot(color='black')
#plt.show()

linearTrend=list()
for i in range(0,len(seriesIndex)):
    data = i*10
    linearTrend.append(data)
linearTrend=Series(linearTrend)

#plt.title("Linear Trend, MultFactor=10")
#linearTrend.plot(color='black')
#plt.show()



random.seed(9001)
x = [random.randint(1,50) for i in range(0,len(seriesIndex))]
y = [random.randint(1,50) for i in range(0,len(seriesIndex))]

x=Series(x)
y=Series(y)
resultXY=pearsonr(x,y)
corrXY=resultXY[0]
#plt.figure()
#plt.subplot(211)
#x.plot(color='red',label='x')
#plt.legend()
#plt.title("X=rand(1,50)  Y=rand(1,50) Pearson={}".format(corrXY))
#plt.subplot(212)
#y.plot(color='orange',label='y')
#plt.legend()
#plt.show()



synSeries1=list()
for i in range(0,len(seriesIndex)):
    data =  x[i] + sineWave[i] + linearTrend[i]
    synSeries1.append(data)

synSeries1=Series(synSeries1)
synSeries1.index=seriesIndex.index

synSeries2=list()
for i in range(0,len(synSeries1)):
    data =  y[i] +  synSeries1[i]
    synSeries2.append(data)

synSeries2=Series(synSeries2)
synSeries2.index=seriesIndex.index


series1=synSeries1
series1=read_csv('Correlation Datasets/covid  variazione positivi.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
#series1=read_csv('Correlation Datasets/covid - Copia.csv', usecols=[8],skiprows=285)
series1Original=series1




series2=synSeries2
series2=read_csv('Correlation Datasets/covid  variazione terap intensiva.csv',header=0, index_col=0, parse_dates=True, squeeze=True)
#series2=read_csv('Correlation Datasets/covid - Copia.csv', usecols=[17],squeeze=True,skiprows=285)
series2Original=series2




minLenght= min(len(series1),len(series2))

series1=series1.iloc[:minLenght]
series2=series2.iloc[:minLenght]



result=pearsonr(series1,series2)

print(result)

plt.figure()
plt.subplot(211)
series1.plot(color='blue',label='series1')
plt.legend()
plt.title("Series1 vs Series2  Pearson={}".format(result[0]))
plt.subplot(212)
series2.plot(color='green',label='series2')
plt.legend()
plt.show()




result1=Funzioni.StationarizeWithPSO(series1Original)
series1Trasf=result1[0]


series1Trasf.index=series1Original.index
series1Trasf=series1Trasf.drop(series1Trasf.index[0:result1[2]])

result2=Funzioni.StationarizeWithPSO(series2Original)
series2Trasf=result2[0]

series2Trasf=series2Trasf.drop(series2Trasf.index[0:result2[2]])

minLenghtTrasf= min(len(series1Trasf),len(series2Trasf))

series1Trasf=series1Trasf.iloc[:minLenghtTrasf]
series2Trasf=series2Trasf.iloc[:minLenghtTrasf]




resultTrasf=pearsonr(series1Trasf,series2Trasf)

print(resultTrasf)


plt.figure()
plt.subplot(211)
series1Trasf.plot(color='blue',label='series1Trasf')
plt.legend()
plt.title("TRASF Series1 vs Series2   Pearson={}".format(resultTrasf[0]))
plt.subplot(212)
series2Trasf.plot(color='green',label='series2Trasf')
plt.legend()
plt.show()


