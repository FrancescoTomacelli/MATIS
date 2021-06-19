import Funzioni
from pandas import read_csv
from matplotlib import pyplot as plt
import time

series= read_csv('D:/Universitaa/TESI/tests/Datasets/series stock/mcDonald_2016_2019.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

start = time.time()
seriesTrasf2,seriesTrasf1,list_diff,list_lamb= Funzioni.ATSS(series)
end = time.time()
elapsed_time= end-start


series.plot(color='blue')
plt.show()

seriesTrasf2.plot(color='red')
plt.show()

print("list_diff", list_diff)
print("list_lamb",list_lamb)
print("elapsed time",elapsed_time)
print("lunghezza serie",len(series))

Funzioni.alarm()