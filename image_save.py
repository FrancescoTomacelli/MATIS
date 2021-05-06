from PIL import Image
from pandas import read_csv
from matplotlib import pyplot
from PIL import Image, ImageDraw

global counter_photo
counter_photo=0

series = read_csv('Datasets/Airline_Passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

image=pyplot.figure()
pyplot.subplot(211)
series.plot(color='blue', label='Original')
pyplot.legend()
pyplot.title('Transformation Applied  Particle={}  lambda={:.2f}'.format(4,5))
pyplot.subplot(212)
series.plot(color='green', label='Trasformed')
pyplot.legend()



pyplot.savefig('D:/Universitaa/TESI/tests/immagini/Syn_12_7_30_'+ str(counter_photo)+'_.png')
counter_photo=counter_photo+1


pyplot.show()



img = Image.new('RGB', (500,500), color=(0,0,0))

d = ImageDraw.Draw(img)
d.text((50, 100), "Hello World", features=100)
d.text((50, 200), "Hello ", fill=(255, 255, 200))
d.text((50, 300), "Hello oooooo", fill=(255, 255, 200))
d.text((50, 400), "Hello Worldaaaaaaaaaaaaaaaaaaaa", fill=(255, 255, 200))
img.save('D:/Universitaa/TESI/tests/immagini/Syn_12_7_30'+ str(counter_photo)+'.png')