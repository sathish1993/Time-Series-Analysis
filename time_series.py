import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA

#Read from csv
cols = pd.read_csv('product_distribution_training_set_transpose.csv',nrows=119).columns
df = pd.read_csv('product_distribution_training_set_transpose.csv',nrows=119, usecols=cols[1:])

columns = df.columns

print columns
#Predicting Overall Sale Quantity -- Start

print "Reading data...\n"
rowsum = df.sum(axis=1).astype(float)
sarray = np.array(rowsum)

print "Working on the obtained data...\n"
ts = pd.Series(sarray)
copyts = pd.Series(sarray)
#copyts.plot(color='red')

print "ARIMA Model is chosen...\n"
#ARIMA
ts.index = pd.to_datetime(ts.index,unit='D')
model_ARIMA = ARIMA(ts, order=(5,1,0))
results_ARIMA = model_ARIMA.fit(disp=0)

print "Forecasting future results for overall sale... \n"	
#forecast out of sample
forecast = results_ARIMA.forecast(steps=29)[0]
forecast[forecast<0] = 0
forecast = forecast.round()

print "Storing forecast results for overall sale in Output.txt... \n"
#storing forecast results in output.txt
fileObject = open("Output.txt", "w")
fileObject.write('0 ')
for data in forecast:
	fileObject.write('%d ' % data)
fileObject.write('\n \n')
 
farray = np.concatenate((sarray,forecast))
newts = pd.Series(farray)
newts.plot(color='blue')
#pyplot.title('Overall Sale Quantity for 146 days')
#pyplot.xlabel('Time')
#pyplot.ylabel('Overall Sale Quantity')
#pyplot.plot(x="time",y=[newts])
#pyplot.show()
sarray.fill(0)
farray.fill(0)

#Predicting Overall Sale Quantity -- End

#Predicting Individual Sale Quantity -- Start
print "Now predicting for 100 Key Products...\n"
outputarray = []
for i in range(0,100):
	print "Working on Key Product %d... \n" % int(columns[i])
	sarray = df[[i]].values.reshape(-1).astype(float)
	ts = pd.Series(sarray)
	copyts = pd.Series(sarray)
	#copyts.plot(color='yellow')
	#pyplot.show()
	#ARIMA
	ts.index = pd.to_datetime(ts.index,unit='D')
	model_ARIMA = ARIMA(ts, order=(5,1,0))
	results_ARIMA = model_ARIMA.fit(disp = 0)

	#forecast out of sample
	forecast = results_ARIMA.forecast(steps=29)[0]
	forecast[forecast<0] = 0
	forecast = forecast.round()
	
	#storing forecast results in output.txt
	print "Storing forecast result of Key Product id %d in Output.txt...\n" % int(columns[i])
	fileObject.write('%d ' % int(columns[i]))
	for data in forecast:
		fileObject.write('%d ' % data)
	fileObject.write('\n\n')
	farray = np.concatenate((sarray,forecast))
	newts = pd.Series(farray)
	newts.plot(color='blue')
	#pyplot.title('Overall Sale Quantity for 146 days for Product id %d '% int(columns[i]) )
	#pyplot.xlabel('Time')
	#pyplot.ylabel('Overall Sale Quantity')
	#pyplot.plot(x="time",y=[newts])
	#pyplot.show()
	if i == 0:
		outputarray = forecast
	else:
		outputarray = [outputarray,forecast]
	
	sarray.fill(0)
	farray.fill(0)

print "Please Check Output.txt for all the forecast results...\n"
#Predicting Individual Sale Quantity -- End

fileObject.close();
