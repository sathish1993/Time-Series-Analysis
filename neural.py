import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

def make_forecast(model, look_back_buffer, timesteps):
	forecast_predict =  np.empty((0, 1), dtype=np.float32)
    	for i in range(timesteps):
        	cur_predict = model.predict(look_back_buffer)
        	forecast_predict = np.concatenate((forecast_predict, cur_predict))
		look_back_buffer = np.delete(look_back_buffer, 0, 1)
        	look_back_buffer = np.concatenate([look_back_buffer, cur_predict],axis=1)
	return forecast_predict

np.random.seed(7)

#Read from csv
cols = pd.read_csv('transpose.csv',nrows=119).columns
df = pd.read_csv('transpose.csv',nrows=119, usecols=cols[1:])

rowsum = df.sum(axis=1).astype(float)
sarray = np.array(rowsum)

train_size = 100
test_size = len(sarray) - train_size

train, test = sarray[0:train_size], sarray[train_size:len(sarray)]


look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


model = Sequential()
model.add(Dense(20, input_dim=look_back, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
result = model.fit(trainX, trainY, epochs=400, batch_size = 5, verbose = 2)

trainScore = model.evaluate(trainX, trainY, verbose = 0)
print 'Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore))
testScore = model.evaluate(testX, testY, verbose = 0)
print 'Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore))

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

forecast_predict = make_forecast(model, testX[-1::], timesteps=28)
print forecast_predict

trainPredictPlot = np.empty_like(sarray)
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict.reshape(-1)


testPredictPlot = np.empty_like(sarray)
testPredictPlot[:] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(sarray)-1] = testPredict.reshape(-1)

new = np.concatenate((trainPredict,testPredict,forecast_predict))
plt.plot(sarray)
plt.plot(trainPredict)
plt.plot(testPredict)
plt.plot(new)
#plt.plot(forecast_predict)
#plt.plot([None for _ in range(look_back)] + [x for x in trainPredictPlot])
#plt.plot([None for _ in range(look_back)] + [None for _ in trainPredictPlot] + [x for x in testPredictPlot])
#plt.plot([None for _ in range(look_back)] + [None for _ in trainPredictPlot] + [None for _ in testPredictPlot] + [x for x in forecast_predict])
plt.show()

