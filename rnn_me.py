# Recurrent Neural Network
#We want to check if our predictions follow
# the same directions as the real stock price 
#and we don’t really care whether our 
#predictions are close the real stock price. 


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values #1:2 returns an array not a vector

# Feature Scaling(standard(divide by std)/normal(divide my max-min)--use when sigmoid fn is used)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    #X_TRAIN WILL CONTAIN INFORMATION OF 60 CONSECUTIVE DAYS AND y_TRAIN WILL CONTAIN STOCK PRICE AT T+1 
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
#CONVERTING LISTS TO NUMPY ARRAY
X_train, y_train = np.array(X_train), np.array(y_train)




#------------------------------
#RESHAPING 3D-to add a new dimension
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
#                          ( batch_size,timesteps,indicators)
#-------------------------------


#BUILDING THE RNN
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout

#INITIALIZE
regressor=Sequential()
#Adding the first lstm layer and dropout regularisation

regressor.add(LSTM(units=50, return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

#Adding the second lstm layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#Adding the third lstm layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#Adding the forth lstm layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units=1))

#Compiling
regressor.compile(optimizer='adam',loss='mean_squared_error')

#Fitting the RNN to training set
regressor.fit(X_train,y_train,epochs=100,batch_size=32)


#MAKE PREDICTIONS

#getting stock price of 2017
# Importing the test set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)



# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


'''
here are different ways to improve the RNN model:

    Getting more training data: we trained our model on the past 5 years of the Google Stock Price but it would be even better to train it on the past 10 years.
    Increasing the number of timesteps: the model remembered the stock prices from the 60 previous financial days to predict the stock price of the next day. That’s because we chose a number of 60 timesteps (3 months). You could try to increase the number of timesteps, by choosing for example 120 timesteps (6 months).
    Adding some other indicators: if you have the financial instinct that the stock price of some other companies might be correlated to the one of Google, you could add this other stock price as a new indicator in the training data.
    Adding more LSTM layers: we built a RNN with four LSTM layers but you could try with even more.
    Adding more neurones in the LSTM layers: we highlighted the fact that we needed a high number of neurones in the LSTM layers to respond better to the complexity of the problem and we chose to include 50 neurones in each of our 4 LSTM layers. You could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.


'''


'''
PARAMETER TUNING-GRID SEARCH
scoring = 'accuracy' ----- CLASSIFICATION
scoring = 'neg_mean_squared_error' ------ REGRESSION 

'''




#Parameter tuning
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_regressor(optimizer):
    regressor = Sequential()
    regressor.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu', input_shape=(X_train.shape[1],1)))
    regressor.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))
    regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    regressor.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return regressor
regressor = KerasRegressor(build_fn = baseline_model)
parameters={'batch_size':[25,32,64],
            'epochs':[100,500],
            'optimizer':['adam','rnsprop']}
grid_search= GridSearchCV(estimator=regressor,
                          param_grid=parameters,
                          scoring='neg_mean_squared_error',
                          cv=10)
grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_




