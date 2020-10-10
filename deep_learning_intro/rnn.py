# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:30:38 2020

@author: BastOS
"""

# Partie 1 - Préparation des données
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Jeu d'entrainement
dataset_train = pd.read_csv('data/rnn_dataset/Google_Stock_Price_Train.csv')
training_set = dataset_train[['Open']].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))
training_set_sc = sc.fit_transform(training_set)

# Création de la structure avec 60 timesteps et 1 sortie
X_train = [] 
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_sc[i-60:i, 0])
    y_train.append(training_set_sc[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))

# Partie 2 - Construction du RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialisation
regressor = Sequential()

# Couche LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# 2e Couche LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# 3e Couche LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# 4e Couche LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))


# Couche de sortie
regressor.add(Dense(units=1))

# Compilation
regressor.compile(optimizer="adam", loss="mean_squared_error")

# Entrainement Tester sur un nombre plus petit d epoch (regarder loss) puis augmenter ..
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Partie 3 - Prédictions et visualisation

# Données de 2017
# Jeu d'entrainement
dataset_test = pd.read_csv('data/rnn_dataset/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test[['Open']].values

# Prédictions pour 2017
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0)

inputs = dataset_total[len(dataset_total)- len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)


X_test = [] 


for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)

# Reshaping
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualisation des résultats
plt.plot(real_stock_price, color="red", label="Prix réel de l'action Google")
plt.plot(predicted_stock_price, color="green", label="Prix prédit de l'action Google")
plt.title("Prédiction de l'action Google")
plt.xlabel("Jour")
plt.ylabel("Prix de l'action")
plt.legend()
plt.show()

# Ajustement des hyperparametres (Optimisation avec GridSearch)

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

def build_regressor(optimizer):
    # Initialisation
    regressor = Sequential()
    
    # Couche LSTM + Dropout
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    # 2e Couche LSTM + Dropout
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    # 3e Couche LSTM + Dropout
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    # 4e Couche LSTM + Dropout
    regressor.add(LSTM(units=50, return_sequences=False))
    regressor.add(Dropout(0.2))


    # Couche de sortie
    regressor.add(Dense(units=1))

    # Compilation
    regressor.compile(optimizer=optimizer, loss="mean_squared_error")
    
    return regressor

regressor_keras = KerasRegressor(build_fn=build_regressor)
parameters = {"batch_size" : [25, 32],
              "epochs" : [100, 200, 300],
              "optimizer" : ["adam"]}

grid_search = GridSearchCV(regressor_keras, param_grid=parameters, scoring="neg_mean_squared_error", cv=10)

grid_search.fit(X_train, y_train)


grid_search.best_params_
grid_search.best_score_

grid_predicted_stock_price = grid_search.predict(X_test)
grid_predicted_stock_price = sc.inverse_transform(grid_predicted_stock_price)

# Visualisation des résultats
plt.plot(real_stock_price, color="red", label="Prix réel de l'action Google")
plt.plot(predicted_stock_price, color="green", label="Prix prédit de l'action Google")
plt.plot(grid_predicted_stock_price, color="blue", label="Prix prédit de l'action Google GridCV")
plt.title("Prédiction de l'action Google")
plt.xlabel("Jour")
plt.ylabel("Prix de l'action")
plt.legend()
plt.show()