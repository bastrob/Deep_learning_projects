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



# Partie 2 - Construction du RNN


# Partie 3 - Prédictions et visualisation