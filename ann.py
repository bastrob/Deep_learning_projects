# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:24:39 2020

@author: BastOS
"""

# Partie 1 : Préparation des données


# Import des librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import du dataset
dataset = pd.read_csv("data/Churn_Modelling.csv")


# Le jeu de données est propre, pas de nettoyage necessaire


# Selection des features
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoder les variables catégoriques
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Country column
ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)

# Male/Female
labelencoder_X = LabelEncoder()
X[:,4] = labelencoder_X.fit_transform(X[:, 4])
X = X[:, 1:]


# Split dataset train / test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Feature Scaling
# Permet de rendre les calculs bcp plus rapides (proche de 0)
# Reduire l ordre de grandeur des variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Partie 2 - Construire le réseau de neurones


# Importation des modules de keras 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# ETAPE 1: Initialiser les poids avec des valeurs proches du O (mais différentes de 0)
# ETAPE 2: Envoyer la première observation dans la couche d'entrée, avec une variable par neurone
# ETAPE 3: Propagation avant: Les neurones sont activés d'une manière dépendant des
# poids qui leur sont attribués. Propager jusqu'a' obtenir la prédiction y.
# ETAPE 4: Comparer la prédiction avec la vraie valeur et mesurer l'erreur avec la fonction de côut
# ETAPE 5: Propagation arrière: L'erreur se re-propage dans le réseau. Mettre à jour les poids selon leur responsabilité dans l'erreur.
# Le taux d'apprentissage détermine de combien on ajuste les poids.
# ETAPE 6: Répéter étapes 1 à 5 et ajuster les poids après chaque observation (apprentissage renforcé - Gradient Stochastique)
# Répéter étapes 1 à 5 et ajuster les poinds après un lot d'observations (apprentissage par lot - Gradient)
# ETAPE 7: Quand tout le jeu de données est passé à travers l'ANN, ca fait une époque.
# Refaire plus d'époques




# Initialiser un reseau de neurones
classifier = Sequential()


  
# 1: Fonction Dense qui s'occupe d'initialiser les poids (proche de 0)
# 2: Connaitre le nombre de variable d'entrée pour connaitre le nombre de neurone sur la couche d'entrée
# 3: Fonction d'activation
# 4: Fonction de coût
# 5: Back propagation
# 6: Apprentissage par lot : sur tous les individus d'un coup 1 seule back propagation 1 seul ajustement des poids à la fin du lot
# Apprentissage par individus : 1 back propagation et ajustement des poids apres chaque individus
# Apprentissage par mini-lot : 1 back propagation et ajustement des poids apres le mini-lot


# Ajouter la couche d'entrée et une couche cachée
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11))
classifier.add(Dropout(rate=0.1))

# Ajouter une deuxieme couche cachée (pas forcement necessaire, juste pour l'aspect pratique)
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.1))

# Ajouter la couche de sortie
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

# Compiler le réseau de neurones, ajout des parametres pour utiliser la descente de gradient stochastique
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# Entrainer le réseau de neurones
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


#################################################################
# Test cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11))
    
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
    
    classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
    
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    return classifier

classifier_keras = KerasClassifier(build_fn=build_classifier,
                                   batch_size=10,
                                   epochs=100)

score = cross_val_score(estimator=classifier_keras, X=X_train, y=y_train, cv=10)
print("score: ", score, "- Moyenne: ", score.mean(), " - Std: ", score.std())



from sklearn.model_selection import StratifiedKFold
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cvscores = []
for train, test in kfold.split(X_train, y_train):
  # create model
	model = Sequential()
	model.add(Dense(6, input_dim=11, activation='relu', kernel_initializer="uniform"))
	model.add(Dense(6, activation='relu', kernel_initializer="uniform"))
	model.add(Dense(1, activation='sigmoid', kernel_initializer="uniform"))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(X_train[train], y_train[train], epochs=100, batch_size=10, verbose=1)
	# evaluate the model
	scores = model.evaluate(X_train[test], y_train[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# Deux methodes : cross validation manuelle ou creer un wrapper de keras pour pouvoir utiliser les fonctionnalités de sklearn
# Fin test cross validation
#################################################################





# Prediction sur le jeu de test
y_pred = classifier.predict(X_test)

# Pour utiliser une matrice de confusion il est necessaire d'utiliser des classes
# ici classe 0: le client va rester, classe 1: le client va partir
# On utilise donc un seuil standard (50%) pour répartir les individus d'apres la prediction
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Acc
(1515 + 189)/(1515 + 57 + 239  +189)

# TP
(189) / ((189 + 57))

from sklearn.metrics import precision_score
precision_score(y_test, y_pred)

new_value = dict()
new_value = {
    "0" : 600,
    "1" : "France",
    "2" : "Male",
    "3" : 40,
    "4" : 3,
    "5" : 60000,
    "6" : 2,
    "7" : 1,
    "8" : 1,
    "9" : 50000
    }
value = pd.DataFrame(new_value, index=[0])

value = ct.transform(value)
value[:, 4] = labelencoder_X.transform(value[:, 4])
value = value[:, 1:]

value = sc.transform(value)

y_pred_new_value = classifier.predict(value)

new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

new_prediction_class = (new_prediction > 0.5)


# Ajustement des hyperparametres (Optimisation avec GridSearch)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11))
    
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
    
    classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
    
    classifier.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    
    return classifier

classifier_keras = KerasClassifier(build_fn=build_classifier)
parameters = {"batch_size" : [25, 32],
              "epochs" : [100, 500],
              "optimizer" : ["adam", "rmsprop"]}

grid_search = GridSearchCV(classifier_keras, param_grid=parameters, scoring="accuracy", cv=10)

y_pred_grid = grid_search.fit(X_train, y_train)

grid_search.best_params_
grid_search.best_score_

