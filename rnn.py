# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:24:39 2020

@author: BastOS
"""

# Classification d'images Chien ou chat

# Partie 1 - Construction du CNN

# Importation des modules
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Initialiser le CNN
classifier = Sequential()
 
 
# Etape 1: La couche de Convolution
# Transformation de l'image, création d'une matrice d'entrée qui, pour chaque pixel de l'image
# stocke une valeur entre 0 et 255
# Création des filtres, 'feature detector', en spécifiant la taille ainsi que le déplacement du filtre sur la matrice d'entrée
# On obtiendra ainsi une feature map
# On repete plusieurs fois l'operation avec des filtres differents ce qui formera la couche de convolution
# Fonction relu pou ajouter de la non linearité dans le modele (pour casser la linéarité, remplace toutes les valeurs négatives par des 0)
classifier.add(Convolution2D(filters=32, kernel_size=[3, 3], strides=1, 
                             input_shape=(64, 64, 3), activation="relu"))


# Etape 2: La couche de Pooling
# Simplifier le modele, sans perdre l'information, on conserve les valeurs élévées (celles qui permettent de reperer les features)
# Permet aussi l'invariance spatiale (identifier les features similaires, meme si elles sont legerement differentes sur chaque photo (ex: larme du guépard, image deformée etc..))
# Reduit l'overfitting, on retire de l'information qui n'est pas importante, sur laquelle on risquerait d overfitter
# On garde uniquement l'information importante
classifier.add(MaxPooling2D(pool_size=(2,2), strides=2))


# Etape 3: La couce de Flattening
# Features maps Pooling dans un grand vecteur vertical
classifier.add(Flatten())


# Etape 4: Ajout d'un ANN completement connecté
classifier.add(Dense(units=128, activation="relu"))

# On souhaite une probabilité de sortie, ici 1 sortie donc sigmoid
classifier.add(Dense(units=1, activation="sigmoid"))


# Compilation
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



# Entrainer le CNN sur nos images
# Image augmentation: préparation de nos images pour éviter le surapprentissage
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'data/rnn_dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'data/rnn_dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=250,
        epochs=50,
        validation_data=test_set,
        validation_steps=63,
        verbose=1)