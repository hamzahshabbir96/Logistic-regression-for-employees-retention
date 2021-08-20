# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 12:50:50 2021

@author: Hamzah
"""

import os
os.getcwd()
from django.conf import settings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file_path= os.path.join(settings.BASE_DIR,'dataset.csv')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

poke_data = pd.read_csv(file_path)

poke_data = pd.DataFrame(poke_data)
#poke_data.head()
#poke_data.isnull().sum()

poke_data['Pr_Male'].fillna(0.500, inplace=True)
poke_data = poke_data.drop(['Type_2', 'Egg_Group_2'], axis=1)

#plt.figure(figsize=(15,15))
#sns.heatmap(poke_data.corr(),annot=True,cmap='viridis',linewidths=.5)

poke_data = poke_data.replace(['Water', 'Ice'], 'Water')
poke_data = poke_data.replace(['Grass', 'Bug'], 'Grass')
poke_data = poke_data.replace(['Ground', 'Rock'], 'Rock')
poke_data = poke_data.replace(['Psychic', 'Dark', 'Ghost', 'Fairy'], 'Dark')
poke_data = poke_data.replace(['Electric', 'Steel'], 'Electric')

poke_data['Body_Style'].value_counts()


#convert series to dict
ref_dict= poke_data['Body_Style'].value_counts().to_dict()

types_poke = pd.get_dummies(poke_data['Type_1'])

color_poke = pd.get_dummies(poke_data['Color'])


X = pd.concat([poke_data, types_poke], axis=1)
X = pd.concat([X, color_poke], axis=1)

#X.head()
X_transform = X.drop(['Number', 'Name', 'Type_1', 'Color', 'Egg_Group_1'], axis = 1)
#X_transform.shape

y = X_transform['isLegendary']
X_final = X_transform.drop(['isLegendary', 'Body_Style'], axis = 1)

Xtrain, Xtest, ytrain, ytest = train_test_split(X_final, y, test_size=0.2)
random_model = RandomForestClassifier(n_estimators=500, random_state = 42)

model_final = random_model.fit(Xtrain, ytrain)

y_pred = model_final.predict(Xtest)

#Checking the accuracy
random_model_accuracy = round(model_final.score(Xtrain, ytrain)*100,2)
#print(round(random_model_accuracy, 2), '%')

random_model_accuracy1 = round(random_model.score(Xtest, ytest)*100,2)
#print(round(random_model_accuracy1, 2), '%')

Ytest = np.array(ytest)
count = 0
for i in range(len(ytest)):
  if Ytest[i] == y_pred[i]:
    count = count + 1

print((count/len(ytest))*100)

#import pickle
#filename = 'pokemon_model.pickle'
#pickle.dump(model_final, open(filename, 'wb'))


