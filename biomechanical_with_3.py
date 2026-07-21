# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:39:39 2026

@author: osemi
"""
#%% Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#%% Create Dataframe and Labeling
df = pd.read_csv("column_3C_weka.csv")

df['class'] = df['class'].str.strip()
    
mapping = {
    "Normal": 0,
    "Hernia": 1,
    "Spondylolisthesis": 2}

df['class'] =   df['class'].map(mapping)

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#%% Model Fitting

rf = RandomForestClassifier(n_estimators = 500, random_state = 42)
rf.fit(X_train, y_train)

#%% Prediction

y_head = rf.predict(X_test)
print("3-Class Model Accuracy Score: ", accuracy_score(y_test, y_head))

#%% Confusion Matrix & Visualization
cm = confusion_matrix(y_test, y_head)

plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot = True, fmt = "d", cmap = "Greens",
            xticklabels = ['Normal (0)', 'Hernia (1)', 'Spondylolisthesis(2)'],
            yticklabels = ['Normal (0)', 'Hernia (1)', 'Spondylolisthesis(2)'])
plt.xlabel('Predicted Class')
plt.ylabel('Real Class')
plt.title("Random Forest (3-Class) - Confusion Matrix")
plt.show()