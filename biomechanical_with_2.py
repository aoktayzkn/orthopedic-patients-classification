# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:58:40 2026

@author: osemi
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("column_2C_weka.csv")

df['class'] = [1 if each == "Abnormal" else 0 for each in df['class']]

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#%% Sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#%% Fit & Predict
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 500, random_state = 42)
rf.fit(X_train, y_train)

y_head = rf.predict(X_test)

from sklearn.metrics import accuracy_score
print("Model Accuracy Score: ", accuracy_score(y_test, y_head))

#%% Confusion Matrix & Visualization
from sklearn.metrics import confusion_matrix
import seaborn as sns # Grafiği daha güzel çizdirmek için

cm = confusion_matrix(y_test, y_head)
print("Konfüzyon Matrisi:\n", cm)

# Visualization
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Normal (0)', 'Abnormal (1)'], 
            yticklabels=['Normal (0)', 'Abnormal (1)'])
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Random Forest - Konfüzyon Matrisi"')
plt.show()