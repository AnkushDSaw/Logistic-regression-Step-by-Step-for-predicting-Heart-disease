# Logistic-regression-Step-by-Step-for-predicting-Heart-disease

##  Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Importing dataset
dataset=pd.read_csv(r'C:\Users\ankus\OneDrive\Desktop\Naresh IT\April\28th,29th_April\framingham.csv')

# Variables
Response
* TenYearCHD: 10-year risk of coronary heart disease (1/0)

Explanatory
* male: 1: male, 0: female
* age: age of participant
* sysBP: systolic blood pressure
# Update some column name 
* male-> gender 1= male, 0-female
* TenYearCHD - HD = heart disease
dataset.rename(columns = {'male':'gender', 'TenYearCHD':'HD'}, inplace = True)

## Split the data
# We have to predict the HD column given the features.
X = dataset.drop(['HD'], axis = 1) # independent variable ( Remove mpg from X data)
y = dataset[['HD']] #dependent variable

# Taking care of Missing Values and Null values
X.isnull().sum()
# Column wise fill missing (nan) values using mean
X['education'].fillna(value=X['education'].mean(), inplace=True)
X.isnull().sum()
X['cigsPerDay'].fillna(value=X['cigsPerDay'].mean(), inplace=True)
X['BPMeds'].fillna(value=X['BPMeds'].mean(), inplace=True)
X['totChol'].fillna(value=X['totChol'].mean(), inplace=True)
X['BMI'].fillna(value=X['BMI'].mean(), inplace=True)
X['heartRate'].fillna(value=X['heartRate'].mean(), inplace=True)
X['glucose'].fillna(value=X['glucose'].mean(), inplace=True)
X.isnull().sum()

# Splitting Dataset- Xtrain and y Train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2,random_state=0)

# Feature Scaling for Improving model Performance
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Model buidling with logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

# Predicting test set results
y_pred = classifier.predict(x_test)

# Evaluating Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# Accuracy of Model
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

# Bias and Vriance
bias= classifier.score(x_train,y_train)
variance=classifier.score(x_test,y_test)
print(bias)
print(variance)












