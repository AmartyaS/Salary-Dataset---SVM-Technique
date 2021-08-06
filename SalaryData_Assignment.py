# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 01:46:51 2021

@author: ASUS
"""

#Importing all the Necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#Loading the dataset
train=pd.read_csv("D:\Data Science Assignments\Python-Assignment\SVM\SalaryData_Train(1).csv")
test=pd.read_csv("D:\Data Science Assignments\Python-Assignment\SVM\SalaryData_Test(1).csv")

#Data Exploration and Manipulation
train.head()
train.describe()
train.dtypes
train.columns
train['sex'].unique()
#Applied One Hot En-coding technique
train=pd.get_dummies(train,columns=["workclass"],prefix=["WC"])
test=pd.get_dummies(test,columns=["workclass"],prefix=["WC"])
train=pd.get_dummies(train, prefix=['ED'],columns=["education"])
test=pd.get_dummies(test,columns=["education"],prefix=["ED"])
train=pd.get_dummies(train, prefix=['Stat'],columns=["maritalstatus"])
test=pd.get_dummies(test,columns=["maritalstatus"],prefix=["Stat"])
train=pd.get_dummies(train, prefix=['Job'],columns=["occupation"])
test=pd.get_dummies(test,columns=["occupation"],prefix=["Job"])
train=pd.get_dummies(train, prefix=['Rel'],columns=["relationship"])
test=pd.get_dummies(test,columns=["relationship"],prefix=["Rel"])
train=pd.get_dummies(train, prefix=['Race'],columns=["race"])
test=pd.get_dummies(test,columns=["race"],prefix=["Race"])
train=pd.get_dummies(train, prefix=['Native'],columns=["native"])
test=pd.get_dummies(test,columns=["native"],prefix=["native"])
train['sex']=[0 if x==' Female' else 1 for x in train['sex']]
test['sex']=[0 if x==' Female' else 1 for x in test['sex']]
#Normalization function
def norm(i):
    x=((i-i.min())/(i.max()-i.min()))
    return x

#Normalizing the data and splitting into training and testing dataset
x_train=norm(train[train.columns.difference(["Salary"])])
x_test=norm(test[test.columns.difference(["Salary"])])
y_train=pd.DataFrame(train['Salary'])
y_test=pd.DataFrame(test['Salary'])


#SVM Model, kernel = Linear
model1=SVC(kernel='linear',random_state=4)
model1.fit(x_train,y_train)
pred1=model1.predict(x_test) #Predicting the values
#Checking accuracy of the model
np.mean(y_test.values.flatten()==pred1)
pd.crosstab(y_test.values.flatten(),pred1)

#SVM Model, kernel = Radial
model2=SVC(kernel='rbf',random_state=4)
model2.fit(x_train,y_train)
pred2=model2.predict(x_test)#Predicting the values
#Checking accuracy of the model
np.mean(y_test.values.flatten()==pred2)
pd.crosstab(y_test.values.flatten(),pred2)

#SVM Model, kernel = Polynomial
model3=SVC(kernel='poly',degree=1)
model3.fit(x_train,y_train)
pred3=model3.predict(x_test)#Predicting the values
#Checking accuracy of the model
np.mean(y_test.values.flatten()==pred3)
pd.crosstab(y_test.values.flatten(),pred3)
