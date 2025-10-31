import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the data 
from sklearn.datasets import load_diabetes

diabetes=load_diabetes()
print(diabetes)
print(diabetes.DESCR)

#PROBLEM STATEMENT- To predict disease progression after one year

print(diabetes.target)
print(diabetes.feature_names)

#To make it a dataset
df=pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
df['target']=diabetes.target
print(df)

#As we see that alll the attributes are scaled properly so for now , we'll skip EDA, Data_cleaning , Data_preprocessing , Feature_engineering
#For this part we'll use only one feature as I am practicing SLR

x=df[['bmi']]
y=df['target']

#Seperating train_test_split data 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

# test and train data seperation
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#If not done earlier we can do scaling here

#Now building model

from sklearn.linear_model import LinearRegression

#Object creation for the model which should be done for every model
model=LinearRegression()
print(model)

model.fit(x_train,y_train) #fit will help to train/understand the mathematical relationship.
 
# To know m i.e slope of eq. 
print(model.coef_)

# To know c i.e the intercept 
print(model.intercept_)

#Finally to get the prediction
y_predicted=model.predict(x_test)


#To visualize this model
plt.scatter(x_test,y_test,color='black',marker='*',label='Actual data')
plt.scatter(x_test,y_predicted,color='blue',linestyle='--',linewidths=3,label='Linear Regression Line')
plt.xlabel("BMI")
plt.ylabel("Progression")
plt.title("Diabetes data model")
plt.show()


