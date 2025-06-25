import numpy as np
import pandas as pd

#Importing the data set
dataset = pd.read_csv('Data.csv')
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values
print(x)
print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print(x)

#Encoding categorical data
#1. Encoding the independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x= np.array(ct.fit_transform(x))
# 2. Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
y= le.fit_transform(y)
print(y)

#Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train)
print(y_train)
print(x_test)
print(y_test)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)
print(x_train)
print(x_test)

