import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#reading the data
data = pd.read_csv(r"C:\Users\kapoor\Desktop\winequality-red.csv", sep=';')
X= data.iloc[:, :-1]
y= data.iloc[:, -1]


#addingextra column
X = np.append(arr = np.ones(X.shape[0],1), values=X, axis=1)

#splitting the data
X_train, X_test,y_train, y_test = train_test_split(X,y)

#scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


#linear regression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
predictions = regressor.predict(X_test)








