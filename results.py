import numpy as np
import pandas as pd
df = pd.read_csv(r"C:\Users\kapoor\Desktop\winequality-red.csv", sep=';')

X = df[list(df.columns)[:-1]]
y = df['quality']
X = np.append(arr = np.ones(X.shape[0],1), values = X, axis = 1)
X_opt = X[:, [0,1,2,4,6,8,9,10,11]]

from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X_opt,y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
predictions = regressor.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test,predictions)

X = np.append(arr = np.ones(X.shape[0],1), values = X, axis = 1)

import statsmodels.formula.api as sm
X_opt = X[:, [0,1,2,4,6,8,9,10,11]]
regressor_OLS = sm.OLS(endog = y, exog = X.opt, ).fit()
regressor_OLS.summary()

#displaying results
import matplotlib.pylab as plb
plt.scatter(y_test, predictions, c = 'g')
plt.xlabel('true quality')
plt.ylabel('predicted quality')
plt.title('predicted quality against true quality')
plt.show()


