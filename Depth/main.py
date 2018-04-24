import pandas as pd
import quandl
import math
import numpy as np

from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

# quandl - 50 free counts a day
# google alphabet stock data
df = quandl.get('WIKI/GOOGL')
df = pd.DataFrame(df)

# simple linear regression is not going to work with the relationships between features
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]

# percentage change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# Volume- how many trades related happened that day
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'

# in machine learning you cannot work with NaN data, but generally a lot of data will be missing
# so lets replace it, 99999 will not influence the algorithm
df.fillna(-99999, inplace=True)

# for how many days in the future the prediction will be done, integer number
forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)

# .shift moves one series in dataframe up or down, negative moves series up (NaN in the last rows
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# features, df.drop return a dataframe without a given column
X = np.array(df.drop(['label'], 1))

# labels
y = np.array(df['label'])

X = preprocessing.scale(X)
# X = X[:-forecast_out + 1]
df.dropna(inplace=True)
y = np.array(df['label'])

# checking the consistent numbers of samples within the input variables
print(len(X), len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf1 = LinearRegression()
# fitting the classifier on training probes
clf1.fit(X_train, y_train)
# test the classifier
accuracy1 = clf1.score(X_test, y_test)

clf2 = svm.SVR()
# fitting the classifier on training probes
clf2.fit(X_train, y_train)
# test the classifier
accuracy2 = clf2.score(X_test, y_test)

print(accuracy1, accuracy2)

print(accuracy)
