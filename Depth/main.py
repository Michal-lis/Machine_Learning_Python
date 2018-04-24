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
# 99999 will not influence the algorithm
df.fillna(-99999, inplace=True)

# for how many days in the future the prediction will be done
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())

# features
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
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
