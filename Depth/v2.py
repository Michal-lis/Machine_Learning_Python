import pandas as pd
import quandl
import math
import numpy as np
import datetime
import matplotlib.pyplot as plt

from matplotlib import style
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

style.use('ggplot')

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
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
df.dropna(inplace=True)

# labels
y = np.array(df['label'])[:-forecast_out]

# checking the consistent numbers of samples within the input variables
print(len(X), len(y))

if len(X) != len(y):
    exit()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# n_jobs specifies the threads
clf = LinearRegression(n_jobs=10)
# fitting the classifier on training probes
clf.fit(X_train, y_train)
# test the classifier
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
