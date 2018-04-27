"""
based on: https://www.kaggle.com/bsivavenu/house-price-calculation-methods-for-beginners
credits: bsivavenu

Skills:
Creative feature engineering 
Advanced regression techniques like random forest and gradient boosting

Marking:
Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_sp = df_train['SalePrice']

# print the general info about the predcited value
print(df_sp.describe())

# visualise it
sns.distplot(df_sp)
# plt.show()

# skewness and kurtosis
print("Skewness: %f" % df_sp.skew())
print("Kurtosis: %f" % df_sp.kurt())
