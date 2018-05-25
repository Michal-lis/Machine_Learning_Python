import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# setting parameters for rc
plt.rc("font", size=14)

import seaborn as sns

from sklearn import preprocessing

sns.set(style='white')  # white beackground for seaborn plots
sns.set(style='whitegrid', color_codes=True)

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df.head())
all_data = train_df.shape[0] + test_df.shape[0]
print('Train data is {} samples, {}% of all.'.format(train_df.shape[0], round(train_df.shape[0] / all_data * 100, 2)))
print('Test data is {} samples, {}% of all.'.format(test_df.shape[0], round(test_df.shape[0] / all_data * 100, 2)))

# DATA QUALITY
# checking how many missing values do we have
print(train_df.isnull().sum())

# AGE
# We can see that there are 177 Age missing, 687 Cabin missing and 2 Embarked missing
print('Percent of missing "Age" records is %.2f%%' % ((train_df['Age'].isnull().sum() / train_df.shape[0]) * 100))
# almost 20% of age data missing, lets visualise the age parameter
# from visualisation we can see than the histogram is skewed, so we will not use the mean but the average value of age
# mean age
print('The mean of "Age" is %.2f' % (train_df["Age"].mean(skipna=True)))
# median age
print('The median of "Age" is %.2f' % (train_df["Age"].median(skipna=True)))

# Cabin
print('Percent of missing "Cabin" records is %.2f%%' % ((train_df['Cabin'].isnull().sum() / train_df.shape[0]) * 100))
# since 77% of Cabin data is missing, we will omit this information in our model
