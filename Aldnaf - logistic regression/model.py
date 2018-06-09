import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.rc("font", size=14)

import seaborn as sns

sns.set(style="white")  # white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)

from sklearn.linear_model import LogisticRegression
from  sklearn.feature_selection import RFE