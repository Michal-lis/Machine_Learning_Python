from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from scipy import misc
import io
import os
import pydotplus
import matplotlib.pyplot as plt
# graphviz for decision tree visualisation
import graphviz
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# setting parameters for rc
plt.rc("font", size=14)
#
import seaborn as sns

from sklearn import preprocessing

data_df = pd.read_csv("data.csv")
# problem with trees: overfitting!
# easy to interpret
# minimal data preparation
classifier = DecisionTreeClassifier(min_samples_split=40)

from sklearn.model_selection import train_test_split

features = ['tempo', 'acousticness', 'energy', 'danceability', 'acousticness', 'instrumentalness', 'liveness',
            'loudness', 'speechiness', 'valence', 'duration_ms', 'key']
train, test = train_test_split(data_df, test_size=0.2)
print('Training set size: {}. Test set size: {}'.format(len(train), len(test)))
X_train = train[features]
Y_train = train['target']

X_test = test[features]
Y_test = test['target']
dt = classifier.fit(X_train, Y_train)


def show_tree(tree, features, path):
    os.environ[
        "PATH"] += os.pathsep + 'C:/Users/Michu/AppData/Local/Programs/Python/Python36x64/Lib/site-packages/graphviz'
    f = io.StringIO()
    export_graphviz(tree, out_file=f, feature_names=features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img = misc.imread(path)
    plt.rcParam['figure.figsize'] = (20, 20)
    plt.imshow(img)


show_tree(dt, features, 'dec_tree.png')
