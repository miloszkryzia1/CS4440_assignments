# -------------------------------------------------------------------------
# AUTHOR: Milosz Kryzia
# FILENAME: roc_curve.py
# SPECIFICATION: this program constructs a roc curve to compare the performance of a decision tree classifier against a random classifier
# FOR: CS 4440 (Data Mining) - Assignment #3
# TIME SPENT: 20 minutes
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

# importing some Python libraries
from sklearn import tree
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

# read the dataset cheat_data.csv and prepare the data_training numpy array
# --> add your Python code here
data_training = np.array(pd.read_csv(
    'cheat_data.csv', sep=',').values)

# transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
# be converted to a float.
# --> add your Python code here
X = data_training[:, :3]
for i in range(2):
    X[:, i] = LabelEncoder().fit_transform(X[:, i])

X[:, 2] = np.array([float(i[:-1])*1000 for i in X[:, 2]])

# transform the original training classes to numbers and add them to the vector y. For instance Yes = 1, No = 0, so Y = [1, 1, 0, 0, ...]
# --> add your Python code here
y = data_training[:, 3]
y = LabelEncoder().fit_transform(y)

# split into train/test sets using 30% for test
# --> add your Python code here
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3)

# generate random thresholds for a no-skill prediction (random classifier)
# --> add your Python code here
ns_probs = np.random.rand(len(testy))

# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)

print(dt_probs)

# keep probabilities for the positive outcome only
# --> add your Python code here
dt_probs = dt_probs[:, 1]

# calculate scores by using both classifiers (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()
