# -------------------------------------------------------------------------
# AUTHOR: Milosz Kryzia
# FILENAME: decision_tree.py
# SPECIFICATION: decision tree classifier implementation using sklearn
# FOR: CS 4440 (Data Mining) - Assignment #3
# TIME SPENT: 1.5 hours
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

# importing some Python libraries
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv',
            'cheat_training_2.csv', 'cheat_training_3.csv']

for ds in dataSets:

    X = []
    Y = []

    # reading a dataset eliminating the header (Pandas library)
    df = pd.read_csv(ds, sep=',', header=0)
    # creating a training matrix without the id (NumPy library)
    data_training = np.array(df.values)[:, 1:]

    # transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    # Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    # be converted to a float.
    X = data_training[:, :3]
    for i in range(2):
        X[:, i] = LabelEncoder().fit_transform(X[:, i])

    X[:, 2] = np.array([float(i[:-1])*1000 for i in X[:, 2]])

    # transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    # --> add your Python code here
    Y = data_training[:, 3]
    Y = LabelEncoder().fit_transform(Y)

    # loop your training and test tasks 10 times here
    model_accuracies = []
    for i in range(10):

        # fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
        clf = clf.fit(X, Y)

        # plotting the decision tree
        # tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=[
        #                'Yes', 'No'], filled=True, rounded=True)
        # plt.show()

        # read the test data and add this data to data_test NumPy
        # --> add your Python code here
        df_test = pd.read_csv('cheat_test.csv', sep=',', header=0)
        data_test = np.array(df_test.values)[:, 1:]

        encoders = [LabelEncoder() for _ in range(3)]

        accurate_predictions = 0
        model_accuracy = 0

        for data in data_test:
            # transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            for i in range(2):
                data[i] = encoders[i].fit_transform([data[i]])[0]
            data[2] = float(data[2][:-1])*1000

            class_label = encoders[2].fit_transform([data[3]])[0]

            # class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            # --> add your Python code here
            class_predicted = clf.predict([data[:3]])[0]

            # compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            # --> add your Python code here
            if class_predicted == class_label:
                accurate_predictions += 1

            # find the average accuracy of this model during the 10 runs (training and test set)
            # --> add your Python code here
        model_accuracy = accurate_predictions / len(data_test)
        model_accuracies.append(model_accuracy)

            # print the accuracy of this model during the 10 runs (training and test set).
            # your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
            # --> add your Python code here

    average_accuracy = sum(model_accuracies) / len(model_accuracies)
    print(f'final accuracy when training on {ds}: {average_accuracy}')
