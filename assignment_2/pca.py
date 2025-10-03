# -------------------------------------------------------------------------
# AUTHOR: Milosz Kryzia
# FILENAME: pca.py
# SPECIFICATION: This program performs PCA on a dataset to identify the feature whose removal maximizes the variance explained by the first principal component (PC1).
# FOR: CS 4440 (Data Mining) - Assignment #2
# TIME SPENT: 15 minutes
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data
#--> add your Python code here
df = pd.read_csv('./heart_disease_dataset.csv')

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#Get the number of features
#--> add your Python code here
num_features = df.shape[1]

highest_variance = 0
feature_removed = ""

# Run PCA using 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    # --> add your Python code here
    reduced_data = df.drop(df.columns[i], axis=1)

    # Run PCA on the reduced dataset
    # --> add your Python code here
    pca = PCA(n_components=1)
    pca.fit(reduced_data)

    #Store PC1 variance and the feature removed
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    # --> add your Python code here
    
    pc1_variance = pca.explained_variance_ratio_[0]
    feature_removed_i = df.columns[i]
    
    # Find the maximum PC1 variance
    # --> add your Python code here
    if pc1_variance > highest_variance:
        highest_variance = pc1_variance
        feature_removed = feature_removed_i

#Print results
#Use the format: Highest PC1 variance found: ? when removing ?

print(f'Highest PC1 variance found: {highest_variance} when removing {feature_removed}')