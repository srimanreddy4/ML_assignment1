import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
np.random.seed(42)
# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])
# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn

columns_drop = ['model year', 'car name', 'origin', 'cylinders']
for column in columns_drop:
    data = data.drop(column, axis=1)

data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()

target_column = data.pop('mpg')
data['mpg'] = target_column

feature_matrix = data.iloc[:, :-1]
target_vector = data.iloc[:, -1]

feature_train, feature_test, target_train, target_test = train_test_split(feature_matrix, target_vector, test_size=0.3, random_state=42)
train_dataset = pd.concat([feature_train, target_train], axis=1)

# Fitting and testing on our decision tree classifier.
custom_tree = DecisionTree(criterion="MSE", max_depth=5)  # Split based on Inf. Gain
custom_tree.fit(feature_train, target_train)
custom_predictions = custom_tree.predict(feature_test)

# Fitting and testing on sklearn decision tree classifier.
sklearn_tree = DecisionTreeRegressor(max_depth=5, criterion="squared_error")
sklearn_tree.fit(feature_train, target_train)
sklearn_predictions = sklearn_tree.predict(feature_test)


sklearn_rmse = np.sqrt(mean_squared_error(target_test, sklearn_predictions))
sklearn_mae = mean_absolute_error(target_test, sklearn_predictions)

print("RMSE of custom decision tree", rmse(custom_predictions, target_test))
print(f"RMSE of sklearn decision tree: {sklearn_rmse}")
print("MAE of custom decision tree: ", mae(custom_predictions, target_test))
print(f"MAE of sklearn decision tree: {sklearn_mae}")