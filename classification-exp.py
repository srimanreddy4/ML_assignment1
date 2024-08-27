import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split,cross_val_score, KFold

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.
feature_df = pd.DataFrame(X)
target_series = pd.Series(y, dtype="category")
target_df = pd.DataFrame(target_series, columns=['target'])
combined_df = feature_df.join(target_df, rsuffix='_target')

features_train, features_test, target_train, target_test = train_test_split(combined_df, target_series, test_size=0.3)

decision_tree = DecisionTree(criterion="gini_index")
decision_tree.fit(features_train, target_train)
predictions = decision_tree.predict(features_test)
target_test = target_test.reset_index(drop=True)

print("Accuracy: ", accuracy(pd.Series(predictions), pd.Series(target_test)))

for class_label in target_test.unique():
    recall_score = recall(predictions, target_test, class_label)
    precision_score = precision(predictions, target_test, class_label)
    print(f"Class: {class_label}")
    print(f"Precision: {precision_score}")
    print(f"Recall: {recall_score}")

max_depths = range(1, 11)  # You can adjust the range based on your problem
outer_fold = KFold(n_splits=5, shuffle=True)
average_scores = []

for depth in max_depths:
    inner_fold = KFold(n_splits=5, shuffle=True)
    decision_tree = DecisionTree(criterion="gini_index", max_depth=depth)
    all_true_labels = []
    all_predicted_labels = []
    
    for train_idx, test_idx in inner_fold.split(combined_df):
        X_train_fold, X_test_fold = combined_df.iloc[train_idx], combined_df.iloc[test_idx]
        y_train_fold, y_test_fold = target_series.iloc[train_idx], target_series.iloc[test_idx]
        
        decision_tree.fit(X_train_fold, y_train_fold)
        fold_predictions = decision_tree.predict(X_test_fold)
        fold_predictions = np.nan_to_num(fold_predictions, nan=0)
        
        all_true_labels.extend(y_test_fold)
        all_predicted_labels.extend(fold_predictions)
    
    average_score = accuracy(pd.Series(all_true_labels), pd.Series(all_predicted_labels))
    average_scores.append(average_score)

best_depth = max_depths[np.argmax(average_scores)]
print(f"Optimal Depth: {best_depth}")