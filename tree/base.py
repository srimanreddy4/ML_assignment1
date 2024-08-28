"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import Node
from tree.utils import *


np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.base_node = None
        self.criterion = criterion
        self.max_depth = max_depth
        
    def build_tree(self, X, y, curr_depth=0):

        num_rows, num_features = np.shape(X)
        feature_list = X.columns[:-1]

        # splitting till max depth
        if curr_depth < self.max_depth:
            # finding the best split
            optimal_split = opt_split_attribute(X, y, self.criterion, feature_list)
            # splitting the dataset
            split_data_set = split_data(X, y, optimal_split['best_feature'], optimal_split['threshold_value'])
            if optimal_split['max_info_gain'] > 0 and check_ifreal(X.iloc[:, 0]):
                left_subtree = self.build_tree(split_data_set[0], split_data_set[1], curr_depth + 1)
                right_subtree = self.build_tree(split_data_set[2], split_data_set[3], curr_depth + 1)
                return Node(optimal_split["best_feature_index"], optimal_split["threshold_value"], 
                            left_subtree, right_subtree, optimal_split["max_info_gain"])
        
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)    
    
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = pd.Series(Y)
        if check_ifreal(Y):
            return np.mean(Y)
        else:
            value_counts = Y.value_counts()
            most_frequent_value = value_counts.idxmax()
            return most_frequent_value 
                        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        if not check_ifreal(X.iloc[:, 0]):
            X = one_hot_encoding(X)
            
        self.base_node = self.build_tree(X, y)    
        pass

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        if tree.is_leaf():
            return tree.value
        feature_value = x.iloc[tree.feature_index]
        if feature_value <= tree.threshold:
            return self.make_prediction(x, tree.left) if tree.left else tree.value
        else:
            return self.make_prediction(x, tree.right) if tree.right else tree.value

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs
        """

        if not check_ifreal(X.iloc[:, 0]):
            X = one_hot_encoding(X)
            
        predictions = [self.make_prediction(x, self.base_node) for _, x in X.iterrows()]
        return pd.Series(predictions)    
    
    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self._print_tree_recursive(self.base_node, depth=0)
        pass
    
    def _print_tree_recursive(self, node, depth):
        if node is None:
            return

        indent = '    ' * depth
        if node.is_leaf():
            print(f"{'Value:'} {node.value}")
        else:
            print(f"?(feature'{node.feature_index}' > {node.threshold})")
            print(f"{indent}   Y: ", end="")
            self._print_tree_recursive(node.left, depth + 1)
            print(f"{indent}   N: ", end="")
            self._print_tree_recursive(node.right, depth + 1)
