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
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        
        
    def build_tree(self, X, y,  curr_depth=0):

        num_samples, num_features = np.shape(X)
        features = X.columns[:-1]

        # splitting till max depth
        if curr_depth<self.max_depth:
            # finding the best split
            best_split = opt_split_attribute(X, y, self.criterion, features)
            # splitting the dataset
            split_dataset = split_data(X, y, best_split['best_feature'], best_split['threshold_value'])
            if best_split['max_info_gain']>0 and check_ifreal(X.iloc[:,0]):
                # recur left
                left_subtree = self.build_tree(split_dataset[0], split_dataset[1], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(split_dataset[2], split_dataset[3], curr_depth+1)
                # return decision node
                return Node(best_split["best_feature_index"], best_split["threshold_value"], 
                            left_subtree, right_subtree, best_split["max_info_gain"])
        
        # leaf_value = self.calculate_leaf_value(X.iloc[:, -1])
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)    
    
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = pd.Series(Y)
        if(check_ifreal(Y)):
            return np.mean(Y)
        else:
            counts = Y.value_counts()
            most_occuring_value = counts.idxmax()
            return most_occuring_value 

        

    
            
            
                        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 
        if not check_ifreal(X.iloc[:,0]):
            print("discrete")
            X = one_hot_encoding(X)
            
        self.root = self.build_tree(X, y)    
        pass

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        # print(f"Node: {tree.feature_index}, Threshold: {tree.threshold}, Value: {tree.value}")

        
        if tree.is_leaf():
            # print("Reached leaf node.")
            return tree.value
        # print(f"Current Feature Value: {x[tree.feature_index]}")
        feature_val = x.iloc[tree.feature_index]
        if feature_val <= tree.threshold:
            # print("Going left")+
            return self.make_prediction(x, tree.left) if tree.left else tree.value
        else:
            # print("Going right")
            return self.make_prediction(x, tree.right) if tree.right else tree.value

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        if not check_ifreal(X.iloc[:,0]):
            X = one_hot_encoding(X)
            
        predictions = [self.make_prediction(x, self.root) for _, x in X.iterrows()]
        return pd.Series(predictions)    
        pass

    # def plot(self) -> None:
    #     """
    #     Function to plot the tree

    #     Output Example:
    #     ?(X1 > 4)
    #         Y: ?(X2 > 7)
    #             Y: Class A
    #             N: Class B
    #         N: Class C
    #     Where Y => Yes and N => No
    #     """
    #     self._print_tree_recursive(self.root, depth=0, side=None)
    #     pass
    # def _print_tree_recursive(self, node, depth, side):
    #     if node is None:
    #         return

    #     indent = '    ' * depth
    #     if node.is_leaf():
    #         print(f"{indent}{side} Leaf: {round(node.value,3)}")
    #     else:
    #         print(f"{indent}{side} Split on {[node.feature_index]} <= {round(node.threshold,3)}, "
    #               f"Information Gain: {round(node.info_gain,3)}")
    #         print(f"{indent}|-- Left:")
    #         self._print_tree_recursive(node.left, depth + 1, side='L')
    #         print(f"{indent}|-- Right:")
    #         self._print_tree_recursive(node.right, depth + 1, side='R')
    
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
        self._print_tree_recursive(self.root, depth=0)
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
