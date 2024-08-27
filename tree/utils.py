
"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these functions are here to simply help you.
"""

import numpy as np
import pandas as pd

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X, drop_first=False, dtype=float)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if y.dtype in  ['int8','int16','int32','int64','uint8','uint16','uint32','uint64','float16','float32','float64','float128']:
        return True
    elif (y.dtype=='object' or y.dtype == 'category'):
        return False
    else: 
        return False


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    probabilities = Y.value_counts(normalize=True).values
    return -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    probabilities = Y.value_counts(normalize=True)
    
    # Calculate Gini index
    gini = 1 - np.sum(probabilities ** 2)
    
    return gini

def mse(Y:pd.Series)-> float:
    return Y.var(ddof=0)

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    attr = attr.sort_values()

    # Compute the mean of consecutive elements
    means = attr.rolling(window=2).mean().dropna()
    
    if check_ifreal(Y):
        
        assert (criterion in ["gini_index", "information_gain", "MSE"])       
        if criterion == "MSE":
            max_gain=0
            value = 0
            for i in means:
                series1 = Y.loc[attr.loc[attr>=i].index]
                series2 = Y.loc[attr[attr<i].index]
                gainz = mse(Y) - (series1.size/attr.size )*mse(series1) - (series2.size/attr.size )*mse(series2)
                if gainz>max_gain:
                    max_gain = gainz
                    value =i
            return max_gain , value
    elif not check_ifreal(Y):
        assert (criterion in ["gini_index", "information_gain", "MSE"]) 
        if criterion == "information_gain" :
            max_gain=0
            value = 0
            for i in means:
                series1 = Y.loc[attr.loc[attr>=i].index]
                series2 = Y.loc[attr.loc[attr<i].index]
                gainz = entropy(Y) - (series1.size/attr.size )*entropy(series1) - (series2.size/attr.size )*entropy(series2)
                if gainz > max_gain:
                    max_gain = gainz
                    value = i
            return max_gain ,value
        elif criterion == "gini_index" :
            max_gain=0
            value = 0
            for i in means:
                series1 = Y.loc[attr.loc[attr>=i].index]
                series2 = Y.loc[attr.loc[attr<i].index]
                gainz = entropy(Y) - (series1.size/attr.size )*entropy(series1) - (series2.size/attr.size )*entropy(series2)
                if gainz > max_gain:
                    max_gain = gainz
                    value = i
            return max_gain ,value        
        
        


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    
    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    best_split = {}
    gains = np.array([np.array(information_gain(y, X[i], criterion)) for i in features])
    # print(gains.shape)
    best_split['best_feature_index'] = np.argmax(gains[:, 0])
    best_split['best_feature'] = features[best_split['best_feature_index']]
    best_split['max_info_gain'] = gains[best_split['best_feature_index'], 0]
    best_split['threshold_value'] = gains[best_split['best_feature_index'], 1]

    return best_split
    
    pass


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    Left_Child_X = X[X[attribute] < value]
    Right_Child_X = X[X[attribute] >= value]
    Left_Child_y = y[X[attribute] < value]
    Right_Child_y = y[X[attribute] >= value]
    return [Left_Child_X, Left_Child_y,Right_Child_X, Right_Child_y]
    pass


class Node():
    def __init__(self, feature_index=None, threshold=None, left_child=None, right_child=None, info_gain=None, value=None):

        
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left_child
        self.right = right_child
        self.info_gain = info_gain
        self.value = value

    def is_leaf(self):
        return self.value is not None
    


