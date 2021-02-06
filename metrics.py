from utils import convert_prob_to_onehot 
import numpy as np


def accuracy(y_true,y_hat):
    y_pred = convert_prob_to_onehot(y_hat)
    return (y_pred==y_true).all(axis=1).mean()
