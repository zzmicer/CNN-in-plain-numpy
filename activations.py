import numpy as np

def softmax(input):
    '''Softmax activation function'''
    exp = np.exp(input)
    return exp/np.sum(exp)


def relu(input):
    '''Rectified Linear Unit activation function'''
    return np.maximum(0,input)