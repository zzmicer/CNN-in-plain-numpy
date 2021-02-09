import numpy as np

def convert_prob_to_onehot(y_prob):
    '''Convert probabilitie from softmax layer to one-hot vectors'''
    onehot = np.zeros(y_prob.shape)
    max_pos = np.argmax(y_prob,axis=1)
    for i,pos in enumerate(max_pos):
        onehot[i][pos] = 1
    return onehot

def softmax(input):
    '''Softmax function'''
    exp = np.exp(input)
    return exp/np.sum(exp)

def sigmoid(input):
    '''Softmax function'''
    return 1/(1+np.exp(input))

def accuracy(y_true,y_hat):
    y_pred = convert_prob_to_onehot(y_hat)
    return (y_pred==y_true).all(axis=1).mean()
