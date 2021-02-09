import numpy as np
from utils import softmax,sigmoid

def categorical_cross_entropy(outs,batch_y):
    '''Cross entropy with the softmax activation function'''
    probs = np.array([softmax(vect) for vect in outs])
    cce = -np.sum(batch_y*np.log(np.clip(probs,1e-20,1.)))/batch_y.shape[0] #categorical_cross_entropy
    grad = probs - batch_y
    return cce, grad

def binary_cross_entropy(outs,batch_y):
    '''Cross entropy with the sigmoid activation function'''
    probs = np.array([sigmoid(x) for x in outs])
    bce = -np.sum(batch_y*np.log(probs)+(1-batch_y)*np.log(1-probs))/batch_y.shape #binary croos entropy
    grad = probs - batch_y
    return bce, grad


    
