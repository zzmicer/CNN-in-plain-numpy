import numpy as np

def convert_prob_to_onehot(y_prob):
    '''Convert probabilitie from softmax layer to one-hot vectors'''
    onehot = np.zeros(y_prob.shape)
    max_pos = np.argmax(y_prob,axis=1)
    for i,pos in enumerate(max_pos):
        onehot[i][pos] = 1
    return onehot