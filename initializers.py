import numpy as np

def xavier_init(n_input,n_output,shape):
    '''Xavier initialization'''
    return np.random.normal(0,np.sqrt(2/(n_input+n_output)),shape)