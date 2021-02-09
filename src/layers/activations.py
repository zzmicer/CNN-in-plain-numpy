import numpy as np
from src.utils import sigmoid

class Relu:
    '''Rectified Linear Unit activation function'''
    def __init__(self):
        self.params = []

    def forward(self, batch):
        self.X = batch
        return np.maximum(0, self.X)

    def backward(self, gradient):
        flow_gradient = gradient.copy()
        flow_gradient[self.X <= 0] = 0
        return flow_gradient, []


class Sigmoid():
    '''Sigmoid activation function'''
    def __init__(self):
        self.params = []

    def forward(self, batch):
        self.X = np.array(sigmoid(x) for x in batch)
        return self.X

    def backward(self, gradient):
        flow_gradient = gradient*self.X*(1-self.X)
        return flow_gradient, []
