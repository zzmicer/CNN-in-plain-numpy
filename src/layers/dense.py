import numpy as np
from src.initializers import xavier_init

class Dense:
    '''
    Regular fully-connected layer
    
    Attributes:
    n_output : int
        Number of output neurons

    n_input : int 
        Number of input neurons

    use_bias: bool
        Boolean, whether the layer uses a bias vector.

    W: np.ndarray
        Weight array of shape (N_input, N_output)
    
    b: np.ndarray
        Bias term, shape (1, N_output)
    
    params: list
        Trainable params of current layer
    '''

    def __init__(self, n_output, n_input, use_bias=True):
        self.n_output = n_output
        self.n_input = n_input
        self.use_bias = use_bias
        self.W = xavier_init(n_input, n_output, (n_input, n_output))
        self.b = xavier_init(n_input, n_output, (1, n_output)
                             ) if use_bias else np.zeros((1, n_output))

        self.params = [self.W, self.b]

    def forward(self, batch):
        '''
        Implements the forward propagation for a dense layer

        Arguments:
        batch: np.ndarray
            current batch of data, shape (N_samples,N_input)

        Returns:
        output: np.ndarray
            dot product of (X,W) + bias term, output shape (N_samples,N_output)
        '''
        self.X = batch
        assert self.X .shape[1] == self.n_input
        output = np.dot(self.X, self.W)+self.b

        return output

    def backward(self, gradient):
        '''
        Implements the backward pass for a dense layer

        Arguments:
        gradient: np.ndarray
            gradient from the previous layer, shape (N_samples,N_output)

        Returns:
        flow_gradient: np.ndarray
            gradient to flow back to the earlier layers, list of gradients to change params of the current layer
        '''
        dW = np.dot(self.X.T, gradient)
        db = np.sum(gradient, axis=0)
        flow_gradient = np.dot(gradient, self.W.T)  # dX
        return flow_gradient, [dW, db]