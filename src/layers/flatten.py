class Flatten:
    '''
    Flattens the input

    Attributes:
    params: list
        Trainable params of current layer
    '''
    def __init__(self):
        self.params = []

    def forward(self, batch):
        '''Reshapes batch to (N_samples,*)'''
        self.input_shape = batch.shape
        out_shape = (batch.shape[0], -1)
        output = batch.ravel().reshape(out_shape)
        return output

    def backward(self, gradient):
        '''Reshapes gradient to input shape'''
        flow_gradient = gradient.reshape(self.input_shape)
        return flow_gradient, []