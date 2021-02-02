import time

class SequentialModel:
    '''Provides training and inference features on this model.'''
    def __init__(self, layers,loss):
        self.layers = layers
        self.loss = loss
        self.params = []
        for layer in layers:
            self.params.append(layer.params)

    def forward(self,batch_x):
        for layer in self.layers:
            X = layer.forward(batch_x)
        return X

    def backward(self,dout):
        grads = []
        for layer in self.layers:
            dout,grad = layer.backward(dout)
            grads.append(grad)
        return grads

    def train_step(self,batch_x,batch_y):
        out = self.forward(batch_x)
        #TODO return loss and grads

    def predict(self,X):
        #TODO self.forward and return max of probabs
    
