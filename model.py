import time
from loss import categorical_cross_entropy
from utils import softmax,sigmoid
import numpy as np

class SequentialModel:
    '''Provides training and inference features on this model.'''
    def __init__(self, layers,loss=categorical_cross_entropy):
        self.layers = layers
        self.loss = loss
        self.params = []
        for layer in layers:
            self.params.append(layer.params)

    def forward(self,batch_x):
        for layer in self.layers:
            X = layer.forward(batch_x)
        return X

    def backward(self,loss_grad):
        grads = []
        for layer in self.layers:
            dout,grad = layer.backward(dout)
            grads.append(grad)
        return grads

    def train_step(self,batch_x,batch_y):
        out = self.forward(batch_x)
        loss, loss_grad, _ = self.loss(out,batch_y)
        grads = self.backward(loss_grad)
        return loss, grads

    def predict_probab(self,X):
        out = self.forward(X)
        if(self.loss.__name__.__contains__('categorical')):
            probs = np.array([softmax(vect) for vect in out])
        elif(self.loss.__name__.__contains__('binary')):
            probs = np.array([sigmoid(x) for x in out])
        return np.max(probs,axis=1)

    
