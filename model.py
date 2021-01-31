import time

class SequentialModel:
    '''Provides training and inference features on this model.'''
    def __init__(self, layers,optimizer):
        self.layers = layers
        self.optimizer = optimizer
        self.train_acc = []
        self.train_loss = []
        self.valid_acc = []
        self.valid_loss = []

    def train(self,X_train,y_train,X_valid,y_valid,epochs=10,batch_size=32):
        '''Train model'''

        for epoch in range(epochs):
            self.forward()
            self.backward()
            self.update_weights()
            
    def forward(self,batch):
        pass
    def backward(self,activation):
        pass
    def update_weights():
        pass