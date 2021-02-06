from sklearn.utils import shuffle
import numpy as np

class MiniBatchSGD:
    '''Mini batch stohastic gradient descent'''
    def __init__(self,nnet,X_train,y_train,batch_size,epoch,learning_rate):
        self.nnet = nnet
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = learning_rate
            
    
    def get_minibatches(self,X,y,shuffle_flag=True):
        minbatches = []
        if(shuffle_flag):
            X,y = shuffle(X,y)

        for i in range(0,X.shape[0],self.batch_size):
            batch_x = X[i:i+self.batch_size,:,:,:]
            batch_y = y[i:i+self.batch_size,]
            minbatches.append((batch_x,batch_y))
        return minbatches


    def update_params(self,params,grads):
        for param,grad in zip(params,reversed(grads)):
            for i in range(len(param)):
                param[i] -= self.lr*grad[i]
                    
    def train(self,verbose=True):
        batches = self.get_minibatches(self.X_train,self.y_train)
        for i in range(self.epoch):
            loss = 0
            if(verbose):
                print("Epoch {0}".format(i+1))
            for x,y in batches:
                loss, grads = self.nnet.train_step(x,y)
                self.update_params(self.nnet.params,grads)
            if(verbose):
                #train_acc = self.nnet.predict(self.X_train)
                print("Train Loss - {0},Train Accuracy - ".format(loss))
    
        

        