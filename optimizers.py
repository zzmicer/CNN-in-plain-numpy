class MiniBatchSGD:
    '''Mini batch stohastic gradient descent'''
    def __init__(self,lr,):
        self.lr = lr
    
    def update(self,layers):
        for layer in layers:
            #TODO get weights, and layer gradients
            #new weights = weigths - lr*gradient
            pass