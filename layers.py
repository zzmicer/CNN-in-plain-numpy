import numpy as np
from activations import *
from initializers import *


class Conv2d:
    '''Created a convolution kernel that is convolved with the layer input to produce a tensor of outputs'''
    def __init__(self, kernel_size, num_filters,input_num_filters,padding='same',activation='relu',use_bias=False):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = 1
        self.use_bias = use_bias
        self.W = xavier_init(input_num_filters,num_filters,(kernel_size,kernel_size,input_num_filters,num_filters))
        self.params = [self.W]
    
    def apply_filter(self,curr_slice,filter):
        '''
        Apply filter on a current slice of data
        
        Arguments: 
        curr_slice - current slice of data, shape (kernel_size,kernel_size,N_input_channels)
        filter - weight array, shape (kernel_size,kernel_size,N_input_channels)
        
        Returns:
        Z - scalar value, result of convoluting the sliding window on a slice of input data
        '''

        assert curr_slice.shape==filter.shape

        res = np.multiply(curr_slice,filter)
        Z = np.sum(res)
        if(self.use_bias):
            #add bias
            pass
        return Z

    def zero_pad(self,batch,n):
        '''Pad image with n zeros'''
        return np.pad(batch,((0,0),(n,n),(n,n),(0,0)))

    def forward(self,batch):
        '''
        Implements the forward propagation for a convolution function
        
        Arguments:
        batch - current batch of data, shape (N_samples,H,W,N_input_channels)

        Returns: array of feature maps for all samples in batch, shape (N_samples,H_new,W_new,N_channels)
        '''
        n_samples, H, W, _ = batch.shape
        new_H,new_W,pad = 0,0,0
        k,s = self.kernel_size, self.stride

        if(self.padding=='same'):
            pad = int((k-1)/2)
            new_H,new_W = H,W
        elif(self.padding=='valid'):
            pad = 0
            new_H,new_W = H-k+1,W-k+1
        else:
            raise KeyError("Unknow padding value {0}".format(self.padding))
        
        output = np.zeros((n_samples,new_H,new_W,self.num_filters))
        padded_imgs = self.zero_pad(batch,pad)

        for i in range(n_samples):
            img_padded = padded_imgs[i,:,:,:]
            for h in range(new_H):
                for w in  range(new_W):
                    for c in range(self.num_filters):
                        #corners of sliding window:
                        h_start,h_end = h*s,h*s+k
                        w_start,w_end = w*s,w*s+k

                        curr_slice = img_padded[h_start:h_end,w_start:w_end,:]
                        output[i,h,w,c] = self.apply_filter(curr_slice,self.W[:,:,:,c])

        return output
    


class MaxPooL2d:
    '''Downsamples the input representation by taking the maximum value over the sliding window defined by kernel_size'''

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.params = []
    
    def forward(self,batch):
        '''
        Implements the forward propagation for a maximum pooling function

        Arguments:
        batch - current batch of data, shape (N_samples,H,W,N_input_channels)

        Returns: array of downsampled images for all samples in batch, shape (N_samples,H_new,W_new,N_input_channels)
        '''
        n_samples, H, W, n_channels = batch.shape
        k,s = self.kernel_size,self.stride
        new_H, new_W = H//k, W//k

        output = np.zeros((n_samples,new_H,new_W,n_channels))

        for i in range(n_samples):
            curr_img = batch[i,:,:,:]
            for h in range(new_H):
                for w in range(new_W):
                    for c in range(n_channels):
                        #corners of sliding window:
                        h_start,h_end = h*s,h*s+k
                        w_start,w_end = w*s,w*s+k

                        curr_slice = curr_img[h_start:h_end,w_start:w_end,c]
                        output[i,h,w,c] = np.max(curr_slice)
        
        return output

class Dense:
    '''Dense Layer'''

    def __init__(self,n_output,n_input,use_bias=True):
        self.n_output = n_output
        self.n_input = n_input
        self.use_bias = use_bias
        self.W = xavier_init(n_input,n_output,(n_input,n_output))
        self.b = np.zeros((1,n_output))
        
        if(use_bias):
            self.b =  xavier_init(n_input,n_output,(1,n_output))
        self.params = [self.W,self.b]
        


    def forward(self,batch):
        self.X = batch
        assert self.X .shape[1] == self.n_input
        output = np.dot(self.X,self.W)+self.b
        
        return output

    def backward(self,gradient):
        pass

class Relu:
    def __init__(self):
        self.params = []

    def forward(self,batch):
        pass

    def backward(self,gradient):
        pass
    

class Softmax():
    def __init__(self):
       self.params = []
    
    def forward(self,batch):
        pass
    def backward(self,gradient):
        pass









        

