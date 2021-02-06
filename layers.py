import numpy as np
from initializers import *
from utils import *


class Conv2d:
    '''Created a convolution kernel that is convolved with the layer input to produce a tensor of outputs'''
    def __init__(self, kernel_size, num_filters,input_shape,padding='same',activation='relu',use_bias=False):
        self.num_filters = num_filters
        self.k = kernel_size
        self.padding = padding
        self.s = 1

        self.in_H, self.in_W, self.in_channels = input_shape

        self.use_bias = use_bias
        self.W = xavier_init(self.in_channels,num_filters,(kernel_size,kernel_size,self.in_channels,num_filters))
        self.params = [self.W]

        if(self.padding=='same'):
            self.pad = int(np.ceil((self.k-1)/2))
            self.new_H,self.new_W = self.in_H,self.in_W
        elif(self.padding=='valid'):
            self.pad = 0
            self.new_H,self.new_W = self.in_H-self.k+1,self.in_W-self.k+1
        else:
            raise KeyError("Unknow padding value {0}".format(self.padding))

        self.out_shape = (self.new_H,self.new_W,self.num_filters)

    
    def apply_filter(self,curr_slice,filter):
        '''
        Apply filter on a current slice of data
        
        Arguments: 
        curr_slice - current slice of data, shape (kernel_size,kernel_size,N_input_channels)
        filter - weight array, shape (kernel_size,kernel_size,N_input_channels)
        
        Returns:
        out - scalar value, result of convoluting the sliding window on a slice of input data
        '''

        assert filter.shape == curr_slice.shape

        res = np.multiply(curr_slice,filter)
        out = np.sum(res)
        if(self.use_bias):
            #add bias
            pass
        return out

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
        self.n_samples = batch.shape[0]
        output = np.zeros((self.n_samples,self.new_H,self.new_W,self.num_filters))
        padded_imgs = self.zero_pad(batch,self.pad)
        self.input = padded_imgs

        for i in range(self.n_samples):
            img_padded = padded_imgs[i,:,:,:]
            for h in range(self.new_H):
                for w in  range(self.new_W):
                    for c in range(self.num_filters):
                        #corners of sliding window:
                        h_start,h_end = h*self.s,h*self.s+self.k
                        w_start,w_end = w*self.s,w*self.s+self.k

                        curr_slice = img_padded[h_start:h_end,w_start:w_end,:]
                        output[i,h,w,c] = self.apply_filter(curr_slice,self.W[:,:,:,c])

        return output

    def backward(self,gradient):
        flow_gradient = np.zeros((self.n_samples,self.in_H,self.in_W,self.in_channels))
        dW = np.zeros_like(self.W)

        flow_gradient = self.zero_pad(flow_gradient,self.pad)

        for i in range(self.n_samples):
            curr_img = self.input[i,:,:,:]
            for h in range(self.new_H):
                for w in range(self.new_W):
                    for c in range(self.num_filters):
                        #corners of sliding window:
                        h_start,h_end = h*self.s,h*self.s+self.k
                        w_start,w_end = w*self.s,w*self.s+self.k

                        curr_slice = curr_img[h_start:h_end,w_start:w_end,:]
                        flow_gradient[i,h_start:h_end,w_start:w_end,:] += self.W[:,:,:,c]*gradient[i,h,w,c]
                        dW[:,:,:,c] += curr_slice*gradient[i,h,w,c]
            

        return flow_gradient[:,self.pad:-self.pad,self.pad:-self.pad,:], [dW]


class MaxPooL2d:
    '''Downsamples the input representation by taking the maximum value over the sliding window defined by kernel_size'''

    def __init__(self, kernel_size, input_shape):
        self.k = kernel_size
        self.s = kernel_size
        self.params = []

        self.in_H, self.in_W, self.n_channels = input_shape
        self.new_H, self.new_W = self.in_H//self.k, self.in_W//self.k

        self.out_shape = (self.new_H,self.new_W,self.n_channels)

    def forward(self,batch):
        '''
        Implements the forward propagation for a maximum pooling function

        Arguments:
        batch - current batch of data, shape (N_samples,H,W,N_input_channels)

        Returns: array of downsampled images for all samples in batch, shape (N_samples,H_new,W_new,N_input_channels)
        '''
        self.input = batch
        self.n_samples, _,_,_ = batch.shape
        
        output = np.zeros((self.n_samples,self.new_H,self.new_W,self.n_channels))

        for i in range(self.n_samples):
            curr_img = batch[i,:,:,:]
            for h in range(self.new_H):
                for w in range(self.new_W):
                    for c in range(self.n_channels):
                        #corners of sliding window:
                        h_start,h_end = h*self.s,h*self.s+self.k
                        w_start,w_end = w*self.s,w*self.s+self.k

                        curr_slice = curr_img[h_start:h_end,w_start:w_end,c]
                        output[i,h,w,c] = np.max(curr_slice)

        return output

    def get_max_pos(self,slice, gradient):
        return gradient*(slice==np.max(slice))

    def backward(self, gradient):
        '''
        Implements the backward pass for a maxpooling layer

        Arguments:
        gradient - gradient from the previous layer(downsampled image), shape (N_samples,H_new,W_new,N_input_channels)

        Returns: 
        '''
        flow_gradient = np.zeros((self.n_samples,self.in_H,self.in_W,self.n_channels))
        
        for i in range(self.n_samples):
            curr_img = self.input[i,:,:,:]
            for h in range(self.new_H):
                for w in range(self.new_W):
                    for c in range(self.n_channels):
                        #corners of sliding window:
                        h_start,h_end = h*self.s,h*self.s+self.k
                        w_start,w_end = w*self.s,w*self.s+self.k

                        curr_slice = curr_img[h_start:h_end,w_start:w_end,c]
                        flow_gradient[i,h_start:h_end,w_start:w_end,c] += self.get_max_pos(curr_slice,gradient[i,h,w,c])

        return flow_gradient, []
    

class Flatten:
    def __init__(self):
        self.params = []
    
    def forward(self,batch):
        self.input_shape = batch.shape
        out_shape = (batch.shape[0],-1)
        output = batch.ravel().reshape(out_shape)
        return output

    def backward(self,gradient):
        flow_gradient = gradient.reshape(self.input_shape)
        return flow_gradient, []


class Dense:
    '''Dense Layer'''

    def __init__(self,n_output,n_input,use_bias=True):
        self.n_output = n_output
        self.n_input = n_input
        self.use_bias = use_bias
        self.W = xavier_init(n_input,n_output,(n_input,n_output))
        self.b = xavier_init(n_input,n_output,(1,n_output)) if use_bias else np.zeros((1,n_output))
         
        self.params = [self.W,self.b]     

    def forward(self,batch):
        '''
        Implements the forward propagation for a dense layer
        
        Arguments:
        batch - current batch of data, shape (N_samples,N_input)

        Returns: dot product of (X,W) + bias term, output shape (N_samples,N_output)
        '''
        self.X = batch
        assert self.X .shape[1] == self.n_input
        output = np.dot(self.X,self.W)+self.b
        
        return output

    def backward(self,gradient):
        '''
        Implements the backward pass for a dense layer

        Arguments:
        gradient - gradient from the previous layer, shape (N_samples,N_output)

        Returns: gradient to flow back to the earlier layers, list of gradients to change params of the current layer
        '''
        dW  = np.dot(self.X.T,gradient)
        db =  np.sum(gradient,axis = 0)
        flow_gradient = np.dot(gradient,self.W.T) #dX
        return flow_gradient, [dW,db]

class Relu:
    def __init__(self):
        self.params = []

    def forward(self,batch):
        self.X = batch
        return np.maximum(0,self.X)

    def backward(self,gradient):
        flow_gradient = gradient.copy()
        flow_gradient[self.X <= 0] = 0
        return flow_gradient, []
    

class Sigmoid():
    def __init__(self):
       self.params = []
    
    def forward(self,batch):
        self.X = np.array(sigmoid(x) for x in batch)
        return self.X 

    def backward(self,gradient):
        flow_gradient = gradient*self.X*(1-self.X)
        return flow_gradient, []









        

