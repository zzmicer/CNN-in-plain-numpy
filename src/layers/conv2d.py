import numpy as np
from src.initializers import xavier_init

class Conv2d:
    '''
    Creates a convolution kernel that is convolved with the input layer to produce a tensor of outputs

    Attributes:
    kernel_size(k) : int
        Height and Width of the 2D convolution window

    num_filters : int 
        The number of output filters in the convolution

    input_shape: tuple(N_samples, in_H, in_W, N_in_channels)
        Input tensor shape, (Number of samples in a batch, Input height, Input width, Number of input channels)

    stride(s): int
        The stride of the convolution

    padding: str
        One of "valid" or "same". 
        "valid" means no padding. 
        "same" results in padding evenly to the left/right or up/down of the input
         such that output has the same height/width dimension as the input.

    use_bias: bool
        Boolean, whether the layer uses a bias vector.

    W: np.ndarray
        Weight array of shape (kernel_size, kernel_size, Number of input channels, Number of output channels)

    out_shape: tuple
        Output tensor shape, (Number of samples in a batch, Output height, Output width, Number of output channels)

    params: list
        Trainable params of current layer
    '''

    def __init__(self, kernel_size, num_filters, input_shape, padding='same', use_bias=False):
        self.num_filters = num_filters
        self.k = kernel_size
        self.s = 1
        self.use_bias = use_bias

        self.n_samples, self.in_H, self.in_W, self.in_channels = input_shape

        self.W = xavier_init(self.in_channels, num_filters,
                             (kernel_size, kernel_size, self.in_channels, num_filters))
        self.params = [self.W]

        if(padding == 'same'):
            self.pad = int(np.ceil((self.k-1)/2))
            self.new_H, self.new_W = self.in_H, self.in_W
        elif(padding == 'valid'):
            self.pad = 0
            self.new_H, self.new_W = self.in_H-self.k+1, self.in_W-self.k+1
        else:
            raise KeyError("Unknow padding value {0}".format(padding))

        self.out_shape = (self.n_samples, self.new_H, self.new_W, self.num_filters)

    def apply_filter(self, curr_slice, filter):
        '''
        Apply filter on a current slice of data

        Arguments: 

        curr_slice : np.ndarray 
            current slice of data, shape (kernel_size,kernel_size,N_input_channels)

        filter : np.ndarray
            weight array, shape (kernel_size,kernel_size,N_input_channels)

        Returns:
        
        out : int
            scalar value, result of convoluting the sliding window on a slice of input data
        '''
        assert filter.shape == curr_slice.shape

        res = np.multiply(curr_slice, filter)
        out = np.sum(res)
        if(self.use_bias):
            # add bias
            pass
        return out

    def zero_pad(self, batch, n):
        '''Pad image with n zeros'''
        return np.pad(batch, ((0, 0), (n, n), (n, n), (0, 0)))

    def forward(self, batch):
        '''
        Implements the forward propagation for a convolution function

        Arguments:
        batch : np.ndarray
            current batch of data, shape (N_samples,H,W,N_input_channels)

        Returns:
        output : np.ndarray
            array of feature maps for all samples in batch, shape (N_samples,H_new,W_new,N_channels)
        '''
        self.n_samples = batch.shape[0]
        output = np.zeros((self.n_samples, self.new_H,
                           self.new_W, self.num_filters))
        padded_imgs = self.zero_pad(batch, self.pad)
        self.input = padded_imgs

        for i in range(self.n_samples):
            img_padded = padded_imgs[i, :, :, :]
            for h in range(self.new_H):
                for w in range(self.new_W):
                    for c in range(self.num_filters):
                        # corners of sliding window:
                        h_start, h_end = h*self.s, h*self.s+self.k
                        w_start, w_end = w*self.s, w*self.s+self.k

                        curr_slice = img_padded[h_start:h_end,
                                                w_start:w_end, :]
                        output[i, h, w, c] = self.apply_filter(
                            curr_slice, self.W[:, :, :, c])
        return output

    def backward(self, gradient):
        '''
        Implements the backpropagation for a convolution function

        Arguments:
        gradient : np.ndarray
            gradient from the previous layer, shape (N_samples,H_new,W_new,N_channels)

        Returns:
        flow_gradient : np.ndarray
            gradient to flow back to the earlier layers, shape(N_samples,H_input,W_input,N_input_channels), 
            list of gradients to change params of the current layer
        '''
        flow_gradient = np.zeros((self.n_samples, self.in_H, self.in_W, self.in_channels))
        dW = np.zeros_like(self.W)

        flow_gradient = self.zero_pad(flow_gradient, self.pad) #add padding

        for i in range(self.n_samples):
            curr_img = self.input[i, :, :, :]
            for h in range(self.new_H):
                for w in range(self.new_W):
                    for c in range(self.num_filters):
                        # corners of sliding window:
                        h_start, h_end = h*self.s, h*self.s+self.k
                        w_start, w_end = w*self.s, w*self.s+self.k

                        curr_slice = curr_img[h_start:h_end, w_start:w_end, :]
                        flow_gradient[i, h_start:h_end, w_start:w_end,
                                      :] += self.W[:, :, :, c]*gradient[i, h, w, c]
                        dW[:, :, :, c] += curr_slice*gradient[i, h, w, c]

        return flow_gradient[:, self.pad:-self.pad, self.pad:-self.pad, :], [dW] #remove padding
