import numpy as np

class MaxPooL2d:
    '''
    Downsamples the input representation by taking the maximum value over the sliding window defined by kernel_size

    Attributes:
    kernel_size : int
        Height and Width of the 2D convolution window
    
    input_shape: tuple(N_samples, in_H, in_W, N_in_channels)
        Input tensor shape, (Number of samples in a batch, Input height, Input width, Number of input channels)

    stride(s): int
        The stride of the pooling

    out_shape: tuple
        Output tensor shape, (Number of samples in a batch, Output height, Output width, Number of output channels)

    params: list
        Trainable params of current layer
    '''

    def __init__(self, kernel_size, input_shape):
        self.k = kernel_size
        self.s = kernel_size
        self.params = []

        self.n_samples, self.in_H, self.in_W, self.n_channels = input_shape
        self.new_H, self.new_W = self.in_H//self.k, self.in_W//self.k

        self.out_shape = (self.new_H, self.new_W, self.n_channels)

    def forward(self, batch):
        '''
        Implements the forward propagation for a maximum pooling function

        Arguments:
        batch : np.ndarray
            current batch of data, shape (N_samples, in_H, in_W, N_input_channels)

        Returns: 
        output: np.ndarray
            array of downsampled images for all samples in batch, shape (N_samples, H_new, W_new, N_input_channels)
        '''
        self.input = batch
        self.n_samples, _, _, _ = batch.shape

        output = np.zeros((self.n_samples, self.new_H,
                           self.new_W, self.n_channels))

        for i in range(self.n_samples):
            curr_img = batch[i, :, :, :]
            for h in range(self.new_H):
                for w in range(self.new_W):
                    for c in range(self.n_channels):
                        # corners of sliding window:
                        h_start, h_end = h*self.s, h*self.s+self.k
                        w_start, w_end = w*self.s, w*self.s+self.k

                        curr_slice = curr_img[h_start:h_end, w_start:w_end, c]
                        output[i, h, w, c] = np.max(curr_slice)

        return output

    def get_max_pos(self, slice, gradient):
        '''Get positions of max elements in input slice'''
        return gradient*(slice == np.max(slice))

    def backward(self, gradient):
        '''
        Implements the backward pass of a maxpooling layer

        Arguments:
        gradient
            gradient from the previous layer(downsampled image), shape (N_samples,H_new,W_new,N_input_channels)

        Returns:
        flow_gradient : np.ndarray
            gradient to flow back to the earlier layers, shape(N_samples,H_input,W_input,N_input_channels), 
            list of gradients to change params of the current layer 
        '''
        flow_gradient = np.zeros((self.n_samples, self.in_H, self.in_W, self.n_channels))

        for i in range(self.n_samples):
            curr_img = self.input[i, :, :, :]
            for h in range(self.new_H):
                for w in range(self.new_W):
                    for c in range(self.n_channels):
                        # corners of sliding window:
                        h_start, h_end = h*self.s, h*self.s+self.k
                        w_start, w_end = w*self.s, w*self.s+self.k

                        curr_slice = curr_img[h_start:h_end, w_start:w_end, c]
                        flow_gradient[i, h_start:h_end, w_start:w_end, c] += self.get_max_pos(curr_slice, gradient[i, h, w, c])

        return flow_gradient, []