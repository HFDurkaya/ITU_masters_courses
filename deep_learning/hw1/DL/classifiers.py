import numpy as np
from bases import Layer

class Softmax(Layer):
    def __init__(self):
        self.probs = None

    def forward(self, x):
        '''
            Softmax function
            :param x: Input for classification (Likelihoods)
            :return: Class Probabilities
        '''
        # Normalize the class scores (i.e output of affine linear layers)
        # In order to avoid numerical unstability.
        # Do not forget to copy the output to object to use it in backward pass
        probs = None
       
        # Your implementation starts
        # NxC shaped multiplication

        x = x - np.max(x, axis=1, keepdims=True)
        
        exp_x = np.exp(x)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True) 
 
        # Save them for backward pass
        self.probs = probs.copy()

        return probs

    def backward(self, y):
        '''
            Implement the backward pass w.r.t. softmax loss
            -----------------------------------------------
            :param y: class labels. (as an array, [1,0,1, ...]) Not as one-hot encoded
            :return: upstream derivate

        '''
        dx = None
        # Your implementation starts

        N = y.shape[0]
        dx = self.probs.copy()
        
        dx[np.arange(N), y] = dx[np.arange(N), y] - 1
        
        dx = dx / N

        # End of your implementation

        return dx
