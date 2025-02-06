import numpy as np
from bases import Layer, LayerWithWeights


class AffineLayer(LayerWithWeights):
    def __init__(self, input_size, output_size, seed=None):
        super(AffineLayer, self).__init__(input_size, output_size, seed=seed)

    def forward(self, x):
        '''
            :param x: activations/inputs from previous layer
            :return: output of affine layer
        '''
        out = None
        ##### YOUR CODE STARTS #####

        # Vectorize the input to [batchsize, others] array
        x_vector = x.reshape(x.shape[0], -1)

        # Do the affine transform
        out = np.dot(x_vector, self.W) + self.b

        ##### YOUR CODE ENDS #######
        # Save x for using in backward pass
        self.x = x.copy()

        return out

    def backward(self, dprev):
        '''
            :param dprev: gradient of next layer:
            :return: downstream gradient
        '''

        batch_size = self.x.shape[0]
        x_vectorized = None
        dx, dw, db = None, None, None

        # YOUR CODE STARTS
        # Vectorize the input to a 1D ndarray
        x_vector = self.x.reshape(batch_size, -1)

        dw = x_vector.T.dot(dprev)  
        db = dprev.sum(axis=0)          
        dx = dprev.dot(self.W.T).reshape(self.x.shape)

        # YOUR CODE ENDS

        # Save them for backward pass
        self.db = db.copy()
        self.dW = dw.copy()
        return dx, dw, db

    def __repr__(self):
        return 'Affine layer'
