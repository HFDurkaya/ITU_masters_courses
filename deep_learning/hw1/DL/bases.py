import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    '''
        Abstract layer class which implements forward and backward methods
    '''

    def __init__(self):
        self.x = None

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('Abstract class!')

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError('Abstract class!')

    def __repr__(self):
        return 'Abstract layer class'

class LayerWithWeights(Layer):
    '''
        Abstract class for layer with weights(CNN, Affine etc...)
    '''

    def __init__(self, input_size, output_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W = np.random.rand(input_size, output_size)
        self.b = np.zeros(output_size)
        self.x = None
        self.db = np.zeros_like(self.b)
        self.dW = np.zeros_like(self.W)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('Abstract class!')

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError('Abstract class!')

    def __repr__(self):
        return 'Abstract layer class'

class Model(Layer):
    def __init__(self, model=None):
        self.layers = model
        self.y = None

    def __call__(self, moduleList):
        for module in moduleList:
            if not isinstance(module, Layer):
                raise TypeError(
                    'All modules in list should be derived from Layer class!')

        self.layers = moduleList

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y):
        self.y = y.copy()
        dprev = y.copy()
        dprev = self.layers[-1].backward(y)

        for layer in reversed(self.layers[:-1]):
            if isinstance(layer, LayerWithWeights):
                dprev = layer.backward(dprev)[0]
            else:
                dprev = layer.backward(dprev)
        return dprev

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        return 'Model consisting of {}'.format('/n -- /t'.join(self.layers))

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]
