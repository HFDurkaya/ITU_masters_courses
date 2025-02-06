import numpy as np
from bases import Layer


class ExponentialLinearUnit(Layer):
    def __init__(self, alpha = 1e-3):
        # Dont forget to save x or relumask for using in backward pass
        self.x = None
        self.alpha = alpha

    def forward(self, x):
        '''
            Forward pass for ELU
            :param x: outputs of previous layer
            :return: ELU activation
        '''
        # Do not forget to copy the output to object to use it in backward pass
        self.x = x.copy()
        # This is used for avoiding the issues related to mutability of python arrays
        x = x.copy()

        ####### Implement ELU activation #########
      
        x[x > 0] = x[x > 0]
        x[x <= 0] = self.alpha * (np.exp(x[x <= 0]) - 1)

        ####### End of your implementation ########
        return x

    def backward(self, dprev):
        '''
            Backward pass of ELU
            :param dprev: gradient of previos layer:
            :return: upstream gradient
        '''
        dx = None
        ##### Your implementation starts #####
        
        dx = np.zeros_like(self.x)
        dx[self.x > 0] = 1
        dx[self.x <= 0] = self.alpha * np.exp(self.x[self.x <= 0])
        
        dx = dx * dprev
        
        ##### End of your implementation #####
        return dx



class ReLU(Layer):

    def __init__(self):
        # Dont forget to save x or relumask for using in backward pass
        self.x = None

    def forward(self, x):
        '''
            Forward pass for ReLU
            :param x: outputs of previous layer
            :return: ReLU activation
        '''
        # Do not forget to copy the output to object to use it in backward pass
        self.x = x.copy()
        # This is used for avoiding the issues related to mutability of python arrays
        x = x.copy()

        # Implement relu activation
        x = np.maximum(0,x)
        # End of your implementation
        return x

    def backward(self, dprev):
        '''
            Backward pass of ReLU
            :param dprev: gradient of previos layer:
            :return: upstream gradient
        '''
        dx = None
        # Your implementation starts
        dx = dprev
        dx[np.maximum(0,self.x) == 0] = 0
        # End of your implementation
        return dx

class Softplus(Layer):

    def __init__(self):
        # Dont forget to save x or relumask for using in backward pass
        self.x = None

    def forward(self, x):
        '''
            Forward pass for Softplus
            :param x: outputs of previous layer
            :return: Softplus activation
        '''
        # Do not forget to copy the output to object to use it in backward pass
        self.x = x.copy()
        # This is used for avoiding the issues related to mutability of python arrays
        x = x.copy()

        # Implement softplus activation
        x = np.log(1 + np.exp(x))
        # End of your implementation
        return x

    def backward(self, dprev):
        '''
            Backward pass of Softplus
            :param dprev: gradient of previos layer:
            :return: upstream gradient
        '''
        dx = None
        # Your implementation starts
        dx = 1/(1 + np.exp(-self.x))*dprev
        # End of your implementation
        return dx


class Sigmoid(Layer):

    def __init__(self):
        # Dont forget to save x or relumask for using in backward pass
        self.x = None

    def forward(self, x):
        '''
            Forward pass for Sigmoid
            :param x: outputs of previous layer
            :return: Sigmoid activation
        '''
        # Do not forget to copy the output to object to use it in backward pass
        self.x = x.copy()
        # This is used for avoiding the issues related to mutability of python arrays
        x = x.copy()

        # Implement Sigmoid activation
        x = 1/(1 + np.exp(-x))
        # End of your implementation
        return x

    def backward(self, dprev):
        '''
            Backward pass of Sigmoid
            :param dprev: gradient of previos layer:
            :return: upstream gradient
        '''
        dx = None
        # Your implementation starts
        dx = (np.exp(-self.x)/(1 + np.exp(-self.x))**2)*dprev
        # End of your implementation
        return dx

class Swish(Layer):

    def __init__(self):
        # Dont forget to save x or relumask for using in backward pass
        self.x = None

    def forward(self, x):
        '''
            Forward pass for Swish
            :param x: outputs of previous layer
            :return: Swish activation
        '''
        # Do not forget to copy the output to object to use it in backward pass
        self.x = x.copy()
        # This is used for avoiding the issues related to mutability of python arrays
        x = x.copy()

        # Implement Swish activation
        x = x/(1 + np.exp(-x))
        # End of your implementation
        return x

    def backward(self, dprev):
        '''
            Backward pass of Swish
            :param dprev: gradient of previos layer:
            :return: upstream gradient
        '''
        dx = None
        # Your implementation starts
        dx = ((1 + np.exp(-self.x) + self.x*np.exp(-self.x))/(1 + np.exp(-self.x))**2)*dprev
        # End of your implementation
        return dx
