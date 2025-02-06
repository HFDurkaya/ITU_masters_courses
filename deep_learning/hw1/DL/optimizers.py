import numpy as np
from bases import LayerWithWeights

class VanillaSGDOptimizer(object):
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr

    def optimize(self):
        for m in self.model:
            if isinstance(m, LayerWithWeights):
                self._optimize(m)

    def _optimize(self, m):
        '''
            Optimizer for VanillaSGDOptimizer
            :param m: module with weights to optimize
        '''
        # Your implementation starts
        m.W -= self.lr*m.dW
        m.b -= self.lr*m.db
        # End of your implementation
  
class RMSprop(VanillaSGDOptimizer):
    def __init__(self, model, lr=1e-3, beta=.9, epsilon=1e-8):
        self.model = model
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.s = {m: {"W": 0, "b": 0} for m in model}

    def _optimize(self, m):
        '''
            Optimizer for RMSprop
            :param m: module with weights to optimize
        '''

        
        #### Your implementation starts ####
        
        grad_w = m.dW
        grad_b = m.db  
        
        # Updating weights and biases using RMSprop
        self.s[m]["W"] = self.beta * self.s[m]["W"] + (1 - self.beta) * (grad_w ** 2)
        self.s[m]["b"] = self.beta * self.s[m]["b"] + (1 - self.beta) * (grad_b ** 2)
        
        m.W -= self.lr * grad_w / (np.sqrt(self.s[m]["W"]) + self.epsilon)
       
        m.b -= self.lr * grad_b / (np.sqrt(self.s[m]["b"]) + self.epsilon)
        
        #### End of your implementation ####


class SGDWithMomentum(VanillaSGDOptimizer):
    def __init__(self, model, lr=1e-3, mu=.5, regularization_str=1e-5):
        self.model = model
        self.reg = regularization_str
        self.lr = lr
        self.mu = mu
        # Save velocities for each model in a dict and use them when needed.
        self.velocities = {m: {"W": 0, "b": 0} for m in model}

    def _optimize(self, m):
        '''
            Optimizer for SGDMomentum
            Do not forget to add L2 regularization!
            :param m: module with weights to optimize
        '''
        # Your implementation starts
        self.velocities[m]["W"] = (self.mu)*self.velocities[m]["W"] + (1-self.mu)*(m.dW) 
        self.velocities[m]["b"] = (self.mu)*self.velocities[m]["b"] + (1-self.mu)*(m.db) 

        m.W = m.W - self.lr*(self.velocities[m]["W"] + self.reg*(m.W))
        m.b = m.b - self.lr*(self.velocities[m]["b"] + self.reg*(m.b))
        # End of your implementation
