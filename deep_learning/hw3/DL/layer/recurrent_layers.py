from .layers_with_weights import LayerWithWeights
import numpy as np


class RNNLayer(LayerWithWeights):
    """ Simple RNN Layer - only calculates hidden states """

    def __init__(self, in_size, out_size):
        """ RNN Layer constructor
        Args:
            in_size: input feature dimension - D
            out_size: hidden state dimension - H
        """
        self.in_size = in_size
        self.out_size = out_size
        self.Wx = np.random.rand(in_size, out_size)
        self.Wh = np.random.rand(out_size, out_size)
        self.b = np.random.rand(out_size)
        self.cache = None
        self.grad = {'dx': None, 'dh0': None, 'dWx': None, 'dWh': None, 'db': None}

    def forward_step(self, x, prev_h):
        """ Forward pass for a single timestep
        Args:
            x: input, of shape (N, D)
            prev_h: previous hidden state, of shape (N, H)
        Returns:
            next_h: next hidden state, of shape (N, H)
            cache: Values necessary for backpropagation, tuple
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        
        
        #Computing next hidden state using tanh activation.
        next_h = np.tanh(self.b + x @ self.Wx + prev_h @ self.Wh)
        #Storing values needed for backpropagation in cache.
        cache = (x, prev_h, next_h)
        #Returning next hidden state and cache.
        return next_h, cache
        

    def forward(self, x, h0):
        """ Forward pass for the whole data sequence (of length T) of size minibatch N
        Values necessary in backpropagation need to be kept in self.cache as a list
        Args:
            x: input, of shape (N, T, D)
            h0: initial hidden state, of shape (N, H)
        Returns:
            h: hidden states of whole sequence, of shape (N, T, H)
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        
        #Getting dimensions of input
        N, T, D = x.shape
        H = self.out_size

        #Initializing hidden states and cache
        h = np.zeros((N, T, H))
        prev_h = h0
        self.cache = []

        #Iterating over each timestep
        for t in range(T):
            #Computing next hidden state and cache for current timestep
            prev_h, step_cache = self.forward_step(x[:, t, :], prev_h)
            #Storing the hidden state
            h[:, t, :] = prev_h
            #Storing the cache
            self.cache.append(step_cache)

        #Returning all hidden states
        return h

    def backward_step(self, dnext_h, cache):
        """ Backward pass for a single timestep
        Args:
            dnext_h: gradient of loss with respect to
                     hidden state, of shape (N, H)
            cache: necessary values from last forward pass
        Returns:
            dx: gradients of input, of shape (N, D)
            dprev_h: gradients of previous hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, H)
            dWh: gradients of weights Wh, of shape (H, H)
            db: gradients of bias b, of shape (H,)
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        
        #Unpacking cache values.
        x, prev_h, next_h = cache
        
        #Computing gradient of the tanh activation function.
        dtanh = dnext_h * (1 - next_h**2)
        
        #Computing gradients of bias.
        db = np.sum(dtanh, axis=0)
        
        #Computing gradients of input-to-hidden weights.
        dWx = x.T @ dtanh
        
        #Computing gradients of hidden-to-hidden weights.
        dWh = prev_h.T @ dtanh
        
        #Computing gradients of input.
        dx = dtanh @ self.Wx.T
        
        #Computing gradients of previous hidden state.
        dprev_h = dtanh @ self.Wh.T
        
        #Returning all computed gradients.
        return dx, dprev_h, dWx, dWh, db

    def backward(self, dh):
        """ Backward pass for whole sequence
        Necessary data for backpropagation should be obtained from self.cache
        Args:
            dh: gradients of all hidden states, of shape (N, T, H)
        Calculates gradients and saves them to the dictionary self.grad
        self.grad = {
            dx: gradients of inputs, of shape (N, T, D)
            dh0: gradients of initial hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, H)
            dWh: gradients of weights Wh, of shape (H, H)
            db: gradients of bias b, of shape (H,)
            }
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        
        
        #Getting dimensions of gradients.
        N,T,H = dh.shape
        D = self.in_size

        #Initializing gradients.
        dx = np.zeros((N,T,D))
        dprev_h = np.zeros((N,H))
        dWx = np.zeros((D,H))
        dWh = np.zeros((H,H))
        db = np.zeros(H)

        #Iterating over each timestep in reverse order.
        for t in reversed(range(T)):
            #Computing gradients for the current timestep.
            step_dx,dprev_h,step_dWx,step_dWh,step_db = self.backward_step(dh[:,t,:] + dprev_h,self.cache[t])
            #Storing the gradients of input.
            dx[:,t,:] = step_dx
            #Accumulating the gradients of weights and bias.
            dWx += step_dWx
            dWh += step_dWh
            db += step_db

        #Saving the computed gradients in the grad dictionary.
        self.grad = {'dx': dx,'dh0': dprev_h,'dWx': dWx,'dWh': dWh,'db': db}
        return self.grad


class LSTMLayer(LayerWithWeights):
    """ Simple LSTM Layer - only calculates hidden states and cell states """

    def __init__(self, in_size, out_size):
        """ LSTM Layer constructor
        Args:
            in_size: input feature dimension - D
            out_size: hidden state dimension - H
        """
        self.in_size = in_size
        self.out_size = out_size
        self.Wx = np.random.rand(in_size, 4 * out_size)
        self.Wh = np.random.rand(out_size, 4 * out_size)
        self.b = np.random.rand(4 * out_size)
        self.cache = None
        self.grad = {'dx': None, 'dh0': None, 'dWx': None,
                     'dWh': None, 'db': None}

    def forward_step(self, x, prev_h, prev_c):
        """ Forward pass for a single timestep
        Args:
            x: input, of shape (N, D)
            prev_h: previous hidden state, of shape (N, H)
            prev_c: previous cell state, of shape (N, H)
        Returns:
            next_h: next hidden state, of shape (N, H)
            next_c: next cell state, of shape (N, H)
            cache: Values necessary for backpropagation, tuple
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        
        
        H = self.out_size

        #Computing the activation vector.
        a = self.b + x @ self.Wx + prev_h @ self.Wh
        #Splitting the activation vector into input, forget, output gates and candidate.
        ai, af, ao, ag = np.split(a, 4, axis=1)

        #Applying sigmoid activation to input, forget, and output gates.
        input_gate = 1 / (1 + np.exp(-ai))
        forget_gate = 1 / (1 + np.exp(-af))
        output_gate = 1 / (1 + np.exp(-ao))
        #Applying tanh activation to candidate.
        candidate = np.tanh(ag)

        #Computing the next cell state.
        next_c = forget_gate * prev_c + input_gate * candidate
        #Computing the next hidden state.
        next_h = output_gate * np.tanh(next_c)

        #Storing values needed for backpropagation in cache.
        cache = (x, prev_h, prev_c, input_gate, forget_gate, output_gate, candidate, next_c)
        return next_h, next_c, cache

    def forward(self, x, h0):
        """ Forward pass for the whole data sequence (of length T) of size minibatch N
        Values necessary in backpropagation need to be kept in self.cache as a list
        Cell state should be initialized to 0.
        Args:
            x: input, of shape (N, T, D)
            h0: initial hidden state, of shape (N, H)
        Returns:
            h: hidden states of whole sequence, of shape (N, T, H)
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        
        #Getting dimensions of input.
        N, T, D=x.shape
        H = self.out_size

        #Initializing hidden states,cell states,and cache.
        h = np.zeros((N, T, H))
        prev_h = h0
        prev_c = np.zeros((N, H))
        self.cache = []

        #Iterating over each timestep.
        for t in range(T):
            #Computing next hidden state,cell state,and cache for current timestep.
            prev_h, prev_c, step_cache = self.forward_step(x[:,t,:], prev_h, prev_c)
            #Storing the hidden state.
            h[:,t,:] = prev_h
            #Storing the cache.
            self.cache.append(step_cache)

        #Returning all hidden states.
        return h

    def backward_step(self, dnext_h, dnext_c, cache):
        """ Backward pass for a single timestep
        Args:
            dnext_h: gradient of loss with respect to
                     hidden state, of shape (N, H)
            dnext_c: gradient of loss with respect to
                     cell state, of shape (N, H)
            cache: necessary values from last forward pass
        Returns:
            dx: gradients of input, of shape (N, D)
            dprev_h: gradients of previous hidden state, of shape (N, H)
            dprev_c: gradients of previous cell state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 4H)
            dWh: gradients of weights Wh, of shape (H, 4H)
            db: gradients of bias b, of shape (4H,)
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        
        #Unpacking cache values.
        x, prev_h, prev_c, input_gate, forget_gate, output_gate, candidate, next_c = cache

        #Computing gradient of next cell state.
        dnext_c += dnext_h * output_gate * (1 - np.tanh(next_c)**2)

        #Computing gradients of gates and candidate.
        dforget_gate = dnext_c * prev_c
        dprev_c = dnext_c * forget_gate
        dinput_gate = dnext_c * candidate
        dcandidate = dnext_c * input_gate

        #Computing gradients of activation functions.
        dai = dinput_gate * input_gate * (1 - input_gate)
        daf = dforget_gate * forget_gate * (1 - forget_gate)
        dao = dnext_h * np.tanh(next_c) * output_gate * (1 - output_gate)
        dag = dcandidate * (1 - candidate**2)

        #Concatenating gradients of activations.
        da = np.hstack((dai, daf, dao, dag))

        #Computing gradients of bias.
        db = np.sum(da, axis=0)
        #Computing gradients of input-to-hidden weights.
        dWx = x.T @ da
        #Computing gradients of hidden-to-hidden weights.
        dWh = prev_h.T @ da
        #Computing gradients of input.
        dx = da @ self.Wx.T
        #Computing gradients of previous hidden state.
        dprev_h = da @ self.Wh.T

        #Returning all computed gradients.
        return dx, dprev_h, dprev_c, dWx, dWh, db

    def backward(self, dh):
        """ Backward pass for whole sequence
        Necessary data for backpropagation should be obtained from self.cache
        Args:
            dh: gradients of all hidden states, of shape (N, T, H)
        Calculates gradients and saves them to the dictionary self.grad
        self.grad = {
            dx: gradients of inputs, of shape (N, T, D)
            dh0: gradients of initial hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 4H)
            dWh: gradients of weights Wh, of shape (H, 4H)
            db: gradients of bias b, of shape (4H,)
            }
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        
        
        #Getting dimensions of gradients.
        N, T, H = dh.shape
        D = self.in_size

        #Initializing gradients.
        dx = np.zeros((N, T, D))
        dprev_h = np.zeros((N, H))
        dprev_c = np.zeros((N, H))
        dWx = np.zeros((D, 4*H))
        dWh = np.zeros((H, 4*H))
        db = np.zeros(4*H)

        #Iterating over each timestep in reverse order.
        for t in reversed(range(T)):
            #Computing gradients for the current timestep.
            step_dx, dprev_h, dprev_c, step_dWx, step_dWh, step_db = self.backward_step(dh[:,t,:] + dprev_h, dprev_c, self.cache[t])
            #Storing the gradients of input.
            dx[:,t,:] = step_dx
            #Accumulating the gradients of weights and bias.
            dWx += step_dWx
            dWh += step_dWh
            db += step_db

        #Saving the computed gradients in the grad dictionary.
        self.grad = {'dx':dx, 'dh0':dprev_h, 'dWx':dWx, 'dWh':dWh, 'db':db}
        return self.grad