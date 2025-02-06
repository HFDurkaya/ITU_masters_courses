from DL.bases import LayerWithWeights
import numpy as np
from copy import copy


class Conv2d(LayerWithWeights):
    def __init__(self, in_size, out_size, kernel_size, stride, padding):
        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.x = None
        self.W = np.random.rand(out_size, in_size, kernel_size, kernel_size)
        self.b = np.random.rand(out_size)
        self.db = np.random.rand(out_size, in_size, kernel_size, kernel_size)
        self.dW = np.random.rand(out_size)

    def forward(self, x):
        N, C, H, W = x.shape
        F, C, FH, FW = self.W.shape
        self.x = copy(x)
        # pad X according to the padding setting
        padded_x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding)), 'constant')

        # Calculate output's H and W according to your lecture notes
        out_H = int(((H + 2*self.padding - FH) / self.stride) + 1)
        out_W = int(((W + 2*self.padding - FW) / self.stride) + 1)

        # Initiliaze the output
        out = np.zeros([N, F, out_H, out_W])

        # TO DO: Do cross-correlation by using for loops
        # YOUR CODE STARTS
        
        for i in range(N):
            for j in range(F):
                for k in range(out_H):
                    for l in range(out_W):
                        region = padded_x[i, :, k * self.stride:k * self.stride + FH, l * self.stride:l * self.stride + FW]
                        out[i, j, k, l] = np.sum(region * self.W[j]) + self.b[j]


        # YOUR CODE ENDS
        
        return out

    def backward(self, dprev):
        dx, dw, db = None, None, None
        padded_x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding)), 'constant')
        N, C, H, W = self.x.shape
        F, C, FH, FW = self.W.shape
        _, _, out_H, out_W = dprev.shape

        dx_temp = np.zeros_like(padded_x).astype(np.float32)
        dw = np.zeros_like(self.W).astype(np.float32)
        db = np.zeros_like(self.b).astype(np.float32)


        # YOUR CODE STARTS
        # Calculate db
        db = np.sum(dprev, axis=(0, 2, 3))

        # Calculate dw
        for i in range(N):
            for j in range(F):
                for k in range(out_H):
                    for l in range(out_W):
                        region = padded_x[i, :, k * self.stride:k * self.stride + FH, l * self.stride:l * self.stride + FW]
                        dw[j] += region * dprev[i, j, k, l]

        # Calculate dx_temp
        for i in range(N):  # Loop over batch
            for j in range(F):  # Loop over filters
                for k in range(out_H):  # Loop over output height
                    for l in range(out_W):  # Loop over output width
                        region = slice(k * self.stride, k * self.stride + FH), slice(l * self.stride, l * self.stride + FW)
                        dx_temp[i, :, region[0], region[1]] += self.W[j] * dprev[i, j, k, l]

        # Remove padding to get the final dx
        if self.padding > 0:
            dx = dx_temp[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_temp


        # YOUR CODE ENDS

        self.db = db.copy()
        self.dW = dw.copy()
        return dx, dw, db

