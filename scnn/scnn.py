import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp

import scnn.chebyshev

def coo2tensor(A):
    assert(sp.isspmatrix_coo(A))
    idxs = torch.LongTensor(np.vstack((A.row, A.col)))
    vals = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(idxs, vals, size = A.shape, requires_grad = False)

class SimplicialConvolution(nn.Module):
    def __init__(self, K, C_in, C_out, enable_bias = True, variance = 1.0, groups = 1):
        assert groups == 1, "Only groups = 1 is currently supported."
        super().__init__()

        assert(C_in > 0)
        assert(C_out > 0)
        assert(K > 0)
        
        self.C_in = C_in
        self.C_out = C_out
        self.K = K
        self.enable_bias = enable_bias

        self.theta = nn.parameter.Parameter(variance*torch.randn((self.C_out, self.C_in, self.K)))
        if self.enable_bias:
            self.bias = nn.parameter.Parameter(torch.zeros((1, self.C_out, 1)))
        else:
            self.bias = 0.0
            
    def forward(self, L, x):
        assert(len(L.shape) == 2)
        assert(L.shape[0] == L.shape[1])
                
        (B, C_in, M) = x.shape
     
        assert(M == L.shape[0])
        assert(C_in == self.C_in)

        X = scnn.chebyshev.assemble(self.K, L, x)
        y = torch.einsum("bimk,oik->bom", (X, self.theta))
        assert(y.shape == (B, self.C_out, M))

        return y + self.bias

# This class does not yet implement the
# Laplacian-power-pre/post-composed with the coboundary. It can be
# simulated by just adding more layers anyway, so keeping it simple
# for now.
#
# Note: You can use this for a adjoints of coboundaries too. Just feed
# a transposed D.
class Coboundary(nn.Module):
    def __init__(self, C_in, C_out, enable_bias = True, variance = 1.0):
        super().__init__()

        assert(C_in > 0)
        assert(C_out > 0)

        self.C_in = C_in
        self.C_out = C_out
        self.enable_bias = enable_bias

        self.theta = nn.parameter.Parameter(variance*torch.randn((self.C_out, self.C_in)))
        if self.enable_bias:
            self.bias = nn.parameter.Parameter(torch.zeros((1, self.C_out, 1)))
        else:
            self.bias = 0.0

    def forward(self, D, x):
        assert(len(D.shape) == 2)
        
        (B, C_in, M) = x.shape
        
        assert(D.shape[1] == M)
        assert(C_in == self.C_in)
        
        N = D.shape[0]

        # This is essentially the equivalent of chebyshev.assemble for
        # the convolutional modules.
        X = []
        for b in range(0, B):
            X12 = []
            for c_in in range(0, self.C_in):
                X12.append(D.mm(x[b, c_in, :].unsqueeze(1)).transpose(0,1)) # D.mm(x[b, c_in, :]) has shape Nx1
            X12 = torch.cat(X12, 0)

            assert(X12.shape == (self.C_in, N))
            X.append(X12.unsqueeze(0))

        X = torch.cat(X, 0)
        assert(X.shape == (B, self.C_in, N))
                   
        y = torch.einsum("oi,bin->bon", (self.theta, X))
        assert(y.shape == (B, self.C_out, N))

        return y + self.bias
