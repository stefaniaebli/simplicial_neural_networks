import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numpy as np

def normalize(L, half_interval = False):
    assert(sp.isspmatrix(L))
    M = L.shape[0]
    assert(M == L.shape[1])
    topeig = spl.eigsh(L, k=1, which="LM", return_eigenvectors = False)[0]   
    #print("Topeig = %f" %(topeig))

    ret = L.copy()
    if half_interval:
        ret *= 1.0/topeig
    else:
        ret *= 2.0/topeig
        ret.setdiag(ret.diagonal(0) - np.ones(M), 0)

    return ret

def assemble(K, L, x):
    (B, C_in, M) = x.shape
    assert(L.shape[0] == M)
    assert(L.shape[0] == L.shape[1])
    assert(K > 0)
    
    X = []
    for b in range(0, B):
        X123 = []
        for c_in in range(0, C_in):
            X23 = []
            X23.append(x[b, c_in, :].unsqueeze(1)) # Constant, k = 0 term.

            if K > 1:
                X23.append(L.mm(X23[0]))
            for k in range(2, K):
                X23.append(2*(L.mm(X23[k-1])) - X23[k-2])

            X23 = torch.cat(X23, 1)
            assert(X23.shape == (M, K))
            X123.append(X23.unsqueeze(0))

        X123 = torch.cat(X123, 0)
        assert(X123.shape == (C_in, M, K))
        X.append(X123.unsqueeze(0))

    X = torch.cat(X, 0)
    assert(X.shape == (B, C_in, M, K))

    return X
