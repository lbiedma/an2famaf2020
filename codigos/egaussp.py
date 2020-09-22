import numpy as np

def egaussp(A, b):
    nav = np.min([A.shape[0] - 1, A.shape[1]])
    U = A.copy()
    y = b.copy()
    for k in range(nav):
        l = np.argmax(np.abs(U[k:, k])) + k
        if k != l:
            U[[k, l], k:] = U[[l, k], k:]
            y[[k, l]] = y[[l, k]]
        v = U[k+1:,k] / U[k, k]
        U[k+1:, k] = 0.0
        U[k+1:, k+1:] = U[k+1:, k+1:] - np.outer(v, U[k, k+1:])
        y[k+1:] = y[k+1:] - v * y[k]

    return U, y

