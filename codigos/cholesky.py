import numpy as np

def cholesky(A):
    G = np.zeros(A.shape)
    n = A.shape[0]
    # I = [:i], J = [i:n],[i:]
    for i in range(n):
        G[i, i:] = A[i, i:] - G[:i, i].T @ G[:i, i:]
        #if G[i, i] <= 0:
        #    print("El elemento {}, {} de G es no positivo".format(i, i))
        #    return None
        G[i, i:] = G[i, i:] / np.sqrt(G[i, i])

    return G

