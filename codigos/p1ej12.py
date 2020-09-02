import numpy as np

def cholesky(A):
    # A.shape nos da una tupla con las dimensiones de la matriz
    forma_A = A.shape
    # Vamos a definir n como la cantidad de filas
    n = A.shape[0]
    # Generemos una matriz inicial, llena de ceros
    G = np.zeros(forma_A)
    for i in range(n):
        # Siguiendo el pseudocodigo
        # I = :i y J = i:n
        # En el caso en que i sea 0, la primera operacion
        # del bucle se hará sin la resta
        G[i, i:n] = A[i, i:n] - G[:i, i] @ G[:i, i:n]
        G[i, i:n] = G[i, i:n] / np.sqrt(G[i, i])

    return G

