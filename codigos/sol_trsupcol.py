def sol_trsupcol(A, b):
    """
    Funcion que resuelve Ax = b con A triangular superior,
    aplicando resolución por columnas
    """
    # Copiamos el vector para poder realizarle cambios
    x = b.copy()
    n = len(b)
    # Comenzamos obteniendo el último elemento de x.
    x[n-1] = b[n-1] / A[n-1, n-1]
    # Avanzamos por el resto de las columnas de la matriz
    for i in range(n-2, -1, -1):
        # Modificamos b de acuerdo a la columna i+1 de A
        b[:i+1] = b[:i+1] - A[:i+1, i+1] * x[i+1]
        # Dividimos por lo restante
        x[i] = b[i] / A[i, i]
    
    return x
