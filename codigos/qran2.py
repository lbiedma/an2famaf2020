import numpy as np

def givens(x1,x2):
    '''
    c, s = givens(x1, x2)
    Calcula el coseno y seno para la rotación de Givens
    que hace (x1,x2) -> (y,0).
    '''
    c = 1.
    s = 0.
    ax1 = abs(x1)
    ax2 = abs(x2)
    if ax1 + ax2 > 0:
        if ax2 > ax1:
            tau = -x1/x2
            s = -np.sign(x2)/np.sqrt(1 + tau**2)
            c = tau*s
        else:
            tau = -x2/x1
            c = np.sign(x1)/np.sqrt(1 + tau**2)
            s = tau*c
    return c, s


def house(x):
    '''
    u, rho = house(x)
    Calcula u y rho tal que Q = I - rho u u^T
    cumple Qx = \|x\|_2 e^1
    '''
    n = len(x)
    rho = 0
    u = x.copy()
    u[0] = 1.

    if n == 1:
        sigma = 0
    else:
        sigma = np.sum(x[1:]**2)

    if sigma>0 or x[0]<0:
        mu = np.sqrt(x[0]**2 + sigma)
        if x[0]<=0:
            gamma = x[0] - mu
        else:
            gamma = -sigma/(x[0] + mu)

        rho = 2*gamma**2/(gamma**2 + sigma)
        u = u/gamma
        u[0] = 1

    return u, rho


def qrgivens(A):
    m, n = A.shape
    p = min(m-1, n)
    Q = np.eye(m)
    R = A.copy()
    for jdx in range(p):
        for idx in range(jdx+1, m):
            c, s = givens(R[jdx, jdx], R[idx, jdx])
            G = np.array([[c, -s], [s, c]])
            R[[jdx, idx], :] = G @ R[[jdx, idx], :]
            Q[:, [jdx, idx]] = Q[:, [jdx, idx]] @ G.T
        
    
    if (m <= n) and (R[m, m] < 0):
        R[m, m:] = -R[m, m:]
        Q[:, m] = -Q[:, m]

    return Q, R


def qrhouseholder(A):
    m, n = A.shape
    Q = np.eye(m)
    p = min(m, n)
    R = A.copy()
    for jdx in range(p):
        u, rho = house(R[jdx:, jdx])
        w = rho * u
        R[jdx:, jdx:] = R[jdx:, jdx:] - np.outer(w, u @ R[jdx:, jdx:])
        Q[:, jdx:] = Q[:, jdx:] - Q[:, jdx:] @ np.outer(w, u)
    
    return Q, R


def qrhholderp(A0):
    '''
    Q, R, P = qrhholderp(A)
    Realiza la descomposición QR con permutaciones de A
    utilizando reflexiones de Householder,
    obteniendo que AP = QR
    '''
    A = A0.copy()
    A = A.astype('float64')
    m, n = A.shape
    Q = np.eye(m)
    P = np.eye(n)
    c = np.sum(A**2, 0)
    p = min([m,n])
    for j in range(p):
        l = np.argmax(c[j:])
        l = j + l
        if c[l]==0:
            return Q, A, P
        else:
            A[:,[j,l]] = A[:,[l,j]]
            P[:,[j,l]] = P[:,[l,j]]
            c[[j,l]] = c[[l,j]]
            u, rho = house(A[j:, j])
            w = rho*u
            A[j:, j:] = A[j:, j:] - np.outer(w, u.T @ A[j:, j:])
            Q[:, j:] = Q[:, j:] - np.outer(Q[:, j:] @ w, u)
            c[j:] = c[j:] - A[j, j:]**2

    return Q, A, P

