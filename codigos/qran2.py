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

