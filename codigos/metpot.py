import numpy as np

def autrayleigh(A, q_zero, tol):
    q = q_zero / np.linalg.norm(q_zero, 2)
    theta = tol + 1
    while abs(theta) > tol:
        rho = np.dot(q, A@q)
        q_hat = q
        I = np.eye(A.shape[0])

        z = np.linalg.solve(A-rho*I, q_hat)
        sigma = np.linalg.norm(z, 2)
        q = z / sigma
        theta = np.dot(q, q_hat) / sigma

    return q, rho + theta

