import numpy as np

def sol_cauchy(A, b, x0, eps, mit):

    r = b - A.dot(x0)
    sigma = np.linalg.norm(r, 2)
    
    for k in range(mit):
        v = A.dot(r)
        t = sigma**2 / np.inner(r, v)
        x = x0 + t * r
        r = r - t * v
        sigma = np.linalg.norm(r)
        if sigma < eps:
            break
        x0 = x.copy()

    return x


def sol_gastinel(A, b, x0, eps, mit):

    r = b - A.dot(x0)
    d = np.sign(r)

    for k in range(mit):
        v = A.dot(d)
        t = np.linalg.norm(r, 1) / d.dot(v)

        x = x0 + t * d
        r = r - t * v
        d = np.sign(r)

        sigma = np.linalg.norm(r, 1)
        if sigma < eps:
            break
        x0 = x.copy()

    return x

