import numpy as np


def wexp(x: float, W, D):
    res = 0
    for w, d in zip(W, D):
        res += w * np.exp(-d * x)
    return res


def quadr_error(WD: np.array, x: np.array, y: np.array):
    W = WD[:WD.size // 2]
    D = WD[WD.size // 2:]
    n = W.size
    m = x.size
    res = 0
    for xi, yi in zip(x, y):
        res += (wexp(xi, W, D) - yi) ** 2
    return res


def jac(WD: np.array, x: np.array, y: np.array):
    W = WD[:WD.size // 2]
    D = WD[WD.size // 2:]
    n = W.size
    res = np.zeros(2 * n)
    for w, d, j in zip(W, D, range(n)):
        for xi, yi in zip(x, y):
            res[j] += 2 * ((wexp(xi, W, D) - yi) * np.exp(-d * xi)).sum()
            res[j + n] += -2 * ((wexp(xi, W, D) - yi) * np.exp(-d * xi) * xi * w).sum()
    return res


def hess(WD: np.array, x: np.array, y: np.array):
    W = WD[:WD.size // 2]
    D = WD[WD.size // 2:]
    n = W.size
    m = x.size
    res = np.zeros((2 * n, 2 * n))

    # левый верхний и правый нижний квадраты
    for i in range(n):
        for j in range(n):
            for k in range(m):
                res[i][j] += 2 * np.exp(-(D[i] + D[j]) * x[k])
                res[i + n][j + n] += 2 * W[i] * W[j] * (x[k]**2) * np.exp(-(D[i]+D[j])*x[k])
                if i == j:
                    res[i+n][j+n] += 2*W[i]*(x[k]**2)*(wexp(x[k], W, D) - y[k]) * np.exp(-D[i]*x[k])

    # левый нижний и правый верхний квадраты
    for i in range(n):
        for j in range(n):
            for k in range(m):
                res[i][j + n] += -2 * W[j] * x[k] * np.exp(-(D[i] + D[j]) * x[k])
                if i == j:
                    res[i][j + n] += -2 * x[k] * (wexp(x[k], W, D) - y[k]) * np.exp(-D[j] * x[k])

            res[j + n][i] = res[i][j + n]
    return res

