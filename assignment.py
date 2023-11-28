# Imports: Do not modify!
import datetime
from math import pi, exp
from copy import copy
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# determine correct working directory
import os
wd = os.path.dirname(os.path.abspath(__file__)) + '/'


#
# Question 1 - Random Number Generation
#

# Q1 (a)
def pdf_cauchy(x, mu=0, sigma=1):
    f = 0.0
    #
    f = 1 / (pi * sigma * (1 + ((x - mu) / sigma) ** 2))
    #
    return f


# Q1 (b)
def pdf_laplace(x, mu=0, b=1):
    f = 0
    #
    f = (1 / (2 * b)) * exp(-abs(x - mu) / b)
    #
    return f


# Q1 (c)
def rng_cauchy(n, mu=0, sigma=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.zeros(n)
    U = np.random.rand(n)

    #  #Inverse CDF function
    x = mu + sigma * np.tan(np.pi * (U - 0.5))
    #
    return x


# Q1 (d)
def rng_std_laplace(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.zeros(n)

    #
    i = 0
    M = pi / 2  # M value

    # Acceptence rate alpha
    def alpha(y):
        return pdf_laplace(y) / (M * pdf_cauchy(y))

    while i < n:
        Y = rng_cauchy(1)
        U = np.random.rand()

        if U <= alpha(Y):
            x[i] = Y
            i += 1

    #
    return x


# Q1 (e)
def hist_std_laplace():
    n = 50000
    seed = 34786
    x = rng_std_laplace(n, seed)
    plt.figure()
    #

    # Laplace density values
    z = np.linspace(-7, 7, num=50000)
    f_z = [pdf_laplace(x) for x in z]

    plt.xlim(-7, 7)
    plt.plot(z, f_z, c="orange", linewidth=2)
    plt.hist(x, bins=200, density=True)
    plt.xlabel('$x$')
    plt.ylabel('Density')

    #
    plt.savefig(wd + "Q1_e.pdf")
    plt.show()
    return True


#
# Question 2 - Numerical Optimization
#

# Do not modify!
def load_data_q2():
    data = pd.read_csv(wd+'Q2_data.csv')
    y = data['y'].values
    x = data['x'].values
    return y, x

# Q2 (a)
def ols_estimator(y, x):
    beta_ols = np.zeros(2)
    ## BEGIN ANSWER
    x = x.reshape(-1, 1)
    x = np.insert(x, 0, 1, axis=1)
    y = y.reshape(-1, 1)
    p1 = np.linalg.inv(np.matmul(x.transpose(), x))
    p2 = np.matmul(x.transpose(), y)
    p3 = np.matmul(p1, p2)
    beta_ols = np.array((p3[0][0], p3[1][0]))
    ## END ANSWER
    return beta_ols

# Q2 (b)
def ols_scatterplot():
    y, x = load_data_q2()
    beta_ols = ols_estimator(y, x)
    plt.figure()
    ## BEGIN ANSWER
    b0 = beta_ols[0]
    b1 = beta_ols[1]
    plt.scatter(x, y)
    plt.axline((0, b0), (3, b0 + 3 * b1), c='black')
    ## END ANSWER
    plt.savefig(wd+"Q2_b.pdf")
    return True

# Q2 (c)
def sar(b,y,x):
    sar = 0
    ## BEGIN ANSWER
    sum = np.sum(np.abs(y - b[0] - b[1] * x))
    sar = sum / len(x)
    ## END ANSWER
    return sar

# Q2 (d)
def sar_grad(b,y,x):
    sar_grad = np.zeros(2)
    ## BEGIN ANSWER
    b0 = b[0]
    b1 = b[1]
    n = len(x)
    part0 = np.sum((y - b0 - b1*x)/np.abs(y - b0 - b1*x) * (-1))/n
    part1 = np.sum((y - b0 - b1*x)/np.abs(y - b0 - b1*x) * (-x))/n
    sar_grad[0] = part0
    sar_grad[1] = part1
    ## END ANSWER
    return sar_grad

# Q2 (e)
def gradient_descent(f, grad, b0, y, x, max_iter=50, f_tol=1e-8):
    # initialization
    b = copy(b0)
    fval_prev = np.Inf
    fval = f(b, y, x)

    it = 0
    while (abs(fval_prev - fval) >= f_tol) and (it <= max_iter):
        ## BEGIN ANSWER
        g = grad(b, y, x)
        h = np.negative(g)
        a = 1.0
        c1 = 0.0001
        p = 0.95
        while f(b + a * h, y, x) > f(b, y, x) + c1 * a * np.matmul(h, g.reshape(-1, 1))[0]:
            a = a * p

        b = b + a * h
        fval_prev = fval
        fval = f(b, y, x)
        ## END ANSWER
        it += 1
    return b

# Q2 (f)
def lad_scatterplot():
    y, x = load_data_q2()
    beta_ols = ols_estimator(y, x)
    beta_lad = np.zeros(2)
    b0 = np.array([0.0, 0.0])
    plt.figure()
    ## BEGIN ANSWER
    beta_lad = gradient_descent(sar, sar_grad, b0, y, x)
    beta_ols_0 = beta_ols[0]
    beta_ols_1 = beta_ols[1]
    beta_lad_0 = beta_lad[0]
    beta_lad_1 = beta_lad[1]
    plt.scatter(x, y)
    plt.axline((0, beta_ols_0), (3, beta_ols_0 + 3 * beta_ols_1), c='black')
    plt.axline((0, beta_lad_0), (3, beta_lad_0 + 3 * beta_lad_1), c='red', linestyle='dashed')
    ## END ANSWER
    plt.savefig(wd+"Q2_f.pdf")
    return True

# Q2 (g)
def lad_nelder_mead():
    y, x = load_data_q2()
    beta_lad = np.zeros(2)
    b0 = np.array([0.0, 0.0])
    ## BEGIN ANSWER
    beta_lad = minimize(sar, b0, method='Nelder-Mead',args=(y, x), tol=1e-6).x
    ## END ANSWER
    return beta_lad


#
# Question 3 - Solving Linear Equations & Eigenvalues
#

# Do not modify!
def load_data_q3():
    data = pd.read_csv(wd+'Q3_data.csv')
    y = data['y'].values
    X = data.iloc[:,1:26].values
    return y, X

# Q3 (a.i)
def ridge_M(X, alpha):
    k = X.shape[1]
    M = np.zeros((k,k))
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    return M

# Q3 (a.ii)
def ridge_z(y, X):
    k = X.shape[1]
    z = np.zeros(k)
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    return z

# Q3 (b)
def ridge_estimator(y, X, alpha):
    k = X.shape[1]
    beta_ridge = np.zeros(k)
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    return beta_ridge

# Q3 (c)
def ridge_ev_decomp_XtX(X):
    k = X.shape[1]
    w = np.zeros(k)
    V = np.zeros((k,k))
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    return w, V

# Q3 (d)
def ridge_Minv(X, alpha):
    k = X.shape[1]
    M_inv = np.zeros((k,k))
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    return M_inv

# Q3 (e)
def ridge_estimator_via_inv(y, X, alpha):
    k = X.shape[1]
    beta_ridge = np.zeros(k)
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    return beta_ridge


if __name__ == "__main__":
    print("Running at %s." % datetime.datetime.now().strftime("%H:%M:%S")) # Do not modify!

    #
    # TODO: While developing your solutions, you might want to add commands below
    # that call the functions above for testing purposes.
    #
    # IMPORTANT: Before you submit your code, comment out or delete all function calls
    # that you added here.
    #
