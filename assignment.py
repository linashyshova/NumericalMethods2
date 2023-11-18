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
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    return f

# Q1 (b)
def pdf_laplace(x, mu=0, b=1):
    f = 0
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    return f

# Q1 (c)
def rng_cauchy(n, mu=0, sigma=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.zeros(n)
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    return x

# Q1 (d)
def rng_std_laplace(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.zeros(n)
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    return x


# Q1 (e)
def hist_std_laplace():
    n = 50000
    seed = 34786
    x = rng_std_laplace(n, seed)
    plt.figure()
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    plt.savefig(wd+"Q1_e.pdf")
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
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    return beta_ols

# Q2 (b)
def ols_scatterplot():
    y, x = load_data_q2()
    beta_ols = ols_estimator(y, x)
    plt.figure()
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    plt.savefig(wd+"Q2_b.pdf")
    return True

# Q2 (c)
def sar(b,y,x):
    sar = 0
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    return sar

# Q2 (d)
def sar_grad(b,y,x):
    sar_grad = np.zeros(2)
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
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
        # ...
        # TODO: Add your code here
        # ...
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
    # ...
    # TODO: Add your code here
    # ...
    ## END ANSWER
    plt.savefig(wd+"Q2_f.pdf")
    return True

# Q2 (g)
def lad_nelder_mead():
    y, x = load_data_q2()
    beta_lad = np.zeros(2)
    b0 = np.array([0.0, 0.0])
    ## BEGIN ANSWER
    # ...
    # TODO: Add your code here
    # ...
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
