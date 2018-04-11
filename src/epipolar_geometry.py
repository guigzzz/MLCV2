import numpy as np

def get_row_A(pair, pair_prime):
    x, y = pair[0], pair[1]
    x_p, y_p = pair_prime[0], pair_prime[1]
    return [x*x_p, y*x_p, x_p, x*y_p, y*y_p, y_p, x, y, 1]

def build_A(pairs, pairs_prime):
    # A = np.zeros((pairs.shape[0], 9))
    # for i, (p, p_p) in enumerate():
    #     A[i] = get_row_A(p, p_p)
    
    # return A

    return np.array([
        get_row_A(p, p_p) 
        for p, p_p in zip(pairs, pairs_prime)
    ])

def get_fundemental_matrix(pairs, pairs_prime):
    A = build_A(pairs, pairs_prime)
    _, _, v = np.linalg.svd(A, full_matrices = False)
    f_bar = v[-1].reshape(3,3)
    u_hat, s_hat, v_hat = np.linalg.svd(f_bar, full_matrices = False)
    s_hat[-1] = 0
    f_hat = np.dot(u_hat, np.dot(np.diag(s_hat), v_hat))
    return f_hat

def get_normalisation_matrix(points):
    mean = np.mean(points, axis = 0)
    d = np.mean(np.linalg.norm(points - mean, axis = 1)) / np.sqrt(2)

    T = np.array([
        [1/d, 0, - mean[0] / d],
        [0, 1/d, - mean[1] / d],
        [0, 0, 1]
    ])
    return T

def get_fundemental_matrix_normalised(pairs, pairs_prime):
    
    T = get_normalisation_matrix(pairs[:, :2])
    norm_pairs = np.dot(pairs, T.T)
    
    T_prime = get_normalisation_matrix(pairs_prime[:, :2])
    norm_pairs_prime = np.dot(pairs_prime, T_prime.T)
    
    F = get_fundemental_matrix(norm_pairs, norm_pairs_prime)
    
    denormed_F = np.dot(T_prime.T, np.dot(F, T))    
    return denormed_F / denormed_F[2, 2]


def get_epipole(F):
    _,_,V = np.linalg.svd(F)
    e = V[-1] # as S[-1] ~ 0
    return e / e[2]

# https://www.cs.auckland.ac.nz/courses/compsci773s1t/lectures/773-GGpdfs/773GG-FundMatrix-A.pdf

def get_rectification_matrix(F):
    R = np.zeros(F.shape)

    e1 = get_epipole(F)
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(e1, [0, 0, 1])
    e2 /= np.linalg.norm(e2)

    e3 = np.cross(e1, e2)

    R[0] = e1
    R[1] = e2
    R[2] = e3

    return R


def fundemental_accuracy(pairs, pairs_prime, F):
    
    pairwise_prods = np.sum(np.dot(pairs, F.T) * pairs_prime, axis = 1)
    return np.abs(pairwise_prods).mean()


def skew(a):
    """ Skew matrix A such that a x v = Av for any v. """

    return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])


def compute_P_from_fundamental(F):
    """    Computes the second camera matrix (assuming P1 = [I 0]) 
        from a fundamental matrix. """
        
    e = get_epipole(F.T) # left epipole
    Te = skew(e)
    return np.dot(Te,F.T).T


import matplotlib.pyplot as plt
def __drawEpipolarLine(line, bounds):
    x_min, x_max, y_min, y_max = bounds
    space = np.linspace(x_min, x_max, 100)

    eq = (line[2] + line[0] * space) / - line[1]
    sel = np.logical_and(y_min < eq, eq < y_max)
    plt.plot(space[sel], eq[sel])

def drawEpipolarLines(F, points):
    lines = np.dot(points, F.T)
    bounds = plt.gca().get_xbound() + plt.gca().get_ybound()

    for line in lines:
        __drawEpipolarLine(line, bounds)
