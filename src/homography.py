
import numpy as np

def get_row_pair_A(pair, pair_prime):
    x, y = pair[0], pair[1]
    x_p, y_p = pair_prime[0], pair_prime[1]
    return [
        [0, 0, 0, -x, -y, -1, y_p*x, y_p*y, y_p],
        [x, y, 1, 0, 0, 0, -x_p*x, -x_p*y, -x_p]
    ]

def build_A(pairs, pairs_prime):
    A = np.zeros((pairs.shape[0] * 2, 9))
    for i, (p, p_prime) in enumerate(zip(pairs, pairs_prime)):
        A[ i*2 : (i+1)*2 ] = get_row_pair_A(p, p_prime)
    
    return A

def get_homography_matrix(pairs, pairs_prime):
    """
    Use Direct Linear Transform to compute the homography given pairs of points 
    """
    A = build_A(pairs, pairs_prime)
    _, _, v = np.linalg.svd(A)
    H = (v[-1] / v[-1, -1]).reshape(3, 3)
    return H

def get_normalisation_matrix(points):
    mean = np.mean(points, axis = 0)
    d = np.mean(np.linalg.norm(points - mean, axis = 1)) / np.sqrt(2)

    T = np.array([
        [1/d, 0, - mean[0] / d],
        [0, 1/d, - mean[1] / d],
        [0, 0, 1]
    ])
    return T

def get_normalised_homography_matrix(pairs, pairs_prime):
    """
    Normalised Direct Linear Transform
    """
    T = get_normalisation_matrix(pairs[:, :2])
    norm_pairs = np.dot(pairs, T.T)
    
    T_prime = get_normalisation_matrix(pairs_prime[:, :2])
    norm_pairs_prime = np.dot(pairs_prime, T_prime.T)
    
    H = get_homography_matrix(norm_pairs, norm_pairs_prime)
    
    denormed_H = np.dot(np.linalg.inv(T_prime), np.dot(H, T))    
    return denormed_H


def ransac_homography(points, points_prime, error_threshold=1, max_iter=1000):
    """
    Adaptive RANSAC algorithm applied to homography calculation
    http://www.uio.no/studier/emner/matnat/its/UNIK4690/v16/forelesninger/lecture_4_3-estimating-homographies-from-feature-correspondences.pdf
    Uses the normalised DLT to compute the homography matrix for the resulting inliers.
    """
    N = 1000000
    Sin = np.zeros(len(points))
    iterations = 0

    while N > iterations and iterations < max_iter:
        indices = np.random.choice(len(points), size=4, replace=False)
        iter_p = points[indices]
        iter_p_prime = points_prime[indices]
        
        H = get_homography_matrix(iter_p, iter_p_prime)
        
        errors = np.linalg.norm(H.dot(points.T).T - points_prime, axis=1) + \
                np.linalg.norm(points - np.linalg.inv(H).dot(points_prime.T).T, axis=1)

        selector = errors < error_threshold

        if np.sum(selector) == len(points): 
            Sin = selector 
            break
            
        if np.sum(selector) > np.sum(Sin):
            print(errors.mean())            
            
            Sin = selector
            w = np.sum(Sin) / len(points)
            N = np.log(1 - 0.99) / np.log(1 - w ** 4)
        
        iterations += 1

    return get_normalised_homography_matrix(points[Sin], points_prime[Sin])
    