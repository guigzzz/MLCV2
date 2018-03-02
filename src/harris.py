from scipy.ndimage.filters import gaussian_filter
from sklearn.feature_extraction import image
import numpy as np

def compute_harris_interest_points(image, harris_alpha = 0.05, blur_sigma = 5):
    """
    Computes the harris reponse matrix given an image.
    https://ags.cs.uni-kl.de/fileadmin/inf_ags/opt-ss14/OPT_SS2014_lec02.pdf has a good rundown.
    Checks if the image is greyscale. 
    If not, the image is converted to grayscale using the matlab formula.
    """
    if len(image.shape) == 3:
        image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        
    dx, dy = np.gradient(image)
    square_dx = np.square(dx)
    square_dy = np.square(dy)
    dxdy = dx * dy
    blurred_square_dx = gaussian_filter(square_dx, sigma = blur_sigma)
    blurred_square_dy = gaussian_filter(square_dy, sigma = blur_sigma)
    blurred_dxdy = gaussian_filter(dxdy, sigma = blur_sigma)
    harris = blurred_square_dx * blurred_square_dy - np.square(blurred_dxdy) - harris_alpha * np.square(blurred_square_dx + blurred_square_dy)
    
    return harris

def patch_non_maxima_suppression(img, patch_shape, return_coords = True):
    """
    Applies local maximising to square patches on the input image.
    The center of each patch in the img is compared to the pixels in the patch.
    If the center is not the maximum in the patch, it is ignored (implicitely set to zero)
    if return coords is true:
        return the image coordinates which are the local patch maximums.
    else:
        set to zero the coordinates corresponding to non-maximum responses
        and return the response matrix.
    """
    patches = image.extract_patches(img, patch_shape=patch_shape)

    patch_pixels = np.product(patch_shape)

    selector = np.argmax(
        patches.reshape(*patches.shape[:2], patch_pixels), 
        axis = 2
        ) == (patch_pixels // 2) 

    if return_coords:
        x, y = np.where( selector )

        y += patch_shape[0] // 2
        x += patch_shape[1] // 2
        
        return y, x

    else:
        y, x = np.where(np.logical_not(selector))
        y += patch_shape[0] // 2
        x += patch_shape[1] // 2

        img[y, x] = 0

        return img








# OLD STUFF

# def get_local_maximums(img, patch_shape, threshold):
#     # uses a sliding window, kind of ghetto
#     # should use gradient information instead
#     img_min, img_max = img.min(), img.max()
#     img -= img_min
#     img /= img_max - img_min
    
#     y_stride, x_stride = patch_shape
    
#     interest_points = []
    
#     for i in range(0, img.shape[0], y_stride):
#         for j in range(0, img.shape[1], x_stride):
#             patch = img[i : i + y_stride, j : j + x_stride]
#             y, x = np.unravel_index(np.argmax(patch), patch.shape)
#             if patch[y, x] > threshold:
#                 interest_points.append((i + y, j + x))
                
#     return np.array(interest_points)

# from math import pi
# def angle_to_indices(angle):
#     """
#     takes an angle and returns index offsets
#     used in non-maxima suppression, to get fetch pixels along gradient direction
#     """
#     angle = max(min(angle, pi), -pi) # threshold to -pi; pi

#     if - pi / 8 <= angle < pi / 8 or (7 * pi / 8 <= angle <= pi or - pi <= angle < - 7 * pi / 8): # horizontal
#         return 0, -1, 0, 1

#     if pi / 8 <= angle < 3 * pi / 8 or - 7 * pi / 8 <= angle < - 5 * pi / 8: # same direction as matrix anti-diagonal
#         return 1, -1, -1, 1

#     if 3 * pi / 8 <= angle < 5 * pi / 8 or - 5 * pi / 8 <= angle < - 3 * pi / 8: # vertical
#         return -1, 0, 1, 0

#     if 5 * pi / 8 <= angle < 7 * pi / 8 or - 3 * pi / 8 <= angle < - pi / 8: # same direction as matrix diagonal
#         return -1, -1, 1, 1

#     print('Shouldn\'t get here, angle was: {}'.format(angle / pi)) # default case for debugging


# def non_maxima_suppression(img):

#     dx, dy = np.gradient(img)
#     gradient_args = np.angle(dx.ravel() +  1j * dy.ravel()).reshape(dx.shape)
#     gradient_mag = np.sqrt(np.square(dx) + np.square(dy))

#     for i in range(1, img.shape[0] - 1):
#         for j in range(1, img.shape[1] - 1):
            
#             i1, j1, i2, j2 = angle_to_indices(gradient_args[i, j])

#             if gradient_mag[i + i1, j + j1] > gradient_mag[i, j] or \
#                 gradient_mag[i + i2, j + j2] > gradient_mag[i, j]: # check if not local max

#                 img[i, j] = 0

#     return img



        


