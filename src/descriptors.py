import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction import image


def histogram_along_rows(data, bins):
    # https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis
    # Setup bins and determine the bin location for each element for the bins
    n_bins = len(bins)
    N = data.shape[-1]
    
    data2D = data.reshape(-1,N)
    idx = np.searchsorted(bins, data2D,'right')-1

    # Some elements would be off limits, so get a mask for those
    bad_mask = (idx==-1) | (idx==n_bins)

    # We need to use bincount to get bin based counts. To have unique IDs for
    # each row and not get confused by the ones from other rows, we need to 
    # offset each row by a scale (using row length for this).
    scaled_idx = n_bins*np.arange(data2D.shape[0])[:,None] + idx

    # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
    limit = n_bins*data2D.shape[0]
    scaled_idx[bad_mask] = limit

    # Get the counts and reshape to multi-dim
    counts = np.bincount(scaled_idx.ravel(), minlength=limit+1)[:-1]
    counts.shape = data.shape[:-1] + (n_bins,)
    return counts


def get_patch_histograms_from_interest_points_vec(img, interest_points, bins, patch_size=11):

    patches = image.extract_patches(img, patch_shape=(patch_size, patch_size))

    y, x = interest_points
    y_shape, x_shape = patches.shape[:2]

    # only use interest points that can be associated to a full
    # patch_size * patch_size descriptor
    selector = np.logical_and(y < y_shape, x < x_shape)
    y = y[selector]
    x = x[selector]

    selected_patches = patches[y - patch_size // 2, x - patch_size // 2]

    # compute histograms for every selected image patch
    return histogram_along_rows(selected_patches.reshape(selected_patches.shape[0], -1), bins), (y, x)

def match_descriptors_nn(descriptors1, descriptors2):
    dists = euclidean_distances(descriptors1, descriptors2)
    return np.argmin(dists, axis = 1)










# OLD STUFF

# def block_normalisation(image_patches, block_size = 2):

#     for i in range(0, image_patches.shape[0], block_size):
#         for j in range(0, image_patches.shape[1], block_size):
#             block = image_patches[i : i + block_size, j : j + block_size]
#             block_norm = np.linalg.norm(block.reshape(-1, image_patches.shape[-1]))

#             image_patches[i : i + block_size, j : j + block_size] /= np.sqrt(block_norm + 1e-10) 

#     return image_patches

# def get_patch_histograms_from_interest_points(image, interest_points, num_bins=8, patch_size=8):

#     bins = np.histogram(np.arange(255), bins=num_bins)[1]

#     histograms = np.zeros((len(interest_points), num_bins))
#     num_valid_interests = 0
#     max_y, max_x = image.shape[:2]

#     for y, x in interest_points:
#         patch_start_y, patch_start_x = y - patch_size // 2, x - patch_size // 2
#         patch_end_y, patch_end_x = y + patch_size // 2, x + patch_size // 2

#         if patch_start_y > 0 and patch_start_x > 0 and patch_end_y < max_y and patch_end_x < max_x:
#             patch = image[patch_start_y : patch_end_y, patch_start_x : patch_end_x]
#             histograms[num_valid_interests] = np.histogram(patch, bins=bins)[0]
#             num_valid_interests += 1

#     return histograms[:num_valid_interests]

# def divide_to_cells(image, kernel_size):
#     y_diff = kernel_size - image.shape[0] % kernel_size
#     x_diff = kernel_size - image.shape[1] % kernel_size
    
#     dt = type(image[0, 0])

#     image = np.concatenate(
#         [image, np.zeros((y_diff, image.shape[1]), dtype=dt)], 
#         axis = 0)

#     image = np.concatenate(
#         [image, np.zeros((image.shape[0], x_diff), dtype=dt)], 
#         axis = 1)

#     new_height = image.shape[0] // kernel_size
    
#     return image.reshape(new_height, kernel_size, -1, kernel_size).swapaxes(1, 2)

# def divide_to_cells_rgb(image, kernel_size):
#     return np.concatenate([
#         divide_to_cells(image[:, :, 0], kernel_size)[..., np.newaxis],
#         divide_to_cells(image[:, :, 1], kernel_size)[..., np.newaxis],
#         divide_to_cells(image[:, :, 2], kernel_size)[..., np.newaxis],
#     ], axis = -1)

# def show_celled_image(image):
#     plt.figure()
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             plt.subplot(*image.shape[:2], i * image.shape[1] + j + 1)
#             plt.imshow(image[i, j])
#             plt.axis('off')
        
#     plt.show()

# def get_histogram_array(descriptors, bins):
#     histograms = np.zeros((descriptors.shape[0], bins.shape[0] - 1))
#     for i, d in enumerate(descriptors):
#         histograms[i] = np.histogram(d, bins=bins)[0]
        
#     return histograms
    
# def get_patch_histograms(image, num_bins = 8, patch_size = 8):
    
#     _, bins = np.histogram(np.arange(255), bins=num_bins)
    
#     if len(image.shape) > 2:
#         celled_image = divide_to_cells_rgb(image, kernel_size = patch_size)
#     else:
#         celled_image = divide_to_cells(image, kernel_size = patch_size)
        

#     histograms = get_histogram_array(
#         celled_image.reshape(-1, *celled_image.shape[-2:]), 
#         bins=bins
#     ).reshape(*celled_image.shape[:2], num_bins)
    
#     return histograms



# if __name__ == '__main__':
#     wild_cat = plt.imread('wild_cat.jpg')
#     wild_cat_cells = divide_to_cells_rgb(wild_cat, 100)
#     show_celled_image(wild_cat_cells)

#     concat_rows = np.concatenate(wild_cat_cells, axis = 1)
#     concat_cols = np.concatenate(concat_rows, axis = 1)
#     plt.figure()
#     plt.imshow(concat_cols[:263, :300])
#     assert (concat_cols[:263, :300] == wild_cat).all() == True
#     assert type(concat_cols[0, 0, 0]) == type(wild_cat[0, 0, 0])
