import numpy as np
import re
import matplotlib.pyplot as plt

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                        dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                        count=int(width)*int(height),
                        offset=len(header)
                        ).reshape((int(height), int(width)))


def get_gradient_magnitude_and_argument(image):
    dx, dy = np.gradient(image)
    gradient_magnitude = np.sqrt(np.square(dx) + np.square(dy))
    gradient_argument = np.angle(dx.ravel() +  1j * dy.ravel()).reshape(dx.shape)
    return gradient_magnitude, gradient_argument

def plot_side_by_side(im1, im2):
    plt.figure(figsize=[10, 20])
    plt.subplot(1, 2, 1)
    plt.imshow(im1)
    plt.subplot(1, 2, 2)
    plt.imshow(im2)
    plt.show()