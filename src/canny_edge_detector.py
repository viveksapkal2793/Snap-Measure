import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def auto_canny(image, sigma=0.33, apertureSize=3, L2gradient=False):
    """
    image: the source image (should be grayscale and of type np.uint8)
    sigma: used for lower and upper threshold calculation (default is 0.33)
    apertureSize: to be passed to the cv.Canny as apertureSize (default is 3)
    L2gradient: formula to calculate image gradients (default is False)
    """

    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(
        image.astype(np.uint8),
        lower,
        upper,
        apertureSize=apertureSize,
        L2gradient=L2gradient,
    )
    return edged
