import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def preprocess(img):
    grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grayscale_img_blurred = cv.GaussianBlur(grayscale_img, (5, 5), 0, 0)

    _, thresholded_img = cv.threshold(grayscale_img_blurred, 130, 255, cv.THRESH_BINARY)

    morph_operation_kernel = np.ones(
        (3, 3), dtype=np.uint8
    ) 
    thresholded_img_eroded = cv.erode(
        thresholded_img, morph_operation_kernel, iterations=2
    )

    preprocessed_img = thresholded_img_eroded
    return preprocessed_img
