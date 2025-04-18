import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from calculate_dimensions import calculate_dimensions
from matplotlib_imshow import matplotlib_imshow

def visualize_detections(src, pts, return_dimensions=False):
    """
    src: image on which to draw the rotated rectangle and display the calculated width and height.
         should be BGR type image.
    pts: points array that outlines the object (to call the calculate_dimensions() function)
    return_dimensions: Whether to return calculated dimensions along with the image

    Returns:
        If return_dimensions is False: the modified image (BGR type)
        If return_dimensions is True: tuple (modified image, (width_cm, height_cm))
    """
    src = src.copy()
    box2d_obj, (w, h) = calculate_dimensions(pts)
    corner_points = cv.boxPoints(box2d_obj).astype(np.int32)
    cv.polylines(src, [corner_points], isClosed=True, color=(0, 255, 0), thickness=1)
    
    # Convert dimensions from mm to cm (divide by 10)
    width_cm = round(w/10, 1)
    height_cm = round(h/10, 1)
    
    # Create a small semi-transparent background for better text visibility
    overlay = src.copy()
    cv.rectangle(overlay, (5, 5), (130, 55), (255, 255, 255), -1)
    cv.addWeighted(overlay, 0.6, src, 0.4, 0, src)
    
    cv.putText(
        src,
        f"Width: {width_cm} cm",
        (10, 25),
        cv.FONT_HERSHEY_SIMPLEX,  # Changed to a cleaner font
        0.5,
        (0, 0, 0),
        1,
    )
    cv.putText(
        src,
        f"Height: {height_cm} cm",
        (10, 50),
        cv.FONT_HERSHEY_SIMPLEX,  # Changed to a cleaner font
        0.5,
        (0, 0, 0),
        1,
    )

    if return_dimensions:
        return src, (width_cm, height_cm)
    else:
        return src