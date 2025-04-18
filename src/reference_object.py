import cv2
import numpy as np
import math

def detect_reference_object(image, reference_contours=None, min_area=1000, max_area=None):
    """
    Detect a reference object in the image.
    
    Args:
        image: Input image
        reference_contours: Pre-detected contours to search within (optional)
        min_area: Minimum area for the reference object
        max_area: Maximum area for the reference object (if None, uses 1/4 of image area)
        
    Returns:
        contour: Contour of the reference object
    """
    if max_area is None:
        max_area = image.shape[0] * image.shape[1] // 4  # 1/4 of image area
    
    # If no contours provided, find them
    if reference_contours is None:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Threshold the image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours = reference_contours
    
    # Find the reference object (assuming it's one of the largest objects with reasonable size)
    valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    
    # Sort by area (largest first)
    valid_contours.sort(key=cv2.contourArea, reverse=True)
    
    if not valid_contours:
        return None
        
    return valid_contours[0]  # Return the largest valid contour

def calculate_pixels_per_metric(reference_contour, reference_dimensions):
    """
    Calculate pixels per metric unit (e.g., cm) using a reference object.
    
    Args:
        reference_contour: Contour of the reference object
        reference_dimensions: (width, height) in real-world units (e.g., cm)
        
    Returns:
        pixels_per_unit: Conversion factor for measurements
    """
    # Find the minimum area rectangle that encloses the contour
    rect = cv2.minAreaRect(reference_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Get width and height of the reference object in pixels
    width_px, height_px = rect[1]
    
    # Make sure width is the longer dimension and height is the shorter
    if width_px < height_px:
        width_px, height_px = height_px, width_px
        
    # Compute pixels per unit based on the known reference dimensions
    ref_width, ref_height = reference_dimensions
    
    # If reference_dimensions are provided in a specific order (e.g., width always first)
    # we should respect that order
    if ref_width < ref_height:
        ref_width, ref_height = ref_height, ref_width
        
    # Calculate pixels per unit (average of width and height ratios)
    pixels_per_unit_width = width_px / ref_width
    pixels_per_unit_height = height_px / ref_height
    
    # Use the average for better accuracy
    pixels_per_unit = (pixels_per_unit_width + pixels_per_unit_height) / 2
    
    return pixels_per_unit, (width_px, height_px), (ref_width, ref_height)

def measure_object(object_contour, pixels_per_unit):
    """
    Measure an object given its contour and the pixels-per-unit conversion.
    
    Args:
        object_contour: Contour of the object to measure
        pixels_per_unit: Pixels per unit conversion factor
        
    Returns:
        dimensions: (width, height) in real-world units
    """
    # Find the minimum area rectangle
    rect = cv2.minAreaRect(object_contour)
    width_px, height_px = rect[1]
    
    # Convert to real-world units
    width = width_px / pixels_per_unit
    height = height_px / pixels_per_unit
    
    # Ensure width is always the larger dimension
    if width < height:
        width, height = height, width
    
    return width, height, rect

def draw_reference_and_measurements(image, reference_contour, object_contour, reference_dimensions, object_dimensions, rect=None):
    """
    Draw the reference object, measured object, and their dimensions on the image.
    
    Args:
        image: Input image
        reference_contour: Contour of the reference object
        object_contour: Contour of the object being measured
        reference_dimensions: (width, height) of reference in real units
        object_dimensions: (width, height) of object in real units
        rect: Minimum area rectangle of the object (optional)
        
    Returns:
        annotated_image: Image with annotations
    """
    output = image.copy()
    
    # Draw reference object
    cv2.drawContours(output, [reference_contour], 0, (0, 255, 0), 2)
    
    # Get reference object center
    M = cv2.moments(reference_contour)
    if M["m00"] != 0:
        cX_ref = int(M["m10"] / M["m00"])
        cY_ref = int(M["m01"] / M["m00"])
    else:
        cX_ref, cY_ref = 0, 0
    
    # Draw reference dimensions
    ref_text = f"Reference: {reference_dimensions[0]:.1f} x {reference_dimensions[1]:.1f} cm"
    cv2.putText(output, ref_text, (cX_ref - 100, cY_ref - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw object contour
    cv2.drawContours(output, [object_contour], 0, (0, 0, 255), 2)
    
    # If we have the rectangle, draw the rotated bounding box
    if rect is not None:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(output, [box], 0, (255, 0, 0), 2)
    
    # Get object center
    M = cv2.moments(object_contour)
    if M["m00"] != 0:
        cX_obj = int(M["m10"] / M["m00"])
        cY_obj = int(M["m01"] / M["m00"])
    else:
        cX_obj, cY_obj = 0, 0
    
    # Draw object dimensions
    obj_text = f"Object: {object_dimensions[0]:.1f} x {object_dimensions[1]:.1f} cm"
    cv2.putText(output, obj_text, (cX_obj - 100, cY_obj - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return output