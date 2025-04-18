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
        
    return valid_contours[0]  # Return the largest valid contour\

def detect_reference_object_debug(image, reference_contours=None, min_area=100, max_area=None, save_debug=True, image_filename="image"):
    """Enhanced version with debugging"""
    # Create a debug image
    debug_img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Extract base filename without path or extension
    import os
    base_filename = os.path.splitext(os.path.basename(image_filename))[0]
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    
    if max_area is None:
        max_area = image.shape[0] * image.shape[1] // 2  # Half of image area
    
    # If no contours provided, find them
    if reference_contours is None:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Use adaptive thresholding instead of global
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try multiple thresholding methods
        # 1. Otsu's method
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Adaptive thresholding
        thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 3. Canny edge detection
        edges = cv2.Canny(blurred, 30, 150)
        
        # Save debug images
        if save_debug:
            cv2.imwrite(f"{debug_dir}/{base_filename}_grayscale.jpg", gray)
            cv2.imwrite(f"{debug_dir}/{base_filename}_thresh_otsu.jpg", thresh1)
            cv2.imwrite(f"{debug_dir}/{base_filename}_thresh_adaptive.jpg", thresh2)
            cv2.imwrite(f"{debug_dir}/{base_filename}_edges.jpg", edges)
        
        # Find contours using different methods
        contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours3, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        edges2 = cv2.Canny(blurred, 10, 100)  # More sensitive
        contours4, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Try inverting the image for white objects
        inverted = cv2.bitwise_not(blurred)
        _, thresh_inv = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours5, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if save_debug:
            cv2.imwrite(f"{debug_dir}/{base_filename}_edges2.jpg", edges2)
            cv2.imwrite(f"{debug_dir}/{base_filename}_inverted.jpg", thresh_inv)

        # Combine all contours
        contours = contours1 + contours2 + contours3 + contours4 + contours5
    else:
        contours = reference_contours
    
    print(f"Found {len(contours)} contours")
    
    # Filter contours by area
    valid_contours = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            valid_contours.append(cnt)
            # Draw all valid contours on debug image
            cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(debug_img, f"{area:.0f}", 
                        tuple(cnt[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    print(f"Found {len(valid_contours)} valid contours (area between {min_area} and {max_area})")
    if save_debug:
        cv2.imwrite(f"{debug_dir}/{base_filename}_all_contours.jpg", debug_img)
    
    if not valid_contours:
        return None
    
    # Sort by area (largest first)
    valid_contours.sort(key=cv2.contourArea, reverse=True)
    
    # Get the largest contour
    largest_contour = valid_contours[0]
    
    # Draw the selected contour in a different color
    cv2.drawContours(debug_img, [largest_contour], -1, (0, 0, 255), 3)
    if save_debug:
        cv2.imwrite(f"{debug_dir}/{base_filename}_selected_contour.jpg", debug_img)
    
    return largest_contour

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

def measure_object(object_contour, pixels_per_unit, correction_factor=2.75):
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
    width = width_px / pixels_per_unit / correction_factor
    height = height_px / pixels_per_unit / correction_factor
    
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