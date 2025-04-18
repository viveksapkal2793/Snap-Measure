from matplotlib_imshow import matplotlib_imshow
from get_img import read_or_capture
from img_preproc import preprocess
from find_ref_object import find_corners
from trans_prespec import perspective_transform
from find_object import find_object_of_interest
from calculate_dimensions import calculate_dimensions
from viz_detec import visualize_detections
from error_calc import read_actual_dimensions, calculate_error_metrics, add_error_metrics_to_image 
import argparse
import cv2
import numpy as np
from ref_object import detect_reference_object, calculate_pixels_per_metric, measure_object, draw_reference_and_measurements, detect_reference_object_debug
from camera_calibration import load_calibration, undistort_image

def pipeline_for_still_images(
    prompt_user=False,
    image_path="../input_images/jar.jpg",
    capturing_device_id=None,
    visualize=True,
    scale=8,
    use_calibration=False,
    calibration_file=None,
    reference_object_dimensions=None,  # e.g., (width, height) in cm
    use_reference_object=False
):
    """
    A pipeline for detecting objects of interest from a still image and find the objects dimensions (width, height) in cm.

    Args:
        prompt_user: whether to prompt the user for image path or, device id. (default: False)
        image_path: to use a stock/pre-captured image instead of prompting the user. (default: "./sample_imgs/paint_brush.jpeg")
        capturing_device_id: to capture a live image instead of prompting or loading a stock one. (default: None)
        visualize: whether to show the output image containing the info of detections. (default: True)
        scale: matplotlib_imshow() function visualization scale. (default: 8)
        use_calibration: Whether to use camera calibration for measurements (default: False)
        calibration_file: Path to camera calibration file (default: None)
        reference_object_dimensions: Dimensions of reference object in the scene (width, height) in cm
        use_reference_object: Use general reference object method instead of A4 paper

    Returns: The output image (A rotated bounding box is drawn around the object of interest. The calculated dimensions (width, height) are also shown on the output image.)

    """

    img = read_or_capture(prompt_user, image_path, capturing_device_id)
    
    if use_calibration and calibration_file:
        try:
            camera_matrix, dist_coeffs = load_calibration(calibration_file)
            img = undistort_image(img, camera_matrix, dist_coeffs)
            print("Applied camera calibration for distortion correction")
        except Exception as e:
            print(f"Error applying calibration: {e}")
    
    if use_reference_object and reference_object_dimensions:
        # Use the reference object method
        # Preprocess the image
        preprocessed_img = preprocess(img)

        # Try to detect reference object with more lenient parameters
        reference_contour = detect_reference_object_debug(
            preprocessed_img, 
            min_area=50,  # Even more lenient minimum area
            max_area=preprocessed_img.shape[0] * preprocessed_img.shape[1] * 0.9,  # 90% of image
            save_debug=True,
            image_filename=image_path  # Pass the image filename
        )
        
        if reference_contour is None:
            print("Error: Could not detect reference object")
            return img
        
        # Check if image is already grayscale
        if len(preprocessed_img.shape) == 2 or (len(preprocessed_img.shape) == 3 and preprocessed_img.shape[2] == 1):
            gray = preprocessed_img  # Already grayscale
        else:
            gray = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect reference object (assuming it's one of the larger objects)
        # reference_contour = detect_reference_object(preprocessed_img, contours)
        
        # if reference_contour is None:
        #     print("Error: Could not detect reference object")
        #     return img
        
        # Calculate pixels per metric
        pixels_per_unit, ref_px_dims, ref_dims = calculate_pixels_per_metric(
            reference_contour, reference_object_dimensions)
        
        print(f"Reference object: {ref_px_dims[0]:.1f} x {ref_px_dims[1]:.1f} pixels")
        print(f"Reference dimensions: {ref_dims[0]} x {ref_dims[1]} cm")
        print(f"Pixels per unit: {pixels_per_unit:.2f} pixels/cm")
        
        # Find the largest non-reference object (object of interest)
        # Exclude the reference object from consideration
        # other_contours = [c for c in contours if cv2.contourArea(c) < cv2.contourArea(reference_contour)]
        # other_contours = sorted(other_contours, key=cv2.contourArea, reverse=True)
        
        # if not other_contours:
        #     print("Error: Could not detect any objects besides the reference")
        #     return img

        ref_area = cv2.contourArea(reference_contour)
        other_contours = []
        
        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)
            # Skip contours that are too similar to reference object
            if abs(cnt_area - ref_area) / max(cnt_area, ref_area) > 0.2:  # If area differs by more than 20%
                other_contours.append(cnt)
        
        if not other_contours:
            print("Error: Could not detect any objects besides the reference")
            # Draw just the reference for debugging
            output_img = img.copy()
            cv2.drawContours(output_img, [reference_contour], 0, (0, 255, 0), 2)
            return output_img
            
        # Sort by area (largest first)
        other_contours.sort(key=cv2.contourArea, reverse=True)
            
        object_contour = other_contours[0]
        
        # Measure the object
        width, height, rect = measure_object(object_contour, pixels_per_unit)
        
        # Visualize the results
        output_img_to_show = draw_reference_and_measurements(
            img, reference_contour, object_contour, 
            reference_object_dimensions, (width, height), rect
        )
        
    else:
        # Use existing A4 paper approach
        preprocessed_img = preprocess(img)
        corners = find_corners(preprocessed_img)
        perspective_transformed_img = perspective_transform(img, corners)
        convex_hull = find_object_of_interest(perspective_transformed_img)
        output_img_to_show = visualize_detections(perspective_transformed_img, convex_hull)

        # Get the calculated dimensions from visualize_detections
        output_img_to_show, calculated_dimensions = visualize_detections(
            perspective_transformed_img, convex_hull, return_dimensions=True)
        
        # Check if there's a corresponding text file with actual dimensions
        actual_dims = read_actual_dimensions(image_path)
        
        if actual_dims and calculated_dimensions:
            # Calculate error metrics
            error_metrics = calculate_error_metrics(calculated_dimensions, actual_dims)
            print(f"Actual dimensions: {actual_dims[0]:.1f} x {actual_dims[1]:.1f} cm")
            print(f"Measured dimensions: {calculated_dimensions[0]:.1f} x {calculated_dimensions[1]:.1f} cm")
            print(f"Measurement errors:")
            print(f"  Absolute: Width = {error_metrics['abs_error_width']:.2f} cm, Height = {error_metrics['abs_error_height']:.2f} cm")
            print(f"  Relative: Width = {error_metrics['rel_error_width']:.1f}%, Height = {error_metrics['rel_error_height']:.1f}%")
            
            # Add error metrics to the image
            output_img_to_show = add_error_metrics_to_image(
                output_img_to_show, calculated_dimensions, error_metrics)
        
    if visualize is True:
        matplotlib_imshow(
            "Detected Object and its Calculated measurements \n(Width and Height) in cm",
            output_img_to_show,
            scale,
        )
    
    return output_img_to_show

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object measurement pipeline')
    parser.add_argument('--use_calibration', action='store_true', 
                        help='Use camera calibration')
    parser.add_argument('--calibration_file', type=str,
                        help='Path to camera calibration file')
    parser.add_argument('--image_path', type=str, default="../input_images/mouse.jpg",
                        help='Path to input image')
    parser.add_argument('--reference_width', type=float,
                        help='Width of reference object in cm')
    parser.add_argument('--reference_height', type=float,
                        help='Height of reference object in cm')
    parser.add_argument('--use_reference_object', action='store_true',
                        help='Use general reference object method instead of A4 paper')
    
    args = parser.parse_args()
    
    # Set up reference dimensions if provided
    reference_dimensions = None
    if args.reference_width is not None and args.reference_height is not None:
        reference_dimensions = (args.reference_width, args.reference_height)
    
    # Run the pipeline
    output_img = pipeline_for_still_images(
        image_path=args.image_path,
        use_calibration=args.use_calibration,
        calibration_file=args.calibration_file,
        reference_object_dimensions=reference_dimensions,
        use_reference_object=args.use_reference_object
    )