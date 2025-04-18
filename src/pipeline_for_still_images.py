from matplotlib_imshow import matplotlib_imshow
from read_or_capture import read_or_capture
from preprocessing import preprocess
from locate_reference_object import find_corners
from perspective_transform import perspective_transform
from locate_object_of_interest import find_object_of_interest
from calculate_dimensions import calculate_dimensions
from visualize_detections import visualize_detections

def pipeline_for_still_images(
    prompt_user=False,
    image_path="../input_images/mouse.jpg",
    capturing_device_id=None,
    visualize=True,
    scale=8,
    use_calibration=False,
    calibration_file=None,
    reference_object_dimensions=None,  # e.g., (width, height) in cm
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

    Returns: The output image (A rotated bounding box is drawn around the object of interest. The calculated dimensions (width, height) are also shown on the output image.)

    """

    img = read_or_capture(prompt_user, image_path, capturing_device_id)
    
    if use_calibration and calibration_file:
        # Load camera calibration
        from camera_calibration import load_calibration, undistort_image
        camera_matrix, dist_coeffs = load_calibration(calibration_file)
        
        # Undistort the image
        img = undistort_image(img, camera_matrix, dist_coeffs)
        
        # Use calibrated measurement approach
        # ...implement calibrated measurement logic here...
        # This will require detecting a reference object if present
        # and establishing the pixel-to-real-world ratio
        
    else:
        # Use existing A4 paper approach
        preprocessed_img = preprocess(img)
        corners = find_corners(preprocessed_img)
        perspective_transformed_img = perspective_transform(img, corners)
        convex_hull = find_object_of_interest(perspective_transformed_img)
        output_img_to_show = visualize_detections(perspective_transformed_img, convex_hull)
    
    if visualize is True:
        matplotlib_imshow(
            "Detected Object and its Calculated measurements \n(Width and Height) in cm",
            output_img_to_show,
            scale,
        )
    
    return output_img_to_show

if __name__ == "__main__":
    import argparse
    
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
    
    args = parser.parse_args()
    
    # Pass arguments to the pipeline
    reference_dimensions = None
    if args.reference_width is not None and args.reference_height is not None:
        reference_dimensions = (args.reference_width, args.reference_height)
    
    output_img = pipeline_for_still_images(
        image_path=args.image_path,
        use_calibration=args.use_calibration,
        calibration_file=args.calibration_file,
        reference_object_dimensions=reference_dimensions
    )