import numpy as np
import cv2
import glob
import pickle
import os

def calibrate_camera(images_path, checkerboard_size=(9,7), square_size=20.0, debug=True):
    """
    Calibrate camera using multiple checkerboard images
    
    Args:
        images_path: Path pattern to calibration images (e.g., "calibration_images/*.jpg")
        checkerboard_size: Number of inner corners (width, height)
        square_size: Size of checkerboard square in mm
        debug: Whether to show debug information
        
    Returns:
        ret: Calibration accuracy
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    
    images = glob.glob(images_path)
    print(f"Found {len(images)} images at path: {images_path}")
    if len(images) == 0:
        print("ERROR: No images found! Check your path and image format.")
        return None, None, None, None, None
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Convert to real-world units (mm)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of calibration images
    successful_images = 0
    
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Could not read image: {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        # Use more aggressive flags for better detection
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)
        
        # If found, add object points, image points
        if ret:
            successful_images += 1
            # Refine corner locations
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            
            if debug:
                print(f"Successfully detected corners in: {fname}")
                # Draw and display the corners
                img_with_corners = img.copy()
                cv2.drawChessboardCorners(img_with_corners, checkerboard_size, corners2, ret)
                
                # Create a debug directory if it doesn't exist
                os.makedirs("../calibration_debug", exist_ok=True)
                
                # Save the image with corners for debugging
                base_name = os.path.basename(fname)
                cv2.imwrite(f"../calibration_debug/corners_{base_name}", img_with_corners)
        else:
            if debug:
                print(f"Failed to detect corners in: {fname}")
    
    print(f"Successfully detected checkerboard pattern in {successful_images} out of {len(images)} images")
    
    if successful_images == 0:
        print("ERROR: Could not detect checkerboard pattern in any of the images.")
        print("Please check if:")
        print("1. The checkerboard_size is correct (you specified width=%d, height=%d)" % checkerboard_size)
        print("2. The checkerboard is fully visible in the images")
        print("3. The images are clear and not blurry")
        return None, None, None, None, None
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    print(f"Total reprojection error: {mean_error/len(objpoints)}")
    
    return ret, mtx, dist, rvecs, tvecs

def save_calibration(filename, camera_matrix, dist_coeffs):
    """Save camera calibration results to file"""
    calibration_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs
    }
    with open(filename, 'wb') as f:
        pickle.dump(calibration_data, f)
    
def load_calibration(filename):
    """Load camera calibration from file"""
    with open(filename, 'rb') as f:
        calibration_data = pickle.load(f)
    return calibration_data['camera_matrix'], calibration_data['dist_coeffs']

def undistort_image(image, camera_matrix, dist_coeffs):
    """Undistort an image using camera calibration parameters"""
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    
    # Undistort
    dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramtx)
    
    # Crop the image (optional)
    x, y, w, h = roi
    if all([x, y, w, h]):  # Check if ROI is valid
        dst = dst[y:y+h, x:x+w]
    
    return dst