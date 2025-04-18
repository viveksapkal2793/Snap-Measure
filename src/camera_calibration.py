import numpy as np
import cv2
import glob
import pickle

def calibrate_camera(images_path, checkerboard_size=(9,7), square_size=20.0):
    """
    Calibrate camera using multiple checkerboard images
    
    Args:
        images_path: Path pattern to calibration images (e.g., "calibration_images/*.jpg")
        checkerboard_size: Number of inner corners (width, height)
        square_size: Size of checkerboard square in mm
        
    Returns:
        ret: Calibration accuracy
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Convert to real-world units (mm)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of calibration images
    images = glob.glob(images_path)
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Optional: Draw and display the corners
            # cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
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