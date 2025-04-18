from camera_calibration import calibrate_camera, save_calibration

# Run calibration with your 26 images
ret, mtx, dist, rvecs, tvecs = calibrate_camera(
    "../calib_data/*.jpg",  # Your 26 images
    checkerboard_size=(9, 7),  # Adjust to your actual pattern
    square_size=20.0  # Size in mm of each square
)

# Print calibration accuracy
print(f"Calibration accuracy: {ret}")

# Save calibration parameters
save_calibration("../calibration/camera_calibration.pkl", mtx, dist)