from camera_calibration import calibrate_camera, save_calibration
import os

# Make sure the path is correct
calib_path = "../calib_data/*.jpg"

# Check if directory exists
import glob
images = glob.glob(calib_path)
print(f"Found {len(images)} calibration images")

if len(images) == 0:
    # Try alternative extensions
    for ext in [".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        alt_path = "../calib_data/*" + ext
        images = glob.glob(alt_path)
        if len(images) > 0:
            print(f"Found {len(images)} images with extension {ext}")
            calib_path = alt_path
            break

# Create output directory if it doesn't exist
os.makedirs("../calibration", exist_ok=True)

# Only proceed if images were found
if len(images) > 0:
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(
        calib_path,
        checkerboard_size=(9, 7),
        square_size=20.0
    )
    
    # Print calibration accuracy
    print(f"Calibration accuracy: {ret}")
    
    # Save calibration parameters
    save_calibration("../calibration/camera_calibration.pkl", mtx, dist)
else:
    print("ERROR: No calibration images found. Please check the path and image format.")
    print("The calibration images should be located in the '../calib_data/' directory.")