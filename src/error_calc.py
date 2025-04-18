import os
import cv2

def read_actual_dimensions(image_path):
    """
    Read actual object dimensions from a text file with the same name as the image.
    Format of text file should be: width height (in cm)
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (width, height) in cm, or None if file doesn't exist
    """
    # Get the base name without extension
    base_path = os.path.splitext(image_path)[0]
    txt_path = base_path + '.txt'
    
    # Check if the text file exists
    if not os.path.exists(txt_path):
        return None
        
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    values = line.strip().split()
                    if len(values) >= 2:
                        width = float(values[0])
                        height = float(values[1])
                        return (width, height)
    except Exception as e:
        print(f"Error reading actual dimensions: {e}")
        return None
    
    return None

def calculate_error_metrics(measured_dims, actual_dims):
    """
    Calculate error metrics between measured and actual dimensions.
    
    Args:
        measured_dims: (width, height) as measured
        actual_dims: (width, height) actual
        
    Returns:
        dict: Error metrics including absolute error, relative error, etc.
    """
    measured_width, measured_height = measured_dims
    actual_width, actual_height = actual_dims
    
    # Absolute errors
    abs_error_width = abs(measured_width - actual_width)
    abs_error_height = abs(measured_height - actual_height)
    
    # Relative errors (percentage)
    rel_error_width = (abs_error_width / actual_width) * 100
    rel_error_height = (abs_error_height / actual_height) * 100
    
    return {
        'actual_width': actual_width,
        'actual_height': actual_height,
        'abs_error_width': abs_error_width,
        'abs_error_height': abs_error_height,
        'rel_error_width': rel_error_width,
        'rel_error_height': rel_error_height
    }

def add_error_metrics_to_image(image, calculated_dimensions, error_metrics):
    """
    Add very small error metrics at the bottom of the image.
    
    Args:
        image: The input image (with measurements already drawn)
        calculated_dimensions: The dimensions calculated by the pipeline
        error_metrics: Dictionary containing error metrics
        
    Returns:
        Annotated image with error metrics
    """
    output = image.copy()
    height, width = output.shape[:2]
    
    # Add a smaller semi-transparent background
    overlay = output.copy()
    bg_start_x, bg_start_y = 5, height - 55  # Position at very bottom
    bg_width, bg_height = 280, 50  # Much smaller area
    cv2.rectangle(overlay, (bg_start_x-3, bg_start_y-3), 
                 (bg_start_x+bg_width, bg_start_y+bg_height), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
    
    # Use very small font size and compact format
    font_size = 0.35
    line_height = 15  # Very small line height
    
    # Draw minimized error metrics
    y_offset = bg_start_y + 12
    cv2.putText(output, f"Act: {error_metrics['actual_width']:.1f}×{error_metrics['actual_height']:.1f}cm", 
               (bg_start_x+5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1)
    y_offset += line_height
    cv2.putText(output, f"Abs: W={error_metrics['abs_error_width']:.2f}cm H={error_metrics['abs_error_height']:.2f}cm", 
               (bg_start_x+5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1)
    y_offset += line_height
    cv2.putText(output, f"Rel: W={error_metrics['rel_error_width']:.1f}% H={error_metrics['rel_error_height']:.1f}%", 
               (bg_start_x+5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1)
    
    return output