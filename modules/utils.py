import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename

# General image preprocessing
def resize_image(image, size=(256, 256)):
    """Resize an image to the specified size."""
    return cv2.resize(image, size)

def convert_to_grayscale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_threshold(image, method=cv2.THRESH_BINARY + cv2.THRESH_OTSU):
    """Apply thresholding to an image."""
    _, binary_image = cv2.threshold(image, 128, 255, method)
    return binary_image

def save_uploaded_file(upload_folder, file):
    """Save uploaded file to the server."""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        return file_path
    return None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_uploads(upload_folder):
    """Clear all files in the uploads folder."""
    if os.path.exists(upload_folder):
        for file in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

# Error handling utilities
def handle_invalid_image(file_path):
    """Check if the uploaded file is a valid image."""
    try:
        image = cv2.imread(file_path)
        if image is None or not isinstance(image, np.ndarray):
            return False
        return True
    except Exception:
        return False
    
def convert_numpy_float32(obj):
    """
    Recursively converts all numpy.float32 to float.
    """
    if isinstance(obj, np.ndarray):
        # Convert each element of the array to float
        return obj.astype(float)
    elif isinstance(obj, dict):
        # Recursively convert dict values
        return {key: convert_numpy_float32(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Recursively convert list elements
        return [convert_numpy_float32(item) for item in obj]
    elif isinstance(obj, np.float32):
        # Convert numpy.float32 to float
        return float(obj)
    return obj

