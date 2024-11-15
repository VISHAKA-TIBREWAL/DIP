import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct

def extract_color_features(image):
    """Extract mean and standard deviation in LAB color space."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mean, std = cv2.meanStdDev(lab_image)
    return np.concatenate([mean.flatten(), std.flatten()]).astype(np.float32)

def extract_color_histogram(image, bins=(8, 8, 8)):
    """Extract a 3D color histogram in the HSV color space."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)

def extract_texture_features(image, radius=1, n_points=8):
    """Extract texture features using Local Binary Patterns (LBP)."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points+3), density=True)
    return hist.astype(np.float32)

def extract_shape_features(binary_image):
    """Extract shape features such as area, perimeter, and aspect ratio."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.array([0, 0, 0], dtype=np.float32)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    return np.array([area, perimeter, aspect_ratio], dtype=np.float32)

def extract_edge_features(image):
    """Extract edge features using Canny edge detection."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return np.array([edges.mean()], dtype=np.float32)

def extract_statistical_features(image):
    """Extract statistical features such as mean and variance of pixel intensities."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray_image)
    variance = np.var(gray_image)
    return np.array([mean, variance], dtype=np.float32)

def extract_frequency_features(image):
    """Extract frequency domain features using Discrete Cosine Transform (DCT)."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct_transform = dct(dct(gray_image.T, norm='ortho').T, norm='ortho')
    dct_flattened = dct_transform.flatten()
    # Use top 10 DCT coefficients for simplicity, but ensure they're always available
    return dct_flattened[:10].astype(np.float32) if len(dct_flattened) >= 10 else np.pad(dct_flattened, (0, 10 - len(dct_flattened)), 'constant').astype(np.float32)

def extract_all_features(image, binary_image):
    """Combine all the extracted features into a single feature vector."""
    color_features = extract_color_features(image)
    color_histogram_features = extract_color_histogram(image)
    texture_features = extract_texture_features(image)
    shape_features = extract_shape_features(binary_image)
    edge_features = extract_edge_features(image)
    statistical_features = extract_statistical_features(image)
    frequency_features = extract_frequency_features(image)
    # Concatenate all features
    return np.concatenate([color_features, color_histogram_features, texture_features, shape_features, edge_features, statistical_features, frequency_features])
