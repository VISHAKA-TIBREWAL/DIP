import cv2
import numpy as np
import os
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from modules.utils import resize_image, convert_to_grayscale, apply_threshold
from modules.feature_extraction import extract_all_features, extract_color_features, extract_color_histogram, extract_texture_features, extract_edge_features, extract_frequency_features, extract_shape_features, extract_statistical_features

FEATURE_WEIGHTS = {
    "color": 0.25,              # Importance for LAB color features
    "color_histogram": 0.25,    # Importance for HSV histogram
    "texture": 0.20,            # Importance for texture patterns (LBP)
    "edge": 0.15,               # Importance for edge features (Canny)
    "frequency": 0.10,          # Importance for frequency domain features (DCT)
    "shape": 0.03,              # Importance for shape-based features
    "statistical": 0.02         # Importance for statistical features (mean/variance)
}

# Preprocessing for ingredient recognition
def preprocess_image(image):
    """
    Preprocess an ingredient image for recognition.
    - Resize, grayscale, and apply thresholding.
    Args:
        image (numpy.ndarray): Input image.
    Returns:
        tuple: (resized_image, binary_image)
    """
    resized_image = resize_image(image)
    gray_image = convert_to_grayscale(resized_image)
    binary_image = apply_threshold(gray_image)
    return resized_image, binary_image

def compute_weighted_similarity(input_features, dataset_features):
    weighted_similarity = 0.0
    for feature_type, weight in FEATURE_WEIGHTS.items():
        if feature_type not in input_features or feature_type not in dataset_features:
            print(f"Feature {feature_type} missing in input or dataset features.")
            continue
        try:
            feature_input = input_features[feature_type]
            feature_dataset = dataset_features[feature_type]

            if not isinstance(feature_input, np.ndarray) or not isinstance(feature_dataset, np.ndarray):
                print(f"Invalid feature type for {feature_type}. Expected numpy arrays.")
                continue

            # Debugging the feature shapes before calculating similarity
            print(f"Computing similarity for feature {feature_type}:")
            print(f"Input feature shape: {feature_input.shape}, Dataset feature shape: {feature_dataset.shape}")

            sim = cosine_similarity(feature_input.reshape(1, -1), feature_dataset.reshape(1, -1))[0][0]
            weighted_similarity += weight * sim
        except Exception as e:
            print(f"Error in similarity computation for feature {feature_type}: {e}")
            continue
    return weighted_similarity



def extract_individual_features(image, binary_image):
    """
    Extract individual feature components for weighted similarity calculation.
    Returns a dictionary of feature vectors.
    """
    from modules.feature_extraction import (
        extract_color_features, extract_color_histogram, extract_texture_features,
        extract_edge_features, extract_frequency_features, extract_shape_features, extract_statistical_features
    )
    
    features = {
        "color": extract_color_features(image),
        "color_histogram": extract_color_histogram(image),
        "texture": extract_texture_features(image),
        "edge": np.array([extract_edge_features(image)]),
        "shape": extract_shape_features(binary_image),
        "statistical": extract_statistical_features(image),
        "frequency": extract_frequency_features(image),
    }

    # Ensure all features are NumPy arrays and have valid shapes
    for key, value in features.items():
        if value is None or not isinstance(value, np.ndarray):
            print(f"Feature {key} is missing or invalid. Setting it to zero vector.")
            features[key] = np.zeros((1,))  # Replace missing features with zero vectors
        else:
            features[key] = np.array(value).flatten()  # Flatten all features for consistency

        print(f"Feature {key}: {features[key].shape}")  # Debugging: print shape of each feature

    return features

def recognize_ingredients(input_image, dataset_path):
    """
    Recognize the most likely ingredient by matching input image features against dataset features.
    Args:
        input_image: The uploaded image of an ingredient.
        dataset_path: Path to the dataset folder containing ingredient images.
    Returns:
        dict: Recognized ingredient with the highest confidence score.
    """
    # Preprocess the input image
    resized_image, binary_image = preprocess_image(input_image)

    # Extract individual features from the input image
    input_features = extract_individual_features(resized_image, binary_image)
    print("Input Features (shapes):", {k: (v.shape if isinstance(v, np.ndarray) else None) for k, v in input_features.items()})

    max_confidence = 0
    best_match = None

    # Iterate over the dataset
    for ingredient_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, ingredient_folder)
        if not os.path.isdir(folder_path):
            continue

        max_similarity = 0  # Max similarity for this folder
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            dataset_image = cv2.imread(image_path)

            if dataset_image is None:
                print(f"Warning: Could not load image {file} in {ingredient_folder}. Skipping.")
                continue

            # Preprocess the dataset image
            resized_dataset_image, binary_dataset_image = preprocess_image(dataset_image)

            # Extract individual features from the dataset image
            dataset_features = extract_individual_features(resized_dataset_image, binary_dataset_image)
            print(f"Dataset Features for {file} (shapes):", {k: (v.shape if isinstance(v, np.ndarray) else None) for k, v in dataset_features.items()})

            # Compute weighted similarity
            try:
                similarity = compute_weighted_similarity(input_features, dataset_features)
                print(f"Similarity for {file}: {similarity}")
            except Exception as e:
                print(f"Error computing similarity for {file} in {ingredient_folder}: {e}")
                continue

            # Update max similarity for this folder
            max_similarity = max(max_similarity, similarity)

        # Update global best match if the current folder's max similarity is the highest so far
        if max_similarity > max_confidence:
            max_confidence = max_similarity
            best_match = {"ingredient": ingredient_folder, "confidence": max_confidence}

    # Return the ingredient with the highest confidence
    if not best_match:  # If no match is found
        return [{"ingredient": None, "confidence": 0}]
    return [best_match]  # Return the best match as a list

def recognize_ingredients_for_multiple_images(images, dataset_path):
    """
    Recognizes ingredients from multiple images.
    Args:
        images (list of numpy.ndarray): List of ingredient images.
        dataset_path (str): Path to the dataset folder containing ingredient images.
    Returns:
        list: List of dictionaries, each containing recognized ingredients.
    """
    all_recognized_ingredients = []
    
    if not isinstance(images, list) or not all(isinstance(img, np.ndarray) for img in images):
        print("Invalid input: 'images' should be a list of numpy arrays.")
        return [{"ingredient": None, "confidence": 0}]
    
    for image in images:
        recognized_ingredients = recognize_ingredients(image, dataset_path)  # Using your original function
        if recognized_ingredients:
            print(f"Recognized Ingredients: {recognized_ingredients}")  # Print the structure
            all_recognized_ingredients.append(recognized_ingredients)
        else:
            print("No ingredients recognized.")
            all_recognized_ingredients.append([{"ingredient": None, "confidence": 0}])  # Ensure it's a list
    
    print("Final recognized ingredients:", all_recognized_ingredients)  # Debug print
    return all_recognized_ingredients
