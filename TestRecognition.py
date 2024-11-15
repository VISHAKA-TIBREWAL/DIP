import os
import cv2
import json
import random
from modules.ingredient_recognition import recognize_ingredients

# Paths
DATASET_PATH = r"C:\Users\visha\OneDrive\Desktop\DIP\dip_ingredients"
TEST_IMAGES_FOLDER = r"C:\Users\visha\OneDrive\Desktop\DIP\test_images"
GROUND_TRUTH_FILE = r"C:\Users\visha\OneDrive\Desktop\DIP\ground_truth.json"

# Step 1: Generate Test Images and Ground Truth Automatically
def generate_test_images(dataset_path, test_images_folder, ground_truth_file, num_samples=3):
    """
    Automatically generate test images and ground truth from the dataset.
    Args:
        dataset_path (str): Path to the ingredient dataset.
        test_images_folder (str): Path to save test images.
        ground_truth_file (str): Path to save the ground truth JSON file.
        num_samples (int): Number of test images per ingredient.
    """
    if not os.path.exists(test_images_folder):
        os.makedirs(test_images_folder)
    
    ground_truth = {}

    for ingredient_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, ingredient_folder)
        if not os.path.isdir(folder_path):
            continue

        # Randomly select a few images from the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        selected_files = random.sample(image_files, min(len(image_files), num_samples))

        for file in selected_files:
            src_path = os.path.join(folder_path, file)
            dest_path = os.path.join(test_images_folder, f"{ingredient_folder}_{file}")
            cv2.imwrite(dest_path, cv2.imread(src_path))  # Save the test image
            ground_truth[f"{ingredient_folder}_{file}"] = ingredient_folder

    # Save ground truth to JSON file
    with open(ground_truth_file, "w") as f:
        json.dump(ground_truth, f, indent=4)
    print(f"Test images saved to {test_images_folder}.")
    print(f"Ground truth saved to {ground_truth_file}.")

# Step 2: Evaluate Ingredient Recognition
def evaluate_recognition(test_images_folder, ground_truth_file, dataset_path):
    """
    Evaluate the ingredient recognition pipeline using test images.
    Args:
        test_images_folder (str): Path to the folder containing test images.
        ground_truth_file (str): Path to the ground truth JSON file.
        dataset_path (str): Path to the ingredient dataset.
    """
    # Load ground truth
    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)

    correct = 0
    total = len(ground_truth)

    # Evaluate each test image
    for test_image, expected_ingredient in ground_truth.items():
        test_image_path = os.path.join(test_images_folder, test_image)
        test_img = cv2.imread(test_image_path)

        if test_img is None:
            print(f"Error: Could not load {test_image}")
            continue

        recognized_ingredients = recognize_ingredients(test_img, dataset_path)

        # Check if the expected ingredient is among recognized ingredients
        if any(ing["ingredient"] == expected_ingredient for ing in recognized_ingredients):
            correct += 1
        else:
            print(f"Missed: {test_image} (Expected: {expected_ingredient})")

    # Accuracy
    accuracy = (correct / total) * 100
    print(f"Recognition Accuracy: {accuracy:.2f}%")

# Main Script
if __name__ == "__main__":
    # Step 1: Generate Test Images and Ground Truth
    generate_test_images(DATASET_PATH, TEST_IMAGES_FOLDER, GROUND_TRUTH_FILE, num_samples=3)

    # Step 2: Evaluate Recognition
    evaluate_recognition(TEST_IMAGES_FOLDER, GROUND_TRUTH_FILE, DATASET_PATH)
