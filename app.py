from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
from modules.utils import save_uploaded_file, handle_invalid_image, convert_numpy_float32
from modules.ingredient_recognition import preprocess_image, recognize_ingredients_for_multiple_images
from modules.recipe_generator import load_recipes, get_matching_recipes, get_recipe_by_name

app = Flask(__name__)

# Load the recipe dataset
DATASET_PATH = r"C:\Users\visha\OneDrive\Desktop\DIP\DIP_DATASET.xlsx"
recipes = load_recipes(DATASET_PATH)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")

@app.route('/recipe/<string:recipe_name>')
def recipe_detail(recipe_name):
    """Render the details of a specific recipe."""
    recipe = get_recipe_by_name(recipe_name)  # Fetch recipe by name from Excel
    if not recipe:
        return jsonify({"error": "Recipe not found"}), 404
    return render_template('recipe.html', recipe=recipe)


@app.route("/upload", methods=["POST"])
def upload():
    """
    Handle the image upload, recognize ingredients,
    and return matching recipes for multiple uploaded images.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "No files selected"}), 400

    recognized_ingredients_all = []
    matching_recipes_all = []

    # Process each uploaded file
    for file in files:
        # Save uploaded file
        file_path = save_uploaded_file(UPLOAD_FOLDER, file)

        # Validate the uploaded image
        if not handle_invalid_image(file_path):
            return jsonify({"error": "Invalid image file"}), 400

        # Preprocess and recognize ingredients from the image
        image = cv2.imread(file_path)
        recognized_ingredients = recognize_ingredients_for_multiple_images([image], "dip_ingredients")  # Path to ingredient dataset
        
        # Now process recognized_ingredients correctly for each file
        if isinstance(recognized_ingredients, list):
            # Case 1: If recognized_ingredients is a list of strings (e.g., ["ingredient1", "ingredient2"])
            if all(isinstance(ingredient, str) for ingredient in recognized_ingredients):
                recognized_ingredients = [{"ingredient": ingredient} for ingredient in recognized_ingredients]

            # Case 2: If recognized_ingredients is a list of dictionaries (as you initially expected)
            elif all(isinstance(ingredient, dict) for ingredient in recognized_ingredients):
                recognized_ingredients = [
                    {key: (float(value) if isinstance(value, np.float32) else value) for key, value in ingredient.items()}
                    for ingredient in recognized_ingredients
                ]
            else:
                print("Error: recognized_ingredients is a list, but it contains elements of unexpected types.")
                return jsonify({"error": "Invalid structure of recognized ingredients"}), 400
        else:
            print("Error: recognized_ingredients is not a list.")
            return jsonify({"error": "Invalid recognized ingredients structure"}), 400

        # Find matching recipes for the recognized ingredients
        matching_recipes = get_matching_recipes(recognized_ingredients, recipes)

        # Convert all numpy.float32 values in the matching recipes to standard float
        matching_recipes = convert_numpy_float32(matching_recipes)

        # Append the results for this image
        recognized_ingredients_all.append(recognized_ingredients)
        matching_recipes_all.append(matching_recipes)

    if not recognized_ingredients_all or not matching_recipes_all:
        return jsonify({"error": "No ingredients recognized or no matching recipes found"}), 400
    
    # Return the results for all images uploaded
    return jsonify({
        "recognized_ingredients": recognized_ingredients_all,
        "recipes": matching_recipes_all
    })




if __name__ == "__main__":
    app.run(debug=True, threaded=True)
