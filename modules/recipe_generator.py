import pandas as pd
from fuzzywuzzy import fuzz
from PIL import Image
import os

def load_recipes(dataset_path):
    """
    Load recipe data from the Excel file.
    Args:
        dataset_path: Path to the Excel file containing recipes.
    Returns:
        pd.DataFrame: DataFrame containing the recipes or None if an error occurs.
    """
    try:
        recipes = pd.read_excel(dataset_path)
        return recipes
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def get_recipe_by_name(recipe_name):
    """Fetch a recipe by its name."""
    # Filter recipes by name
    recipes_df = load_recipes(dataset_path)
    recipe = recipes_df[recipes_df['Recipe Name'] == recipe_name]
    if not recipe.empty:
        return {
            'name': recipe['Recipe Name'].values[0],
            'description': recipe['Description'].values[0],
            'ingredients': recipe['Ingredients'].values[0],
            'instructions': recipe['Instructions'].values[0],
            'preparation_time': recipe['Preparation Time'].values[0],
            'cooking_time': recipe['Cooking Time'].values[0],
            'total_time': recipe['Total time'].values[0],
            'nutrition_value': recipe['Nutrition Value (per serving)'].values[0],
            'final_dish_image': recipe['Final Dish Image'].values[0] if pd.notna(recipe['Final Dish Image'].values[0]) else 'default_image.png'
        }
    else:
        return None  # If no recipe is found by name, return None

def compress_image(input_path, output_path, quality=75, max_width=500):
    """
    Compress the image by resizing and adjusting the quality.
    Args:
        input_path: Path to the input image.
        output_path: Path to save the compressed image.
        quality: Compression quality (1-100), default is 85.
        max_width: Resize the image if its width exceeds this value.
    """
    try:
        with Image.open(input_path) as img:
            width, height = img.size
            
            # Resize the image if the width is larger than the max width
            if width > max_width:
                ratio = max_width / float(width)
                new_height = int(float(height) * ratio)
                img = img.resize((max_width, new_height), Image.ANTIALIAS)
            
            # Save the image with reduced quality
            img.save(output_path, quality=quality, optimize=True)
            print(f"Image compressed and saved to: {output_path}")
    except Exception as e:
        print(f"Error compressing image {input_path}: {e}")

def get_matching_recipes(recognized_ingredients, recipes):
    """
    Get matching recipes based on recognized ingredients.
    """
    # Ensure recognized_ingredients is a list of strings
    if isinstance(recognized_ingredients, list) and all(isinstance(ingredient, str) for ingredient in recognized_ingredients):
        recognized_names = [ingredient.strip().lower() for ingredient in recognized_ingredients]
    else:
        print("Error: recognized_ingredients should be a list of strings.")
        return []

    matching_recipes = []
    
    for recipe in recipes:
        # Assuming each recipe has an 'Ingredients' field that contains a list of ingredients
        recipe_ingredients = [ingredient.strip().lower() for ingredient in recipe['Ingredients']]
        
        # Check if all recognized ingredients are in the recipe's ingredients
        if all(ingredient in recipe_ingredients for ingredient in recognized_names):
            matching_recipes.append(recipe)
    
    return matching_recipes


# Attempt to split the instructions into individual steps
def split_instructions(instructions):
    """
    Split the instructions into individual steps based on common separators.
    Args:
        instructions: A string containing the recipe instructions.
    Returns:
        A list of instruction steps.
    """
    # First, handle cases where there is a clear separator
    if ". " in instructions:
        instructions = instructions.split(". ")
    elif "; " in instructions:
        instructions = instructions.split("; ")
    else:
        # If no separator is found, use line breaks or try splitting by other criteria
        instructions = instructions.splitlines()

    # Remove empty lines and ensure that the instructions are cleaned up
    instructions = [line.strip() for line in instructions if line.strip()]

    # If needed, add a period at the end of each instruction (if it was removed earlier)
    instructions = [line + '.' if not line.endswith('.') else line for line in instructions]

    return instructions

# Example Usage
if __name__ == "__main__":
    # Path to your recipe dataset
    dataset_path = r"C:\Users\visha\OneDrive\Desktop\dip project\DIP_DATASET.xlsx"

    # Load recipes
    recipes = load_recipes(dataset_path)

    if recipes is not None:
        # Example recognized ingredients
        recognized_ingredients = [
            {"ingredient": "Tomato", "confidence": 0.85},
            {"ingredient": "Potato", "confidence": 0.78},
            {"ingredient": "Onion", "confidence": 0.65},
        ]

        # Get matching recipes
        final_dish_folder = r"C:\Users\visha\OneDrive\Desktop\dip project\static\final_dishes"
        matching_recipes = get_matching_recipes(recognized_ingredients, recipes, final_dish_folder)
        
        # Display matching recipes
        print("Matching Recipes:")
        for recipe in matching_recipes:
            print(f"Recipe Name: {recipe['Recipe Name']}")
            print(f"Ingredients: {recipe['Ingredients']}")

            # Split the instructions using the improved function
            instructions = split_instructions(recipe['Instructions'])

            # Print each step on a new line
            for line in instructions:
                print(f"  - {line.strip()}")

            print(f"Preparation Time: {recipe['Preparation Time']}")
            print(f"Cooking Time: {recipe['Cooking Time']}")
            print(f"Total Time: {recipe['Total Time']}")
            print(f"Nutrition Value: {recipe['Nutrition Value']}")
            print(f"Final Dish Image: {recipe['Final Dish Image']}")
            print(f"Matched Count: {recipe['Matched Count']}")
            print()
