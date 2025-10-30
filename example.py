#!/usr/bin/env python3
"""
Example script demonstrating Fruitify usage.
This script shows how to use the Fruitify prediction functions programmatically.
"""
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import sys


def predict_fruit_simple(image_path, model_path="mive-doost-dari?.h5", labels_path="labels.txt"):
    """
    Simple function to predict fruit type from an image.
    
    Args:
        image_path: Path to the fruit image
        model_path: Path to the trained model
        labels_path: Path to the labels file
        
    Returns:
        Tuple of (predicted_class, confidence_score)
    """
    # Load model and labels
    model = load_model(model_path, compile=False)
    
    with open(labels_path, "r") as f:
        class_names = f.readlines()
    
    # Prepare image
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    
    # Predict
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    
    # Remove index prefix from class name
    if ' ' in class_name:
        class_name = class_name.split(' ', 1)[1]
    
    return class_name, confidence_score


def main():
    """Example usage of the prediction function."""
    print("Fruitify - Example Usage")
    print("=" * 50)
    
    # Example 1: Predict a single image
    print("\nExample 1: Single image prediction")
    try:
        fruit, confidence = predict_fruit_simple("goje.jpg")
        print(f"  Image: goje.jpg")
        print(f"  Predicted: {fruit}")
        print(f"  Confidence: {confidence:.2%}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Example 2: Predict multiple images
    print("\nExample 2: Multiple image predictions")
    test_images = ["moz.png", "porteghal.png", "sibg.png"]
    
    for img in test_images:
        try:
            fruit, confidence = predict_fruit_simple(img)
            print(f"  {img:20s} -> {fruit:15s} ({confidence:.2%})")
        except Exception as e:
            print(f"  {img:20s} -> Error: {e}")
    
    print("\n" + "=" * 50)
    print("For more advanced usage, try:")
    print("  python main.py --help")
    print("  python top3.py --help")
    print("  python batch_predict.py --help")


if __name__ == "__main__":
    main()
