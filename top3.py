"""
Fruitify - Top 3 Fruit Predictions
Shows the top 3 most likely fruit classifications for an image.
"""
import argparse
import sys
import os
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


def load_class_names(labels_file="labels.txt"):
    """Load class names from labels file.
    
    Args:
        labels_file: Path to the labels file
        
    Returns:
        List of class names
        
    Raises:
        FileNotFoundError: If labels file doesn't exist
    """
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    with open(labels_file, "r") as f:
        return f.readlines()


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image for model prediction.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the image (width, height)
        
    Returns:
        Preprocessed image array ready for prediction
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be processed
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # Create batch with single image
        data = np.ndarray(shape=(1, *target_size, 3), dtype=np.float32)
        data[0] = normalized_image_array
        return data
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")


def predict_top_n(model, image_data, class_names, top_n=3):
    """Predict the top N fruit classes from preprocessed image data.
    
    Args:
        model: Loaded Keras model
        image_data: Preprocessed image array
        class_names: List of class names
        top_n: Number of top predictions to return
        
    Returns:
        List of tuples (class_name, confidence_score)
    """
    prediction = model.predict(image_data, verbose=0)
    top_indices = np.argsort(-prediction[0])[:top_n]
    
    results = []
    for i in top_indices:
        class_name = class_names[i].strip()
        confidence_score = prediction[0][i]
        
        # Remove the index prefix (e.g., "0 " from "0 sib ghermez")
        if ' ' in class_name:
            class_name = class_name.split(' ', 1)[1]
        
        results.append((class_name, confidence_score))
    
    return results


def main():
    """Main function to run fruit classification with top 3 predictions."""
    parser = argparse.ArgumentParser(
        description="Get top 3 fruit predictions for an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python top3.py moz.png
  python top3.py --model custom_model.h5 --labels custom_labels.txt goje.jpg
  python top3.py --top 5 porteghal.png  # Get top 5 predictions
        """
    )
    parser.add_argument(
        "image",
        nargs='?',
        default="moz.png",
        help="Path to the fruit image to classify (default: moz.png)"
    )
    parser.add_argument(
        "--model",
        default="mive-doost-dari?.h5",
        help="Path to the trained model file (default: mive-doost-dari?.h5)"
    )
    parser.add_argument(
        "--labels",
        default="labels.txt",
        help="Path to the labels file (default: labels.txt)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of top predictions to show (default: 3)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load model
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}", file=sys.stderr)
            return 1
        
        print(f"Loading model from {args.model}...")
        try:
            model = load_model(args.model, compile=False)
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            print("Note: Model compatibility issues may occur with different TensorFlow/Keras versions.", file=sys.stderr)
            return 1
        
        # Load class names
        print(f"Loading labels from {args.labels}...")
        class_names = load_class_names(args.labels)
        
        # Validate top_n parameter
        if args.top < 1 or args.top > len(class_names):
            print(f"Error: --top must be between 1 and {len(class_names)}", file=sys.stderr)
            return 1
        
        # Load and preprocess image
        print(f"Processing image: {args.image}...")
        image_data = load_and_preprocess_image(args.image)
        
        # Make predictions
        print(f"Making top {args.top} predictions...")
        predictions = predict_top_n(model, image_data, class_names, args.top)
        
        # Display results
        print("\n" + "="*50)
        print(f"Top {args.top} Predictions:")
        print("="*50)
        for rank, (class_name, confidence) in enumerate(predictions, 1):
            print(f"{rank}. {class_name:20s} - {confidence:.4f} ({confidence*100:.2f}%)")
        print("="*50)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
