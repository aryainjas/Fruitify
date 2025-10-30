"""
Fruitify - Batch Fruit Image Classification
Process multiple fruit images at once and save results to a CSV file.
"""
import argparse
import sys
import os
import csv
from pathlib import Path
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
        Preprocessed image array ready for prediction, or None if error
    """
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
        print(f"  Error processing {image_path}: {e}", file=sys.stderr)
        return None


def predict_fruit(model, image_data, class_names):
    """Predict the fruit class from preprocessed image data.
    
    Args:
        model: Loaded Keras model
        image_data: Preprocessed image array
        class_names: List of class names
        
    Returns:
        Tuple of (class_name, confidence_score)
    """
    prediction = model.predict(image_data, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    
    # Remove the index prefix (e.g., "0 " from "0 sib ghermez")
    if ' ' in class_name:
        class_name = class_name.split(' ', 1)[1]
    
    return class_name, confidence_score


def get_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
    """Get all image files from a directory.
    
    Args:
        directory: Directory path to search
        extensions: Tuple of valid image extensions
        
    Returns:
        List of image file paths
    """
    image_files = []
    for ext in extensions:
        image_files.extend(Path(directory).glob(f"*{ext}"))
        image_files.extend(Path(directory).glob(f"*{ext.upper()}"))
    return sorted(image_files)


def main():
    """Main function for batch fruit classification."""
    parser = argparse.ArgumentParser(
        description="Batch classify multiple fruit images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in current directory
  python batch_predict.py
  
  # Process all images in a specific directory
  python batch_predict.py --input /path/to/images
  
  # Save results to a specific CSV file
  python batch_predict.py --output results.csv
  
  # Process specific image files
  python batch_predict.py --files image1.jpg image2.png image3.jpg
        """
    )
    parser.add_argument(
        "--input",
        "-i",
        default=".",
        help="Directory containing images to classify (default: current directory)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="predictions.csv",
        help="Output CSV file for results (default: predictions.csv)"
    )
    parser.add_argument(
        "--files",
        "-f",
        nargs='+',
        help="Specific image files to process (overrides --input)"
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
        
        # Get image files
        if args.files:
            image_files = [Path(f) for f in args.files]
            print(f"Processing {len(image_files)} specified images...")
        else:
            print(f"Scanning directory: {args.input}")
            image_files = get_image_files(args.input)
            # Filter out sample images that come with the repo
            sample_images = {'goje.jpg', 'moz.png', 'porteghal.png', 'sibg.png'}
            image_files = [f for f in image_files if f.name not in sample_images]
            print(f"Found {len(image_files)} images to process...")
        
        if not image_files:
            print("No images found to process!")
            return 1
        
        # Process images and collect results
        results = []
        successful = 0
        failed = 0
        
        print(f"\nProcessing images...")
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] {image_path.name}...", end=' ')
            
            # Check if file exists
            if not image_path.exists():
                print(f"SKIP (not found)")
                failed += 1
                continue
            
            # Load and preprocess
            image_data = load_and_preprocess_image(str(image_path))
            if image_data is None:
                print(f"FAIL")
                failed += 1
                continue
            
            # Predict
            class_name, confidence = predict_fruit(model, image_data, class_names)
            results.append({
                'filename': image_path.name,
                'path': str(image_path),
                'predicted_class': class_name,
                'confidence': confidence
            })
            print(f"OK ({class_name}, {confidence:.2%})")
            successful += 1
        
        # Save results to CSV
        if results:
            print(f"\nSaving results to {args.output}...")
            with open(args.output, 'w', newline='') as csvfile:
                fieldnames = ['filename', 'path', 'predicted_class', 'confidence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"Results saved successfully!")
        
        # Summary
        print("\n" + "="*50)
        print(f"Summary:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Results saved to: {args.output}")
        print("="*50)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
