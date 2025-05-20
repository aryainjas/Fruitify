from flask import Flask, render_template, request, jsonify # Added request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import io # Added io

# Load model and class names
try:
    model = load_model("mive-doost-dari?.h5", compile=True)
    with open("labels.txt", "r", encoding="utf-8") as f: # Ensure correct file opening and closing
        class_names = f.readlines()
except Exception as e:
    print(f"Error loading model or labels: {e}")
    model = None
    class_names = []

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if not model or not class_names:
        return jsonify({"error": "Model or labels not loaded properly"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            image = Image.open(file.stream).convert("RGB")
            
            # Preprocess the image
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array
            
            # Make prediction
            prediction_output = model.predict(data)
            
            # Get top 3 predictions
            top_indices = np.argsort(-prediction_output[0])[:3]
            
            results = []
            for i in top_indices:
                class_name_full = class_names[i].strip() # Remove newline
                # Remove leading number and space (e.g., "0 sib ghermez" -> "sghermez")
                actual_class_name = " ".join(class_name_full.split(" ")[1:]) 
                confidence_score = float(prediction_output[0][i]) # Convert to float
                results.append({"class_name": actual_class_name, "confidence_score": confidence_score})
            
            return jsonify({"predictions": results})

        except Exception as e:
            print(f"Error during prediction: {e}") # Log error for debugging
            return jsonify({"error": "Error processing image or making prediction", "details": str(e)}), 500
    
    return jsonify({"error": "Unknown error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
