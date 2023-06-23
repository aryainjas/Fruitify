# Fruitify

Fruitify is a simple fruit recognition project implemented in Python using the TensorFlow framework and Keras library. It allows you to predict the type of fruit present in an image.

## Setup

To run this project, you need to follow these steps:

1. Install the required dependencies by running the following command:
pip install tensorflow keras pillow


2. Download the trained model file (`mive-doost-dari?.h5`) and class labels file (`labels.txt`) to the same directory as your Python script.

3. Replace the placeholder image path (`"moz.png"`) in the code with the path to your fruit photo that you want to classify.

4. Execute the Python script to see the predictions.

## Usage

After setting up the project, you can use the following code snippet to classify a fruit image:

```python
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the pre-trained model
model = load_model("keras_Model.h5", compile=True)

# Read the class names from the labels file
class_names = open("labels.txt", "r").readlines()

# Prepare the input image
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open("moz.png").convert("RGB")  # Replace "moz.png" with your fruit photo address
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data[0] = normalized_image_array

# Perform the prediction
prediction = model.predict(data)
top_indices = np.argsort(-prediction[0])[:3]

# Print the top predictions
print("Top Predictions:")
for i in top_indices:
 class_name = class_names[i]
 confidence_score = prediction[0][i]
 print("Class:", class_name[2:], "Confidence Score:", confidence_score)
```
Make sure to replace "moz.png" with the actual path to your fruit photo.

## Training Details

The dataset for this project has been pre-trained using the following configurations:

- Epochs: 100
- Batch Size: 32
- Learning Rate: 0.001

These parameters were used during the training process to optimize the model's performance in recognizing various fruit types.
The dataset used for training in this project consists of 9,478 photos.
For the training of the dataset, approximately half of the photos were sourced from open data and publicly available datasets. The remaining photos were captured and collected specifically for this project, ensuring a diverse and comprehensive collection of fruit images. Additionally, new photos were created from the existing images to further augment the dataset and enhance the training process.
