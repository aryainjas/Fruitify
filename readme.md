# Fruitify

Fruitify is a simple fruit recognition project implemented in Python using the TensorFlow framework and Keras library. It allows you to predict the type of fruit present in an image.
[![CodeFactor](https://www.codefactor.io/repository/github/aryainjas/fruitify/badge)](https://www.codefactor.io/repository/github/aryainjas/fruitify)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Faryainjas%2FFruitify&count_bg=%2379C83D&title_bg=%23555555&icon=react.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)]()
![moz](https://github.com/aryainjas/Fruitify/assets/36337300/5a13102a-59fc-4f43-b206-215d27e8abc1)
#
![sib ghermez](https://github.com/aryainjas/Fruitify/assets/36337300/b83af9d0-2bd0-4c9a-8810-dbcdc83083dc)

## Translate:
#Apple (red) : sib (ghermez)
#Apple (yellow) : sib (zard)
#Banana : moz 
#Apricot : zardaloo
#Avocado : Avocado
#Tangerine : narengi
#Dates : khorma
#Kiwi : kiwi
#Lemon : limoo
#Mulberry : toot
#Peach fig : hooloo anjiri
#pear : golabi
#Orange : porteghal
#Pineapple : ananas
#tomato : goje

## Setup

To run this project, you need to follow these steps:

1. Install the required dependencies by running the following command:
```bash
pip install -r requirements.txt
```

Alternatively, you can install the packages individually:
```bash
pip install tensorflow keras pillow numpy
```

2. Ensure the trained model file (`mive-doost-dari?.h5`) and class labels file (`labels.txt`) are in the same directory as your Python scripts.

3. Run the classification script with your fruit image.

## Usage

The project now includes two enhanced scripts with command-line interfaces:

### Single Prediction (main.py)
Get the most likely fruit classification:

```bash
# Basic usage with default image (goje.jpg)
python main.py

# Classify a specific image
python main.py moz.png

# Use custom model and labels
python main.py --model custom_model.h5 --labels custom_labels.txt sibg.png

# Show help
python main.py --help
```

### Top 3 Predictions (top3.py)
Get the top 3 most likely classifications:

```bash
# Basic usage with default image (moz.png)
python top3.py

# Classify a specific image
python top3.py porteghal.png

# Get top 5 predictions instead of 3
python top3.py --top 5 goje.jpg

# Use custom model and labels
python top3.py --model custom_model.h5 --labels custom_labels.txt moz.png

# Show help
python top3.py --help
```

### Features:
- **Command-line argument support**: No need to edit code to change image paths
- **Error handling**: Clear error messages for missing files or invalid images
- **Flexible options**: Customize model, labels, and number of predictions
- **Better output formatting**: Clean, readable prediction results
- **Help documentation**: Built-in help with `--help` flag

### Legacy Code Example
If you prefer to use the code programmatically, here's the original approach:

```python
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the pre-trained model
model = load_model("mive-doost-dari?.h5", compile=True)

# Read the class names from the labels file
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Prepare the input image
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open("moz.png").convert("RGB")  # Replace with your fruit photo path
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
    class_name = class_names[i].strip()
    confidence_score = prediction[0][i]
    # Remove index prefix from class name
    if ' ' in class_name:
        class_name = class_name.split(' ', 1)[1]
    print(f"Class: {class_name}, Confidence Score: {confidence_score:.4f}")
```

## Training Details

The dataset for this project has been pre-trained using the following configurations:

- Epochs: 100
- Batch Size: 32
- Learning Rate: 0.001
I utilized the provided code below to train my model.
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_directory = '/path/to/images'
labels_file = '/path/to/labels.txt'

with open(labels_file, 'r') as f:
    labels = f.read().splitlines()

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  
)

train_generator = datagen.flow_from_directory(
    image_directory,
    target_size=(224, 224),  
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    image_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = tf.keras.models.Sequential([
    tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=100
)

model.save('mive-doost-dari?.h5')
with open('labels.txt', 'w') as f:
    f.write('\n'.join(labels))
 ```
These parameters were used during the training process to optimize the model's performance in recognizing various fruit types.
The dataset used for training in this project consists of 9,478 photos.
For the training of the dataset, approximately half of the photos were sourced from open data and publicly available datasets. The remaining photos were captured and collected specifically for this project, ensuring a diverse and comprehensive collection of fruit images. Additionally, new photos were created from the existing images to further augment the dataset and enhance the training process.
![conf-matrix](https://github.com/aryainjas/Fruitify/assets/36337300/cd1e98fd-6b73-4c19-b367-5ee763827eb0)

