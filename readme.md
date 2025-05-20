# Fruitify - Fun Fruit Recognizer Web App! 🍎🍌🍓

Fruitify is a project that uses a deep learning model to recognize various types of fruits from images. This version features a fun and interactive web dashboard that makes fruit classification easy and engaging!

[![CodeFactor](https://www.codefactor.io/repository/github/aryainjas/fruitify/badge)](https://www.codefactor.io/repository/github/aryainjas/fruitify)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Faryainjas%2FFruitify&count_bg=%2379C83D&title_bg=%23555555&icon=react.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)]()

<!-- Optional: Keep one or two representative images, or add a screenshot of the web app later -->
![moz](https://github.com/aryainjas/Fruitify/assets/36337300/5a13102a-59fc-4f43-b206-215d27e8abc1)

## ✨ Features

*   **Interactive Web Interface:** Easily upload fruit images through your browser.
*   **Real-time Predictions:** Get instant classification results.
*   **Top 3 Guesses:** See the model's top 3 fruit predictions with confidence scores.
*   **Fun & User-Friendly:** Designed to be engaging and simple to use!

## 🚀 Getting Started

### Prerequisites

*   Python 3.7+
*   pip (Python package installer)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aryainjas/Fruitify.git
    cd Fruitify
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The model (`mive-doost-dari?.h5`) and labels (`labels.txt`) are included in the repository.
    Install the required Python packages using:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Web Dashboard

1.  **Start the Flask application:**
    ```bash
    python app.py
    ```

2.  **Open your web browser:**
    Navigate to `http://127.0.0.1:5000`

3.  **Upload a fruit image and see the magic!** ✨

## 🍎 Fruit Name Translations
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

## 🧠 Model & Training Details
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
