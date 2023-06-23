from keras.models import load_model  
from PIL import Image, ImageOps  
import numpy as np
model = load_model("mive-doost-dari?.h5", compile=True)
class_names = open("labels.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open("goje.jpg").convert("RGB") #change the "goje.jpg" to your fruit photo address , if it is not in the same directory.
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data[0] = normalized_image_array
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
