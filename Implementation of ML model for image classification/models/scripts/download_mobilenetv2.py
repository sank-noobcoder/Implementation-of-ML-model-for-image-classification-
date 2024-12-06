import os
from tensorflow.keras.applications import MobileNetV2

# Directory to save the model
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# Download MobileNetV2 and save it in HDF5 format (.h5)
mobilenet_model = MobileNetV2(weights='imagenet')
mobilenet_model.save(os.path.join(model_dir, 'mobilenetv2.h5'))

print("MobileNetV2 model downloaded and saved successfully!")
