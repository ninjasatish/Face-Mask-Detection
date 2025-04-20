import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Try loading the Keras-compatible model first, fallback to HDF5
try:
    model = load_model("mask_detector_model_compatible.keras", compile=False)
except:
    model = load_model("mask_detector_model_compatible.h5", compile=False)

# Load an image for testing
img_path = "test_image.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (64, 64))
img = np.expand_dims(img, axis=0) / 255.0  # Normalize

# Predict
prediction = model.predict(img)
label = "Mask" if np.argmax(prediction) == 0 else "No Mask"

# Show the image with prediction
cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
cv2.imshow("Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
