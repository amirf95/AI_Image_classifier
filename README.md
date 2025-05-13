# 🤖 Image Classification App using Streamlit and MobileNetV2

This project is a simple web application that allows users to upload an image and get predictions using the MobileNetV2 deep learning model. It's built with Streamlit, a fast way to build data apps in Python.

---

## 📦 Requirements

- Python 3.9 to 3.11 (TensorFlow does not support 3.12 yet)
- TensorFlow
- Streamlit
- OpenCV
- Pillow
- NumPy

Install dependencies:

pip install tensorflow streamlit opencv-python pillow numpy

---

## ▶️ Run the App

Make sure you're in your project directory and run:

streamlit run main.py

---

## 🧠 File: main.py — Code Explanation

### 📌 Imports:

import cv2                        # For image resizing (OpenCV)
import numpy as np               # For array operations
import streamlit as st           # For building the interactive web UI
from PIL import Image            # To handle image uploads

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,                 # Pre-trained CNN model
    preprocess_input,           # Preprocessing function for MobileNetV2
    decode_predictions          # Decodes raw model output into class labels
)

---

### 🔄 Functions:

#### 1. load_model()
Loads the pre-trained MobileNetV2 model with ImageNet weights.

def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

---

#### 2. preprocess_image(image)
Prepares the uploaded image for MobileNetV2 input.

def preprocess_image(image):
    img = np.array(image)                   # Convert from PIL image to NumPy array
    img = cv2.resize(img, (224, 224))       # Resize to 224x224 as required by MobileNetV2
    img = np.expand_dims(img, axis=0)       # Add batch dimension (1, 224, 224, 3)
    img = preprocess_input(img)             # Normalize pixel values as required
    return img

---

#### 3. classify_image(model, image)
Takes the image, processes it, makes a prediction, and returns top 3 labels with confidence scores.

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error in classification: {e}")
        return None

---

### 🖼️ Main App UI

#### main()
This function creates the Streamlit interface: title, image uploader, and button for classification.

def main():
    st.set_page_config(page_title="Image Classification App", page_icon="🤖", layout="centered")

    st.title("Image Classification App")
    st.write("Upload an image to classify it using MobileNetV2.")

    @st.cache_resource
    def load_cached_model():
        return load_model()

    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        btn = st.button("Classify Image")
        if btn:
            with st.spinner("Classifying..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)
                if predictions:
                    st.subheader("Predictions:")
                    for _, label, score in predictions:
                        st.write(f"{label}: {score:.2%}")

if __name__ == "__main__":
    main()

---

## 🧪 Example Output

If you upload an image of a dog, you might see:

Predictions:
Labrador_retriever: 85.23%
golden_retriever: 10.45%
kuvasz: 2.34%

---

## 🙋 FAQ

Q: Why MobileNetV2?  
A: It's fast, lightweight, and trained on ImageNet — ideal for real-time classification in web apps.

Q: Can I use another model like ResNet or EfficientNet?  
A: Yes! Just import from the appropriate submodule and adjust preprocess_input() and decode_predictions() accordingly.

---

## 🚀 Future Ideas

- Add webcam support  
- Plot predictions as bar chart  
- Deploy on Streamlit Cloud  
- Add model confidence threshold

---

## 👩‍💻 Author

Built by Emir — Software Engineering student passionate about AI & bots & real-world applications.
