import streamlit as st
import numpy as np
import pickle
from PIL import Image
import cv2

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title
st.title("🧠 Handwritten Digit Recognition")
st.write("Upload a digit image (28x28) and click 'Predict' to see the result.")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a digit (0-9)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image', width=150)

    # Predict button
    if st.button("Predict"):
        # Preprocess image
        img = np.array(image)
        img = cv2.resize(img, (28, 28))  # Resize to 28x28
        img = img / 255.0  # Normalize
        img = img.reshape(1, 28, 28, 1)  # Reshape for CNN input

        # Predict
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)

        st.success(f"✅ Predicted Digit: **{predicted_digit}**")
