import streamlit as st
import pandas as pd
import math
from pathlib import Path

import cv2
import pytesseract
from PIL import Image
import numpy as np


# Set up pytesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract.exe'


# Function to detect number plate using OpenCV and pytesseract
def detect_number_plate(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # Detect edges
    edged = cv2.Canny(filtered, 30, 200)

    # Find contours based on edges
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Loop through the contours and look for a rectangular shape that could be a license plate
    number_plate = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:  # Assuming the plate is rectangular
            number_plate = approx
            break

    if number_plate is None:
        return None, "Number plate not detected"

    # Mask everything except the detected plate
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [number_plate], -1, 255, -1)

    # Extract the number plate area
    out = cv2.bitwise_and(image, image, mask=mask)

    # OCR on the number plate region
    img = Image.open('2.jpg')  # Replace with an actual image
    text = pytesseract.image_to_string(img)
    print(text)    
    return out, text.strip()

# Streamlit UI
st.title("OCR Number Plate Detection")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Detect number plate and show result
    plate_image, detected_text = detect_number_plate(image)

    if plate_image is not None:
        st.image(plate_image, caption="Detected Number Plate", use_column_width=True)
        st.write(f"Detected Text: {detected_text}")
    else:
        st.write("Number plate not detected")