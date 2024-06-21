import streamlit as st
import cv2
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Create upload and result directories if they don't exist
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the YOLO model
model = YOLO('yolov8n.pt')

def detect_objects(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    object_counts = {}
    for result in results:
        for cls in result.boxes.cls:
            label = model.names[int(cls)]
            object_counts[label] = object_counts.get(label, 0) + 1
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            label = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, image)
    return object_counts, result_path

# Streamlit app
st.title('Upload an Image for Object Detection and counting')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    input_image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image with a fixed width
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=False, width=300)

    # Detect objects in the uploaded image
    object_counts, output_image_path = detect_objects(input_image_path)

    # Display object counts
    st.subheader("Object Counts")
    for label, count in object_counts.items():
        st.write(f"{label}: {count}")

    # Display the output image with a fixed width
    output_image = Image.open(output_image_path)
    st.image(output_image, caption='Output Image', use_column_width=False, width=300)
