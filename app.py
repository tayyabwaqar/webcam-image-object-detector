# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:05:47 2024

@author: Tayyab
"""

import streamlit as st
import cv2
import numpy as np
from collections import deque
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import time

# Set page config at the very beginning
st.set_page_config(layout="wide", page_title="AI Object Detection", page_icon="🔍")

# Custom CSS to improve app appearance
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #4CAF50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# YOLO Model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Object Classes and Colors
classNames = ["Person", "Bicycle", "Car", "Motorbike", "Aeroplane", "Bus", "Train", "Truck", "Boat",
              "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat",
              "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella",
              "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat",
              "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle", "Wine Glass", "Cup",
              "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli",
              "Carrot", "Hot dog", "Pizza", "Donut", "Cake", "Chair", "Sofa", "Potted Plant", "Bed",
              "Dining Table", "Toilet", "TV Monitor", "Laptop", "Mouse", "Remote", "Keyboard", "Mobile Phone",
              "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors",
              "Teddy Bear", "Hair Dryer", "Toothbrush"
              ]

# Generate a color for each class
np.random.seed(42)
colors = {name: tuple(map(int, np.random.randint(0, 255, 3))) for name in classNames}

st.title("AI-Powered Object Detection")
st.markdown("Detect objects in images or through your webcam using state-of-the-art AI technology.")

# Sidebar
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05,
                                         help="Adjust the confidence threshold for object detection")
selected_classes = st.sidebar.multiselect("Select classes to detect", classNames, default=classNames,
                                          help="Choose which types of objects you want to detect")

# Choose between webcam and image upload
option = st.radio("Select input source:", ("Upload Image", "Use Webcam"),
                  help="Choose whether to upload an image or use your webcam for object detection")

def process_frame(frame, results):
    detections = {}
    for r in results:
        boxes = r.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            class_index = int(box.cls[0])
            class_name = classNames[class_index]
            
            if confidence > confidence_threshold and class_name in selected_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = colors[class_name]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw confidence and class name
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Count detections
                if class_name in detections:
                    detections[class_name] += 1
                else:
                    detections[class_name] = 1
    
    return frame, detections

if option == "Use Webcam":
    st.markdown("### Webcam Object Detection")
    st.markdown("Click 'Start' to begin real-time object detection using your webcam. Click 'Stop' to end the session.")
    
    # Create two columns for Start and Stop buttons
    col1, col2 = st.columns(2)
    start_button = col1.button('Start')
    stop_button = col2.button('Stop')

    # Initialize session state to keep track of webcam status
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False

    FRAME_WINDOW = st.image([])

    # For FPS calculation
    frame_times = deque(maxlen=30)

    if start_button:
        st.session_state.webcam_running = True
        # Try to open the camera
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Unable to access the webcam. This feature may not be available in the current environment.")
            st.session_state.webcam_running = False

    if stop_button:
        st.session_state.webcam_running = False
        if 'camera' in locals():
            camera.release()
        st.write('Webcam stopped')

    while st.session_state.webcam_running:
        try:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            start_time = time.time()
            
            results = model(frame, stream=True)
            frame, detections = process_frame(frame, results)

            # Calculate FPS
            end_time = time.time()
            frame_time = end_time - start_time
            frame_times.append(frame_time)
            fps = 1 / (sum(frame_times) / len(frame_times))

            # Update statistics
            st.sidebar.text(f"FPS: {fps:.2f}\nDetections: {sum(detections.values())}")

            FRAME_WINDOW.image(frame)

            # Check if the stop button was clicked
            if stop_button:
                break
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            break

    if not st.session_state.webcam_running:
        st.write('Webcam is not running. Click "Start" to begin.')
        if 'camera' in locals():
            camera.release()

elif option == "Upload Image":
    st.markdown("### Image Upload Object Detection")
    st.markdown("Upload an image to perform object detection.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        with st.spinner("Detecting objects..."):
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Ensure the image is in RGB format
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            results = model(img_array, stream=True)
            
            img_array, detections = process_frame(img_array, results)

        st.image(img_array, caption='Detected Objects', use_column_width=True)

        # Create and display summary table
        if detections:
            st.markdown("### Detection Summary")
            df = pd.DataFrame(list(detections.items()), columns=['Object', 'Count'])
            df = df.sort_values('Count', ascending=False).reset_index(drop=True)

            # Calculate column widths based on content
            max_object_length = max(df['Object'].apply(len).max(), len('Object'))
            max_count_length = max(df['Count'].apply(lambda x: len(str(x))).max(), len('Count'))

            # Style the DataFrame
            def style_df(df):
                return df.style.set_table_attributes('style="margin: auto;"') \
                               .set_caption("Detected Objects Summary") \
                               .set_table_styles([{
                                   'selector': 'thead th',
                                   'props': [('background-color', '#4CAF50'), ('color', 'white')]
                               }, {
                                   'selector': 'tbody td',
                                   'props': [('text-align', 'center')]
                               }]) \
                               .set_properties(**{
                                   'Object': [f'width: {max_object_length * 10}px'],
                                   'Count': [f'width: {max_count_length * 10}px']
                               })

            # Display the styled DataFrame
            st.dataframe(style_df(df), use_container_width=False)
        else:
            st.warning("No objects detected in the image.")

# Add footer
st.markdown("---")
st.markdown("Developed using Streamlit and YOLO")
