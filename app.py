# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:05:47 2024

@author: Tayyab
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from PIL import Image
import queue

# Set page config
st.set_page_config(layout="wide", page_title="Detectify - AI Object Detection", page_icon="🔍")

# YOLO Model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Object Classes
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
st.markdown("Detect objects in images or through your webcam using AI technology.")

# Sidebar
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05,
                                         help="Adjust the confidence threshold for object detection")
selected_classes = st.sidebar.multiselect("Select classes to detect", classNames, default=classNames,
                                          help="Choose which types of objects you want to detect")

# Choose between webcam and image upload
option = st.radio("Select input source:", ("Use Webcam", "Upload Image"))

def process_frame(frame):
    results = model(frame, stream=True)
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
    st.markdown("Click 'Start' to begin real-time object detection using your webcam.")
    
    result_queue = queue.Queue()
    
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed_img, detections = process_frame(img)
        result_queue.put(detections)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            try:
                result = result_queue.get(timeout=1.0)
                df = pd.DataFrame(list(result.items()), columns=["Object", "Count"])
                df = df.sort_values("Count", ascending=False).reset_index(drop=True)
                labels_placeholder.table(df)
            except queue.Empty:
                continue

elif option == "Upload Image":
    st.markdown("### Image Upload Object Detection")
    st.markdown("Upload an image to perform object detection.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        with st.spinner("Detecting objects..."):
            img_array = np.array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_array, detections = process_frame(img_array)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        st.image(img_array, caption='Detected Objects', use_column_width=True)

        if detections:
            st.markdown("### Detection Summary")
            df = pd.DataFrame(list(detections.items()), columns=['Object', 'Count'])
            df = df.sort_values('Count', ascending=False).reset_index(drop=True)
            st.dataframe(df, use_container_width=False)
        else:
            st.warning("No objects detected in the image.")

# Add footer
st.markdown("---")
st.markdown("Developed using Streamlit and YOLO")
