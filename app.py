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

st.title("Object Detection with YOLOv8")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="object_detection",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
