# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:05:47 2024

@author: Tayyab
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2

st.title("Webcam Test")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
