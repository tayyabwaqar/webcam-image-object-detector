# AI-Powered Object Detection App

## Overview

The **AI-Powered Object Detection App** is a web application built with Streamlit and YOLOv8, designed for real-time object detection in images and video streams. This app leverages AI technology to identify and label various objects in both uploaded images and live webcam feeds.

## Live Demo

You can try the live demo of the application at the following link: 
[Live Demo - Detectify](https://detectifyai.streamlit.app/)

## Key Features

- **Real-Time Detection**: Utilize your webcam to detect and classify objects in real-time, with instant feedback on detected items.
- **Image Upload**: Easily upload images to perform object detection, with results displayed alongside the original image.
- **Customizable Settings**: Adjust the confidence threshold and select specific object classes to tailor detection to your needs.
- **Interactive Interface**: A clean and intuitive user interface that allows for seamless interaction and visualization of detection results.
- **Detection Summary Table**: View a summary of detected objects, including counts for each type, presented in a neatly formatted table.

## Technologies Used

- **Streamlit**: For creating the interactive web interface.
- **OpenCV**: For image processing and handling webcam input.
- **YOLOv8**: For advanced object detection capabilities.
- **Pandas**: For creating and manipulating the detection summary dataframe.
- **streamlit-webrtc**: For handling real-time video streaming from the webcam.

## Installation

To run the app locally, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
streamlit run app.py
