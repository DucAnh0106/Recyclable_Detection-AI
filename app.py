import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# 1. Page Config
st.set_page_config(page_title="Wander Bin AI Test", page_icon="♻️")
st.title("♻️ Wander Bin - R&D Mode")
st.write("Hold trash in front of your webcam to test the AI.")

# 2. Load the Model (with caching so it doesn't reload every frame)
@st.cache_resource
def load_model():
    # Make sure this points to your BEST model
    return YOLO('runs/detect/train2/weights/best.pt')

try:
    model = load_model()
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Confidence Slider
confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# 4. Camera Loop
# We use a checkbox to stop the camera so you can close it properly
run_camera = st.checkbox("Start Camera", value=True)
FRAME_WINDOW = st.image([]) # Create an empty placeholder

camera = cv2.VideoCapture(1) # 0 is usually the default webcam

while run_camera:
    ret, frame = camera.read()
    if not ret:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- CHANGE THIS LINE ---
    # conf=0.55: "Only speak if you are 55% sure." (Filters out weak guesses)
    # verbose=False: "Don't print logs to the terminal." (Stops the spam)
    results = model.predict(frame, conf=0.55, verbose=False) 
    
    annotated_frame = results[0].plot()
    FRAME_WINDOW.image(annotated_frame)

# Release camera when checkbox is unchecked
camera.release()
