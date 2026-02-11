import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Page Config
st.set_page_config(page_title="Wander Bin AI", page_icon="♻️")

# --- CSS HACK FOR TARGET BOX ---
# This draws a red box in the center of the camera widget!
st.markdown(
    """
    <style>
    /* This targets the camera container */
    div[data-testid="stCameraInput"] video {
        clip-path: inset(0% 0% 0% 0%); /* specific to some browsers */
    }
    /* The Target Box Overlay */
    div[data-testid="stCameraInput"]::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 300px; /* Width of the target box */
        height: 300px; /* Height of the target box */
        transform: translate(-50%, -50%);
        border: 4px solid #00FF00; /* Green Border */
        border-radius: 10px;
        z-index: 99;
        pointer-events: none; /* Let clicks pass through */
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5); /* Dim the outside area */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("♻️ Wander Bin - Scanner")
st.caption("Place the object inside the GREEN BOX and press 'Take Photo'.")

# 2. Load Model
@st.cache_resource
def load_model():
    return YOLO('models/wander_bin_v1.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Camera Input
img_file_buffer = st.camera_input("Scanner", label_visibility="hidden")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
    
    # --- SMART CROP LOGIC ---
    # We crop exactly the center 300x300 pixels to match the CSS box
    height, width, _ = img_array.shape
    crop_size = 300
    
    start_y = max(0, (height - crop_size) // 2)
    start_x = max(0, (width - crop_size) // 2)
    end_y = min(height, start_y + crop_size)
    end_x = min(width, start_x + crop_size)
    
    cropped_image = image.crop((start_x, start_y, end_x, end_y))
    
    # Show the user what the AI saw
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(cropped_image, caption="AI Input", width=150)

    # --- INFERENCE ---
    results = model.predict(cropped_image, conf=0.30) # Lower confidence allowed

    with col2:
        if len(results[0].boxes) > 0:
            top_box = results[0].boxes[0]
            class_name = results[0].names[int(top_box.cls[0])]
            conf = float(top_box.conf[0])
            
            # FORMATTED OUTPUT
            if class_name in ['can', 'plastic_bottle', 'glass']:
                st.success(f"### ✅ RECYCLABLE: {class_name.upper()}")
            else:
                st.info(f"### 📄 RECYCLABLE: {class_name.upper()}")
                
            st.metric("Confidence", f"{conf:.1%}")
        else:
            st.warning("❓ No clear object detected.")
            st.write("Try rotating the object or turning on a light.")
