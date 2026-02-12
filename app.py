import streamlit as st
from google import genai
from PIL import Image

# 1. Configuration
st.set_page_config(page_title="Wander Bin AI", page_icon="♻️")
st.title("♻️ Wander Bin - Smart Scanner")
st.caption("Powered by Gemini 2.0 Flash")

# 2. API Key Input
# Paste your NEW key (starting with AIza) here
api_key = st.text_input("Enter Google API Key:", type="password")

if not api_key:
    st.warning("⚠️ Please enter your API Key to start.")
    st.stop()

# 3. Configure the Client
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring API: {e}")
    st.stop()

# 4. Camera Input
img_file_buffer = st.camera_input("Take a photo of the trash")

if img_file_buffer is not None:
    # Display the image
    image = Image.open(img_file_buffer)
    st.image(image, caption="Captured Image", use_column_width=True)

    # 5. The Prompt
    # We ask Gemini to act as a recycling expert for Singapore
    prompt = """
    You are an expert recycling assistant for Singapore.
    Look at this image and identify the main object.
    
    Is it recyclable in Singapore's blue bins?
    
    Format your answer exactly like this:
    Object: [Name]
    Recyclable: [Yes/No]
    Material: [Plastic/Paper/Metal/Glass/General Waste]
    Action: [One short instruction, e.g. "Rinse and recycle" or "Throw in general waste"]
    """

    with st.spinner("Analyzing with Gemini 2.0 Flash..."):
        try:
            # Using the model from your list: models/gemini-2.0-flash
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image]
            )
            
            st.success("Analysis Complete!")
            st.markdown(f"### {response.text}")
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Double check that your API Key is correct and has access to this model.")
