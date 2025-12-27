# app.py
import streamlit as st
import requests
import mimetypes
from PIL import Image
import io
import os

# Set page config
st.set_page_config(
    page_title=" AI Image Similarity Search",
    page_icon="🔍",
    layout="wide"
)

# Configuration for different environments
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .similarity-badge {
        text-align: center;
        padding: 5px 10px;
        border-radius: 5px;
        margin-top: 5px;
        font-weight: bold;
    }
    .similarity-high { background-color: #4CAF50; color: white; }
    .similarity-medium { background-color: #FF9800; color: white; }
    .similarity-low { background-color: #F44336; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 AI Image Similarity Search Engine")
st.markdown("### Upload any image and get visually similar recommendations")

# File uploader with more options
uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP"
)

if uploaded:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### Query Image")

        # Display uploaded image
        image = Image.open(uploaded)
        st.image(image, width=150, caption="Uploaded Image")

        # Show file info
        st.caption(f"**File:** {uploaded.name}")
        st.caption(f"**Size:** {uploaded.size / 1024:.1f} KB")

    with col2:
        # Prepare for API call
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}

        with st.spinner("🔍 Finding similar images..."):
            try:
                # Make API request with timeout
                response = requests.post(
                    f"{BACKEND_URL}/search/",
                    files=files,
                    timeout=30  # 30 second timeout
                )

                if response.status_code == 200:
                    res = response.json()

                    if "error" in res:
                        st.error(f" Error: {res['error']}")
                        st.stop()

                    st.markdown("### Top 5 Similar Images")

                    # Create columns for results
                    cols = st.columns(5)

                    if "distances" in res and len(res["distances"]) > 0:
                        distances = res["distances"]

                        for i, img_path in enumerate(res["images"][:5]):
                            with cols[i]:
                                try:
                                    # Display image
                                    st.image(img_path, width=400)

                                except Exception as e:
                                    st.error(f"Error loading image {i + 1}")
                                    st.code(f"Path: {img_path}")
                    else:
                        # If no distances provided, just show images
                        for i, img_path in enumerate(res["images"][:5]):
                            cols[i].image(img_path, width=300)

                elif response.status_code == 400:
                    st.error(" Invalid file type. Please upload an image file (JPG, PNG, WEBP).")
                elif response.status_code == 500:
                    st.error(" Server error. Please try again later.")
                else:
                    st.error(f" Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(" Cannot connect to the backend server. Make sure it's running on port 8000.")
                st.info("To start the backend server, run: `python backend.py`")
            except requests.exceptions.Timeout:
                st.error(" Request timeout. The server is taking too long to respond.")
            except Exception as e:
                st.error(f" An error occurred: {str(e)}")
