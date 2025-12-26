import streamlit as st
import requests
import mimetypes
import warnings

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(page_title="AI Image Similarity Search", layout="wide")


st.title("🔍 AI Image Similarity Search Engine")
st.markdown("### Upload any image and get visually similar recommendations")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    st.markdown("### Query Image")
    st.image(uploaded, width=300)

    mime_type, _ = mimetypes.guess_type(uploaded.name)
    if mime_type is None:
        mime_type = "application/octet-stream"

    files = {"file": (uploaded.name, uploaded.getvalue(), mime_type)}

    with st.spinner("Finding similar images..."):
        response = requests.post("http://127.0.0.1:8000/search/", files=files)
        res = response.json()

    if "error" in res:
        st.error(res["error"])
        st.stop()

    st.markdown("### Top 5 Similar Images")

    cols = st.columns(5)

    # Check if distances are in the response
    if "distances" in res and len(res["distances"]) > 0:
        distances = res["distances"]

        for i, img_path in enumerate(res["images"][:5]):  # Show only top 5
            with cols[i]:
                # Display image
                st.image(img_path,width=300)

                # Display similarity percentage
                if i < len(distances):
                    # Convert distance to similarity percentage
                    # Assuming distance is between 0-1 where lower = more similar
                    similarity = max(0, 100 - (distances[i] * 100))
                    similarity = round(similarity, 1)
                    color = "green"
                    st.markdown(
                        f'<div style="text-align: center; background-color: {color}; '
                        f'color: white; padding: 5px; border-radius: 5px; margin-top: 5px;">'
                        f'Similarity: {similarity}%</div>',
                        unsafe_allow_html=True
                    )
    else:
        # If no distances provided, just show images
        for i, img_path in enumerate(res["images"][:5]):
            cols[i].image(img_path, width='stretch')