import streamlit as st
import requests
import mimetypes
import warnings
warnings.filterwarnings("ignore")

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
    for i, img_path in enumerate(res["images"]):
        cols[i].image(img_path, width='stretch')

    # st.markdown("### Similarity Scores")
    # st.write(res["distances"])
