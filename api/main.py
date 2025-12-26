from fastapi import FastAPI, UploadFile, File
from inference.search import find_similar
import uuid, os, shutil

app = FastAPI(title="AI Image Similarity API")

@app.post("/search/")
async def search_image(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    path = f"temp/{uuid.uuid4().hex}.jpg"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    images, dist = find_similar(path, k=5)
    return {"images": images, "distances": dist.tolist()}

