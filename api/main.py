import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from inference.search import find_similar

app = FastAPI(title="AI Image Similarity API")

# Configure CORS for both local and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory if it doesn't exist
os.makedirs("temp", exist_ok=True)


@app.post("/search/")
async def search_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
        if file.content_type not in allowed_types:
            return JSONResponse(
                status_code=400,
                content={"error": f"File type not supported. Allowed types: {', '.join(allowed_types)}"}
            )

        # Generate unique filename
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        filename = f"{uuid.uuid4().hex}{ext}"
        path = f"temp/{filename}"

        # Save uploaded file
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Find similar images
        images, dist = find_similar(path, k=5)

        # Clean up temp file
        try:
            os.remove(path)
        except:
            pass

        return {"images": images, "distances": dist.tolist()}

    except Exception as e:
        # Clean up on error
        if 'path' in locals() and os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"}
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# For local development
if __name__ == "__main__":
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        reload=True  # Auto-reload on code changes
    )