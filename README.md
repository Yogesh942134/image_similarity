
## Detailed Directory Overview

### Backend API (`api/`)
| File | Description |
|------|-------------|
| `main.py` | FastAPI server providing RESTful endpoints for image similarity search |

###  Inference Engine (`inference/`)
| File | Description |
|------|-------------|
| `build_index_mydata.py` | Builds FAISS index from image embeddings |
| `search.py` | Real-time similarity search against the vector database |

###  Models (`models/`)
| File | Description |
|------|-------------|
| `embedding_model.py` | Custom EfficientNet architecture with triplet embedding |
| `embedding.pth` | Pre-trained model weights |
| `faiss.index` | FAISS vector index for fast similarity search |
| `image_paths.npy` | Mapping between vectors and original image paths |

### Training Pipeline (`training/`)
| File | Description |
|------|-------------|
| `train.py` | Main training script with monitoring |
| `triplet_mydata.py` | Triplet data loader and augmentation |
| `loss.py` | Custom triplet loss implementation |
| `data/` | Training dataset organized by classes |
| `models/` | Training checkpoints and saved models |

###  Frontend UI (`ui/`)
| File | Description |
|------|-------------|
| `app.py` | Streamlit web interface with image upload and visualization |

###  Utilities
| Directory | Description |
|-----------|-------------|
| `temp/` | Temporary storage for uploaded query images |

