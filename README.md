image_similarity/
│
├── api/
│   └── main.py                    # FastAPI backend for similarity search
│
├── inference/
│   ├── build_index_mydata.py      # Builds FAISS vector database from your dataset
│   └── search.py                  # Real-time similarity search engine
│
├── models/
│   ├── embedding_model.py         # EfficientNet Triplet embedding model
│   ├── embedding.pth              # Trained embedding model
│   ├── faiss.index                # FAISS similarity index
│   └── image_paths.npy            # Image path mapping for search results
│
├── training/
│   ├── data/                      # Training images
│   ├── models/                    # Training checkpoints
│   ├── train.py                   # Model training pipeline
│   ├── triplet_mydata.py          # Triplet image generator
│   └── loss.py                    # Triplet loss function
│
├── temp/                          # Temporary query image storage
│
├── ui/
│   └── app.py                     # Streamlit user interface
│
├── requirements.txt
└── README.md

