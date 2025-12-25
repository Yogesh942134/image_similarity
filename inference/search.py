import faiss, torch, numpy as np
from PIL import Image
from torchvision import transforms
from models.embedding_model import EmbeddingNet

MODEL_PATH = r"models\embedding.pth"
FAISS_PATH = r"models\faiss.index"
PATHS_PATH = r"models\image_paths.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = EmbeddingNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# Load FAISS index + image paths
index = faiss.read_index(FAISS_PATH)
image_paths = np.load(PATHS_PATH)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def find_similar(image_path, k=5):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(x).cpu().numpy().astype("float32")

    D, I = index.search(emb, k)

    results = [image_paths[i] for i in I[0]]
    return results, D[0]
