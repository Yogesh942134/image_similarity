import os, faiss, torch, numpy as np
from PIL import Image
from torchvision import transforms
from models.embedding_model import EmbeddingNet

DATA_ROOT = r"D:\Projects\image_similarity\training\data\mydata"
MODEL_PATH = r"D:\Projects\image_similarity\models\embedding.pth"
FAISS_PATH = r"D:\Projects\image_similarity\models\faiss.index"
PATHS_SAVE = r"D:\Projects\image_similarity\models\image_paths.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmbeddingNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

embeddings = []
paths = []

print("Building FAISS index on your images...")

for cls in os.listdir(DATA_ROOT):
    for img_name in os.listdir(os.path.join(DATA_ROOT, cls)):
        path = os.path.join(DATA_ROOT, cls, img_name)
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(x).cpu().numpy()

        embeddings.append(emb)
        paths.append(path)

embeddings = np.vstack(embeddings).astype("float32")

index = faiss.IndexFlatL2(128)
index.add(embeddings)

os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
faiss.write_index(index, FAISS_PATH)
np.save(PATHS_SAVE, np.array(paths))

print("FAISS index built successfully on your images.")
