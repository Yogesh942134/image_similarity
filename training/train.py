import torch, os
from torch.utils.data import DataLoader
from training.triplet_mydata import TripletMyData
from training.loss import TripletLoss
from models.embedding_model import EmbeddingNet
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = TripletMyData(root="data/mydata")

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,   # keep 4
        pin_memory=True
    )

    model = EmbeddingNet().to(device)
    criterion = TripletLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(10):
        total_loss = 0
        for a,p,n in tqdm(loader):
            a,p,n = a.to(device),p.to(device),n.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = criterion(model(a),model(p),model(n))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

    CHECKPOINT = "models/embedding.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/embedding.pth")
    print("Model trained and saved.")


if __name__ == "__main__":
    main()
