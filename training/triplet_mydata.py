import os, random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict

class TripletMyData(Dataset):
    def __init__(self, root="data/mydata"):
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        self.class_map = defaultdict(list)
        for cls in os.listdir(root):
            for img in os.listdir(os.path.join(root, cls)):
                self.class_map[cls].append(os.path.join(root, cls, img))

        self.classes = list(self.class_map.keys())

    def __getitem__(self, idx):
        cls = random.choice(self.classes)
        a, p = random.sample(self.class_map[cls], 2)

        neg_cls = random.choice([c for c in self.classes if c != cls])
        n = random.choice(self.class_map[neg_cls])

        return (
            self.transform(Image.open(a).convert("RGB")),
            self.transform(Image.open(p).convert("RGB")),
            self.transform(Image.open(n).convert("RGB"))
        )

    def __len__(self):
        return 50000
