import torch.nn as nn
import torchvision.models as models

class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        base.fc = nn.Linear(512, 128)   # 128-D embedding
        self.model = base

    def forward(self, x):
        return self.model(x)
