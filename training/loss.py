import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, a, p, n):
        pos = F.pairwise_distance(a, p)
        neg = F.pairwise_distance(a, n)
        return torch.mean(torch.clamp(pos - neg + self.margin, min=0))
