import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class ProjectionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, z):
        return self.proj(z)

class MultiViewContrastiveModel(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = BaseEncoder(in_dim, hidden_dim)
        self.proj_pos = ProjectionHead(hidden_dim)  # ⊕ view
        self.proj_neg = ProjectionHead(hidden_dim)  # Θ view

    def forward(self, x):
        z = self.encoder(x)              # Shared base representation
        z_pos = F.normalize(self.proj_pos(z), dim=1)  # Positive projection
        z_neg = F.normalize(self.proj_neg(z), dim=1)  # Negative projection
        return z, z_pos, z_neg
