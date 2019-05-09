from torch import nn
import torch.nn.functional as F


class BCELoss2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bce_loss = nn.BCELoss(*args, **kwargs)

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        probs_flat = probs.view(-1)  # Flatten
        targets_flat = targets.view(-1)  # Flatten
        return self.bce_loss(probs_flat, targets_flat)
