import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDropoutLinear(nn.Module):
    def __init__(self, in_features, out_features, n_drops=5, dropout_rate=0.5):
        super().__init__()
        self.drops = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(n_drops)])
        self.fc = nn.Linear(in_features, out_features)
        self.n_drops = n_drops

    def forward(self, x):
        if self.training:
            out = sum(self.fc(drop(x)) for drop in self.drops) / self.n_drops
            return out
        else:
            return self.fc(x)


class AttentionWeighting(nn.Module):
    def __init__(self, n_features, pattern="A"):
        super().__init__()
        self.pattern = pattern
        if pattern == "A":
            self.att_layer = nn.Linear(n_features, 1)
        elif pattern == "B":
            self.att_layer = nn.Sequential(
                nn.Linear(n_features, 256), nn.Tanh(), nn.Linear(256, 1)
            )
        else:
            raise ValueError(f"Invalid attention pattern: {pattern}")

    def forward(self, features_list):
        stacked = torch.stack(features_list, dim=0)
        att_scores = torch.stack(
            [self.att_layer(feat) for feat in features_list], dim=0
        )
        att_weights = F.softmax(att_scores, dim=0)
        weighted_features = (stacked * att_weights).sum(dim=0)
        return weighted_features


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight

    def forward(self, preds, target):
        n_class = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.epsilon / (n_class - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.epsilon)
        if self.weight is not None:
            true_dist = true_dist * self.weight.unsqueeze(0)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))
