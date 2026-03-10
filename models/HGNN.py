import torch
import torch.nn as nn

class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X


class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")

    def forward(self, x, H):
        x = self.fc(x)
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        x = self.e2v(E, H)
        return x


class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.threshold = 0.2
        self.hgconv = HyPConv(c1, c2)
        self.bn = nn.BatchNorm1d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        feature = x.clone()
        distance = torch.cdist(feature, feature)
        hg = (distance < self.threshold).float().to(x.device).to(x.dtype)
        x = self.hgconv(x, hg).to(x.device).to(x.dtype) + x
        x = x.transpose(1, 2).contiguous()
        x = self.act(self.bn(x))
        return x