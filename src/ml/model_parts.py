import torch, torch.nn as nn

class Reshape(nn.Module): # nn.Unflatten in new versions does the same
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(-1, *self.shape)
    def extra_repr(self):
        return str(self.shape)

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
    def forward(self, x):
        return nn.functional.normalize(x)

class GaussianNoise(nn.Module):
    def __init__(self, nl):
        super().__init__()
        self.nl = nl
    def forward(self, x):
        noise = torch.zeros(x.shape, device=x.device).normal_() # mean=0., std=1.
        noise *= self.nl
        return x + noise