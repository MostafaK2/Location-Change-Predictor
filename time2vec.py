import torch
import torch.nn as nn
import torch.nn.functional as F

class Time2Vec(nn.Module):
    def __init__(self, vector_size):
        super(Time2Vec, self).__init__()
        self.vector_size = vector_size

        # Linear[0] Periodic[1, .... n]
        self.l1 = nn.Linear(1, 1)
        self.periodic_term = nn.Linear(1, vector_size - 1)
        
    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        x0 = self.l1(t)
        x1n = torch.sin(self.periodic_term(t))
        return torch.cat([x0, x1n], dim=-1)
