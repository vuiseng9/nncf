from collections import OrderedDict
import math
import torch
from torch import nn
import torch.nn.functional as F

class GLF_unitC(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.A = nn.Parameter(torch.rand(1, device=device))
        self.B = nn.Parameter(torch.rand(1, device=device))
        self.K = nn.Parameter(torch.rand(1, device=device))
        self.Q = nn.Parameter(torch.rand(1, device=device))
        self.V = nn.Parameter(torch.rand(1, device=device))
    
    def forward(self, x):
        term1 = self.A
        nominator_term2 = self.K - self.A
        denominator_term2 = (1 + self.Q*(torch.exp(-1*self.B*x))) ** (1/self.V)
        return term1 + (nominator_term2/denominator_term2)
        
    def set_coef(self,
        A=-1, B=1, K=1, Q=1, V=1):
        one=torch.tensor([1.], device=self.device)
        sd={
            'A': one*A,
            'B': one*B,
            'K': one*K,
            'Q': one*Q,
            'V': one*V,
        }
        self.load_state_dict(sd)

    def __repr__(self):
        return "GLF_unitC: {}".format(self.state_dict())