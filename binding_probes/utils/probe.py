import torch.nn as nn
import torch
import math

class QuadraticProbe(nn.Module):
    '''IsSameObject(x,y) = sigmoid(x^T W y + b). Ideally, rank(W) < C'''
    def __init__(self, in_dim):
        super(QuadraticProbe, self).__init__()
        self.W = nn.Linear(in_dim, in_dim)
        
    def forward(self, x, y):
        W_sym = (self.W.weight + self.W.weight.T) / 2 / math.sqrt(x.shape[1])
        out = x @ W_sym @ y.T + self.W.bias[0]
        return out
