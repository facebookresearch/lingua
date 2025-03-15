from torch.nn import Module, Parameter
from torch import tanh, zeros, ones

class DyT(Module):
    def __init__(self, C, init_α):
        super().__init__()
        self.α = Parameter(ones(1) * init_α)
        self.γ = Parameter(ones(C))
        self.β = Parameter(zeros(C))
    def forward(self, x):
        x = tanh(self.α * x)
        return self.γ * x + self.β
