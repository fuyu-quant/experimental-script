import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim):
        super(LearnableGatedPooling, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.gate_linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        weighted_x = x * self.weights
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(2)  # (batch_size, seq_len)
        gated_x = weighted_x * gate_values.unsqueeze(2)
        pooled_vector = torch.mean(gated_x, dim=1)  # average pooling operation
        return pooled_vector
