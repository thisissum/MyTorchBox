import torch
from torch import nn


class Highway(nn.Module):
    """Highway implementation, which allows info follow to deeper layer
    """

    def __init__(self, hidden_dim, num_layers=1, dropout=0.5):
        super(Highway, self).__init__()
        assert num_layers >= 1, "num_layers must not less than 1"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers)]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # add relu to get residual
        transformed_x = torch.relu(self.linears[0](x))
        gated_x = torch.sigmoid(self.gates[0](x))  # add sigmoid to get gate
        out = transformed_x * gated_x + (1 - gated_x) * transformed_x

        for i in range(1, len(self.num_layers)):
            out = self.dropout(out)  # add dropout if num_layers bigger than 1
            transformed_out = torch.relu(self.linears[i](out))
            gated_out = torch.sigmoid(self.linears[i](out))
            out = transformed_out * gated_out + (1 - gated_out) * out

        return out
