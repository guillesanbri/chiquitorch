import torch
import torch.nn as nn

from typing import List


class SimpleArithmetic(nn.Module):

    def forward(self, x, y):
        return x * 5.0 + y / 2.0


class AllInputKinds(nn.Module):
    """Exercises export InputKind(s): PARAMETER, BUFFER (persistent and
    non-persistent), CONSTANT_TENSOR and USER_INPUT (positional + keyword)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)  # parameters: lin.weight, lin.bias
        self.register_buffer("running", torch.zeros(3))  # persistent buffer
        self.register_buffer("tmp", torch.ones(3), persistent=False)  # non-persistent

    def forward(self, x, *, gain):
        const = torch.tensor([2.0, 2.0, 2.0])  # constant tensor
        return self.lin(x).relu() * gain + self.running + self.tmp + const


class StructuredOutputs(nn.Module):
    """Returns a nested output pytree (a dict with non-alphabetical keys plus
    a nested tuple) to exercise insertion-order flattening in the runners."""

    def forward(self, x):
        # tree_leaves order follows insertion: scaled, bias, sum, mean.
        return {
            "scaled": x * 2.0,
            "bias": x + 1.0,
            "pair": (x.sum(0), x.mean(0)),
        }


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()

        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
