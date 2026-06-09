import torch

from chiquitorch import compile

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

model = DummyModel()
x = torch.randn(1, 8)
exported_program = torch.export.export(model, (x,))

# at this point we have an FX graph with ATen ops
graph = exported_program.graph_module.graph
state_dict = exported_program.state_dict

print(graph)
print(state_dict)
print("="*80)

compile(model, (x,))