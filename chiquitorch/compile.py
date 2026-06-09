import torch

from pathlib import Path

def compile(model, inputs, output_dir="generated/"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    ep = torch.export.export(model, inputs)
    ep = ep.run_decompositions(decomp_table=None)

    print(ep.graph_module.graph)
    print(ep.state_dict)