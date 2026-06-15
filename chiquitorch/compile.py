import torch

from pathlib import Path


def compile(model, args, kwargs=None, output_dir="generated/"):
    # output_dir = Path(output_dir)
    # output_dir.mkdir(exist_ok=True)

    ep = torch.export.export(model, args, kwargs)
    # use the Core ATen decomp table
    ep = ep.run_decompositions(decomp_table=None)
    return ep

    # print(ep.graph_module.graph)
    # print(ep.state_dict)
