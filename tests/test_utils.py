import torch

import chiquitorch
from .zoo import AllInputKinds
from .utils import make_flat_for_lifted_module


def test_flat_inputs_match_torch():
    """Lifting must match torch's flat-input builder across params,
    buffers, constants and positional + keyword inputs."""
    model = AllInputKinds().eval()
    args = (torch.randn(2, 4),)
    kwargs = {"gain": torch.tensor(3.0)}
    ep = chiquitorch.compile(model, args=args, kwargs=kwargs)

    mine = make_flat_for_lifted_module(ep, args, kwargs)
    oracle = ep._graph_module_flat_inputs(args, kwargs)

    assert len(mine) == len(oracle)
    assert all(torch.equal(a, b) for a, b in zip(mine, oracle))
