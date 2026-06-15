import numpy as np

from torch.export.graph_signature import InputKind
from torch.utils import _pytree as pytree


def assert_flat_allclose(
    got: list[np.ndarray],
    expected: list[np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert two flat leaf lists match element-wise.

    Compares lengths, per leaf shape, dtype and ``np.allclose``.
    Prevents ``np.allclose`` broadcasting and erros due to ragged
    leaves.

    Args:
        got: Flat list of arrays under test.
        expected: Flat list of reference arrays (the ``rtol`` baseline).
        rtol: Relative tolerance for ``np.allclose``.
        atol: Absolute tolerance for ``np.allclose``.
    """
    assert len(got) == len(expected), f"leaf count: {len(got)} != {len(expected)}"
    for i, (a, b) in enumerate(zip(got, expected)):
        assert a.shape == b.shape, f"leaf {i} shape: {a.shape} != {b.shape}"
        assert a.dtype == b.dtype, f"leaf {i} dtype: {a.dtype} != {b.dtype}"
        assert np.allclose(a, b, rtol=rtol, atol=atol), f"leaf {i} values differ"


# equivalent to ep._graph_module_flat_inputs
def make_flat_for_lifted_module(ep, args, kwargs=None):
    flat_user, _ = pytree.tree_flatten((args, kwargs or {}))
    user_it = iter(flat_user)
    flat = []
    for spec in ep.graph_signature.input_specs:
        if spec.kind == InputKind.PARAMETER:
            # learnable params (weights, biases, etc.)
            flat.append(ep.state_dict[spec.target])
        elif spec.kind == InputKind.BUFFER:
            # registered module state, but not trained
            src = ep.state_dict if spec.persistent else ep.constants
            flat.append(src[spec.target])
        elif spec.kind == InputKind.CONSTANT_TENSOR:
            # literal values in the fwd pass
            flat.append(ep.constants[spec.target])
        elif spec.kind == InputKind.USER_INPUT:
            flat.append(next(user_it))
        else:
            raise RuntimeError(f"Unhandled input kind: {spec.kind}")
    return flat
