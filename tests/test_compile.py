import numpy as np
import pytest

from .zoo import SimpleArithmetic, StructuredOutputs
from .runners import resolve_runner
from .utils import assert_flat_allclose

BACKENDS = [
    "fx_graph_module",
]  #  "python_dispatcher", "python_codegen"]

# TODO: Decide between this and a reference FP64 run with a FP32
# error measurement.
# TODO: Add tests for Mlp and other modules
# TODO: Look into hypothesis
# TODO: Store failed inputs and rerun them in all CI runs + seeds
rng = np.random.default_rng()


@pytest.mark.parametrize("backend", BACKENDS)
def test_compile_simple_arithmetic(backend):
    x = rng.random((1, 3, 224, 224), dtype=np.float32)
    y = rng.random((1, 3, 224, 224), dtype=np.float32)

    simple_arithmetic = SimpleArithmetic()
    ref_runner = resolve_runner("eager_pytorch")(simple_arithmetic, args=(x, y))
    test_runner = resolve_runner(backend)(simple_arithmetic, args=(x, y))

    for _ in range(100):
        x_i = rng.random((2, 3, 224, 224), dtype=np.float32)
        y_i = rng.random((2, 3, 224, 224), dtype=np.float32)
        out_ref = ref_runner(args=(x_i, y_i))
        out_test = test_runner(args=(x_i, y_i))
        assert_flat_allclose(out_test, out_ref)


@pytest.mark.parametrize("backend", BACKENDS)
def test_compile_structured_outputs(backend):
    x = rng.random((1, 3, 224, 224), dtype=np.float32)

    structured_output = StructuredOutputs()
    ref_runner = resolve_runner("eager_pytorch")(structured_output, args=(x,))
    test_runner = resolve_runner(backend)(structured_output, args=(x,))

    for _ in range(100):
        x_i = rng.random((1, 3, 224, 224), dtype=np.float32)
        out_ref = ref_runner(args=(x_i,))
        out_test = test_runner(args=(x_i,))
        assert_flat_allclose(out_test, out_ref)
