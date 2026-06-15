import torch
import torch.nn as nn
import numpy as np
from torch.utils import _pytree as pytree

from abc import ABC, abstractmethod

import chiquitorch
from .utils import make_flat_for_lifted_module


def resolve_runner(backend_name):
    return {
        "eager_pytorch": EagerPyTorchRunner,
        "fx_graph_module": FxGraphModuleRunner,
    }[backend_name]


def np_args_kwargs_to_torch(args, kwargs):
    args = tuple([torch.tensor(arg) for arg in args])
    kwargs = {k: torch.tensor(v) for k, v in kwargs.items()}
    return args, kwargs


class BaseRunner(ABC):
    """Interface for executing a model under one backend.

    A runner adapts a backend (eager, exported graph module, ...) to a common
    numeric contract so outputs can be compared across backends. Inputs and
    outputs are always FP32 ``np.ndarray``; tensor conversion is internal.

    I/O contract:
        Inputs are positional ``args`` (a tuple) and ``kwargs`` (a dict).
        Outputs are a flat ``list`` of arrays: the model output is flattened
        with ``pytree.tree_leaves``, collapsing nested tuples/lists/dicts to
        their leaf tensors. Dicts flatten in key *insertion* order (not
        sorted); a tensor is a single leaf (never iterated), so a single-tensor
        output yields a one-element list.

    Lifecycle:
        ``setup`` (once, from example inputs) -> ``run`` (repeatable) ->
        ``teardown``. ``__init__`` calls ``setup``; ``__call__`` delegates to
        ``run``.
    """

    @abstractmethod
    def setup(
        self,
        target: nn.Module,
        args: tuple[np.ndarray, ...],
        kwargs: dict[str, np.ndarray],
    ) -> None:
        """Build backend state from ``target`` and the example inputs."""
        ...

    @abstractmethod
    def run(
        self,
        args: tuple[np.ndarray, ...],
        kwargs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        """Execute the model and return its outputs as flat FP32 arrays.

        Args:
            args: Positional inputs as FP32 arrays.
            kwargs: Keyword inputs as FP32 arrays.

        Returns:
            Outputs flattened to leaf arrays in insertion order (see class
            docstring).
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Release any resources acquired in ``setup``."""
        ...

    def __init__(
        self,
        target: nn.Module,
        args: tuple[np.ndarray, ...],
        kwargs: dict[str, np.ndarray] | None = None,
    ):
        self.setup(target, args, kwargs or {})

    def __call__(
        self,
        args: tuple[np.ndarray, ...],
        kwargs: dict[str, np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        return self.run(args, kwargs or {})


class EagerPyTorchRunner(BaseRunner):

    def setup(self, target, args, kwargs):
        self.target = target

    def run(self, args, kwargs):
        args, kwargs = np_args_kwargs_to_torch(args, kwargs)
        return [o.numpy() for o in pytree.tree_leaves(self.target(*args, **kwargs))]

    def teardown(self):
        pass


class FxGraphModuleRunner(BaseRunner):

    def setup(self, target, args, kwargs):
        args, kwargs = np_args_kwargs_to_torch(args, kwargs)
        ep = chiquitorch.compile(target, args=args, kwargs=kwargs)
        # we could use torch.fx.Interpreter if we wanted to intercept
        # the calls to the nodes and customize the behaviour.
        # self.target = torch.fx.Interpreter(ep.module())
        # and then self.target.run(*args_and_kwargs_vs)
        self.ep = ep
        self.target = ep.graph_module

    def run(self, args, kwargs):
        args, kwargs = np_args_kwargs_to_torch(args, kwargs)
        outs = self.target(*make_flat_for_lifted_module(self.ep, args, kwargs))
        return [o.numpy() for o in pytree.tree_leaves(outs)]

    def teardown(self):
        pass
