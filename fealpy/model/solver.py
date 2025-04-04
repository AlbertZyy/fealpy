
from ..backend import TensorLike as _DT
from ..solver import cg
from ._types import SparseLinSys, DenseLinSys
from .base import Model


class CGSolver(Model[SparseLinSys, _DT]):
    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs

    def run(self, ls: SparseLinSys, /) -> _DT:
        A, b = ls
        return cg(A, b, **self._kwargs)
