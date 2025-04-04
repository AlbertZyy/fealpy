
from typing import Any, TypeVar

from ..backend import TensorLike as _DT
from ..functionspace import Function
from .base import Model
from ._types import SupportsError, SupportsIntegral

_T1 = TypeVar("_T1")


class AsFunction(Model[_DT, Function]):
    def __init__(self, space, coordtype: str = 'barycentric'):
        super().__init__()
        self.space = space
        self.coordtype = coordtype

    def run(self, x: _DT):
        return Function(self.space, x, coordtype=self.coordtype)


class Print(Model[_T1, _T1]):
    def run(self, x: Any):
        print(x)
        return x


class Error(Model[Any, _DT]):
    def __init__(
            self,
            mesh: SupportsError,
            *,
            q: int = 3,
            power: int = 2,
            celltype: bool = False
    ):
        super().__init__()
        self.mesh = mesh
        self.q = q
        self.power = power
        self.cell_type = celltype
        self.kwargs = {'q': q, 'power': power, 'celltype': celltype}

    def run(self, uv: Any) -> _DT:
        u, v = uv
        return self.mesh.error(u, v, **self.kwargs)
