from typing import (
    Tuple, Literal, Union, Callable,
    Protocol
)

from ..backend import TensorLike as _DT
from ..sparse import SparseTensor


SparseLinSys = Tuple[SparseTensor, _DT]
DenseLinSys = Tuple[_DT, _DT]
LinSys = Union[SparseLinSys, DenseLinSys]
btype = Literal["dirichlet", "neumann", "robin", "mixed", "d", "n", "r", "m"]
TensorOrFunc = Union[_DT, Callable[[_DT], _DT]]


class PDEData(Protocol):
    solution        : TensorOrFunc
    gradient        : TensorOrFunc
    source          : TensorOrFunc
    dirichlet       : TensorOrFunc
    is_dirichlet_bc : TensorOrFunc
    neumann         : TensorOrFunc
    is_neumann_bc   : TensorOrFunc
    robin           : TensorOrFunc
    is_robin_bc     : TensorOrFunc

class SupportsIntegral(Protocol):
    def integral(self, f: TensorOrFunc, *, q=3, **kwargs) -> _DT: ...

class SupportsError(Protocol):
    def error(self, u:TensorOrFunc, v:TensorOrFunc, *, q=3, power=2, **kwargs) -> _DT: ...

LinSysAndData = Tuple[SparseTensor, _DT, PDEData]