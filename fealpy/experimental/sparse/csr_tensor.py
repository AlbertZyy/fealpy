
from typing import Optional, Union, overload, List
from math import prod

from ..backend import TensorLike, Number, Size
from ..backend import backend_manager as bm
from .sparse_tensor import SparseTensor
from .utils import (
    check_shape_match, check_spshape_match
)
from ._spspmm import spspmm_csr
from ._spmm import spmm_csr


class CSRTensor(SparseTensor):
    def __init__(self, crow: TensorLike, col: TensorLike, values: Optional[TensorLike],
                 spshape: Optional[Size]=None) -> None:
        """Initializes CSR format sparse tensor.

        Parameters:
            crow (Tensor): _description_
            col (Tensor): _description_
            values (Tensor | None): _description_
            spshape (Size | None, optional): _description_
        """
        self._crow = crow
        self._col = col
        self._values = values

        if spshape is None:
            nrow = crow.shape[0] - 1
            ncol = bm.max(col) + 1
            self._spshape = (nrow, ncol)
        else:
            self._spshape = tuple(spshape)

        self._check(crow, col, values, spshape)

    def _check(self, crow: TensorLike, col: TensorLike, values: Optional[TensorLike], spshape: Size):
        if crow.ndim != 1:
            raise ValueError(f"crow must be a 1-D tensor, but got {crow.ndim}")
        if col.ndim != 1:
            raise ValueError(f"col must be a 1-D tensor, but got {col.ndim}")
        if len(spshape) != 2:
                raise ValueError(f"spshape must be a 2-tuple for CSR format, but got {spshape}")

        if spshape[0] != crow.shape[0] - 1:
            raise ValueError(f"crow.shape[0] - 1 must be equal to spshape[0], "
                             f"but got {crow.shape[0] - 1} and {spshape[0]}")

        if isinstance(values, TensorLike):
            if values.ndim < 1:
                raise ValueError(f"values must be at least 1-D, but got {values.ndim}")

            if values.shape[-1] != col.shape[-1]:
                raise ValueError(f"values must have the same size as col ({col.shape[-1]}) "
                                 "in the last dimension (number of non-zero elements), "
                                 f"but got {values.shape[-1]}")
        elif values is None:
            pass
        else:
            raise ValueError(f"values must be a Tensor or None, but got {type(values)}")

    def __repr__(self) -> str:
        return f"CSRTensor(crow={self._crow}, col={self._col}, "\
               + f"values={self._values}, shape={self.shape})"

    @property
    def itype(self): return self._crow.dtype

    @property
    def nnz(self): return self._col.shape[1]

    def crow(self) -> TensorLike:
        """Return the row location of non-zero elements."""
        return self._crow

    def col(self) -> TensorLike:
        """Return the column of non-zero elements."""
        return self._col

    def values(self) -> Optional[TensorLike]:
        """Return the non-zero elements"""
        return self._values

    def to_dense(self, *, fill_value: Number=1.0, **kwargs) -> TensorLike:
        """Convert the CSRTensor to a dense tensor and return as a new object.

        Parameters:
            fill_value (int | float, optional): The value to fill the dense tensor with
                when `self.values()` is None.

        Returns:
            Tensor: The dense tensor.
        """
        context = self.values_context()
        context.update(kwargs)
        dense_tensor = bm.zeros(self.shape, **context)

        for i in range(1, self._crow.shape[0]):
            start = self._crow[i - 1]
            end = self._crow[i]
            val = fill_value if (self._values is None) else self._values[..., start:end]
            dense_tensor[..., i - 1, self._col[start:end]] = val

        return dense_tensor

    @overload
    def reshape(self, shape: Size, /) -> 'CSRTensor': ...
    @overload
    def reshape(self, *shape: int) -> 'CSRTensor': ...
    def reshape(self, *shape) -> 'CSRTensor':
        pass

    def ravel(self) -> 'CSRTensor':
        pass

    def flatten(self) -> 'CSRTensor':
        pass

    def copy(self):
        return CSRTensor(bm.copy(self._crow), bm.copy(self._col),
                         bm.copy(self._values), self._spshape)

    def neg(self) -> 'CSRTensor':
        """Negation of the CSR tensor. Returns self if values is None."""
        if self._values is None:
            return self
        else:
            return CSRTensor(self._crow, self._col, -self._values, self._spshape)

    @overload
    def add(self, other: Union[Number, 'CSRTensor'], alpha: Number=1) -> 'CSRTensor': ...
    @overload
    def add(self, other: TensorLike, alpha: Number=1) -> TensorLike: ...
    def add(self, other: Union[Number, 'CSRTensor', TensorLike], alpha: Number=1) -> Union['CSRTensor', TensorLike]:
        pass

    def mul(self, other: Union[Number, 'CSRTensor', TensorLike]) -> 'CSRTensor':
        pass

    def div(self, other: Union[Number, TensorLike]) -> 'CSRTensor':
        pass

    def pow(self, other: Union[TensorLike, Number]) -> 'CSRTensor':
        pass

    @overload
    def matmul(self, other: 'CSRTensor') -> 'CSRTensor': ...
    @overload
    def matmul(self, other: TensorLike) -> TensorLike: ...
    def matmul(self, other: Union['CSRTensor', TensorLike]):
        pass
