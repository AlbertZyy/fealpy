
from typing import Optional

import torch

_Size = torch.Size
Tensor = torch.Tensor


def _dense_shape(values: Optional[Tensor]):
    if values is None:
        return _Size()
    else:
        return values.shape[:-1]


def _dense_ndim(values: Optional[Tensor]):
    if values is None:
        return 0
    else:
        return values.ndim - 1


def shape_to_strides(shape: _Size, item_size: int):
    strides = [item_size, ]

    for i in range(1, len(shape)):
        strides.append(strides[-1] * shape[-i])

    return tuple(reversed(strides))


def _flatten_indices(indices: Tensor, shape: _Size) -> Tensor:
    nnz = indices.shape[-1]
    strides = shape_to_strides(shape, 1)
    flatten = torch.zeros((nnz,), dtype=indices.dtype, device=indices.device)

    for d, s in enumerate(strides):
        flatten += indices[d, :] * s

    return flatten.unsqueeze_(0)


def check_shape_match(shape1: _Size, shape2: _Size):
    if shape1 != shape2:
        raise ValueError(f"shape mismatch: {shape1} != {shape2}")


def check_spshape_match(spshape1: _Size, spshape2: _Size):
    if spshape1 != spshape2:
        raise ValueError(f"sparse shape mismatch: {spshape1} != {spshape2}")
