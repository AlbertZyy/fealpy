from typing import (
    Tuple, Dict, Any, Callable,
    Generic, TypeVar, overload
)

from ..backend import TensorLike as _DT


_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_T3 = TypeVar("_T3")
_T4 = TypeVar("_T4")
Solver = Callable[[_DT, _DT], Dict[str, Any]]


class ModelMeta(type):
    pass


class Model(Generic[_T1, _T2], metaclass=ModelMeta):
    r"""Initialize a numerical model."""
    def run(self, x: _T1) -> _T2:
        raise NotImplementedError

    def __rshift__(self, other: "Model[_T2, _T3]") -> "Sequential[_T1, _T3]":
        if isinstance(other, Sequential):
            return Sequential(self, *other.models)
        return Sequential(self, other)

    def __rrshift__(self, other: _T1) -> _T2:
        return self.run(other)

    def __and__(self, other: "Model"):
        if isinstance(other, Parallel):
            return Parallel(self, *other.models)
        return Parallel(self, other)


class Const(Model[Any, _T1]):
    def __init__(self, value: _T1, /):
        super().__init__()
        self._value = value

    def run(self, x: Any) -> _T1:
        return self._value


class Identity(Model[_T1, _T1]):
    def run(self, x: _T1) -> _T1:
        return x


class FuncModel(Model[_T1, _T2]):
    def __init__(self, func: Callable[[_T1], _T2]):
        super().__init__()
        self._func = func

    def run(self, x: _T1) -> _T2:
        return self._func(x)


class Container(Model[_T1, _T2]):
    models : Tuple[Model[Any, Any], ...]

    def __len__(self) -> int:
        return len(self.models)

    def __getitem__(self, index: int) -> Model[Any, Any]:
        return self.models[index]


class Sequential(Container[_T1, _T2]):
    @overload
    def __init__(self, arg1: Model[_T1, _T2], /): ...
    @overload
    def __init__(self, arg1: Model[_T1, _T3], arg2: Model[_T3, _T2], /): ...
    @overload
    def __init__(self, arg1: Model[_T1, _T3], arg2: Model[_T3, _T4], arg3: Model[_T4, _T2], /): ...
    def __init__(self, *args: Model[Any, Any]):
        self.models = tuple(args)

    def run(self, x: _T1) -> _T2:
        for model in self.models:
            x = model.run(x)
        return x

    def __rshift__(self, other: Model[_T2, _T3]) -> "Sequential[_T1, _T3]":
        if isinstance(other, Sequential):
            return Sequential(*self.models, *other.models)
        return Sequential(*self.models, other)


class Parallel(Container[_T1, _T2]):
    def __init__(self, *args: Model[_T1, Any]):
        self.models = tuple(args)

    def __and__(self, other: "Model"):
        if isinstance(other, Parallel):
            return Parallel(*self.models, *other.models)
        return Parallel(*self.models, other)

    def run(self, x: _T1) -> _T2:
        return tuple(model.run(x) for model in self.models)
