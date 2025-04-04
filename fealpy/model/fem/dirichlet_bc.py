
from typing import Tuple

from ...fem import DirichletBC
from .._types import LinSysAndData, SparseLinSys
from ..base import Model


class DirichletBCModel(Model[LinSysAndData, SparseLinSys]):
    def __init__(self, space, mixed: bool = False):
        super().__init__()
        self.space = space
        self.mixed = mixed

    def run(self, ls: LinSysAndData):
        A, F, pde = ls
        threshold = pde.is_dirichlet_bc if self.mixed else None
        self.dbc = DirichletBC(self.space, gd=pde.dirichlet, threshold=threshold)
        self.dbc.apply(A, F, gd=pde.dirichlet)
        return A, F
