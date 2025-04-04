
from ...fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    ScalarNeumannBCIntegrator
)
from .._types import SparseLinSys, PDEData
from ..base import Model, Identity, Parallel, FuncModel
from .common import BformToCSR, LformToArray
from .dirichlet_bc import DirichletBCModel


class FEMPoisson():
    def __init__(self, space, bdry: str):
        super().__init__()
        self.space = space
        self.bdry = bdry
        self.refresh()

    def refresh(self):
        self.DI = ScalarDiffusionIntegrator(method='fast')
        self.SI = ScalarSourceIntegrator()
        self.bform = BilinearForm(self.space) << self.DI
        self.lform = LinearForm(self.space) << self.SI

        if self.bdry in ("neumann", "n", "mixed", "m"):
            self.NI = ScalarNeumannBCIntegrator(1.)
            self.lform.add_integrator(self.NI)

    def update_bform(self, data: PDEData):
        return self.bform

    def update_lform(self, data: PDEData):
        self.SI.source = data.source
        self.SI.clear()
        return self.lform

    @property
    def Bform(self):
        return FuncModel(self.update_bform)

    @property
    def Lform(self):
        return FuncModel(self.update_lform)

    @property
    def Matrix(self):
        return self.Bform >> BformToCSR()

    @property
    def Vector(self):
        return self.Lform >> LformToArray()

    def LinearSystem(self, dbc=True) -> Parallel[PDEData, SparseLinSys]:
        if dbc and self.bdry in ("dirichlet", "d", "mixed", "m"):
            mixed = self.bdry in ("mixed", "m")
            return Parallel(self.Matrix, self.Vector, Identity()) \
                >> DirichletBCModel(self.space, mixed=mixed)

        else:
            return Parallel(self.Matrix, self.Vector)