
from ...backend import TensorLike as _DT
from ...sparse import CSRTensor
from ...fem import BilinearForm, LinearForm
from ..base import Model


class BformToCSR(Model[BilinearForm, CSRTensor]):
    def run(self, form: BilinearForm):
        return form.assembly(format='csr')


class LformToArray(Model[LinearForm, _DT]):
    def run(self, form: LinearForm):
        return form.assembly(format='dense')
