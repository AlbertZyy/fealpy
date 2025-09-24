from fealpy.backend import bm
from fealpy.mesh import TriangleMesh


from fealpy.mesh.neural_mesh import NeuralMesh2D
from fealpy.functionspace.barron_space import BarronSpace


def source(p):
    x, y = p[..., 0], p[..., 1]
    return 2*bm.pi*bm.cos(bm.pi*x)*bm.cos(bm.pi*y)


GD = 2
weight = bm.random.randn(512, GD)
bias = bm.random.randn(512)
nmesh = NeuralMesh2D(weight, bias)
bspace = BarronSpace(nmesh, p=1)

tri_mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=20, ny=20)
qf = tri_mesh.quadrature_formula(q=3)
bcs, ws = qf.get_quadrature_points_and_weights()
pts = tri_mesh.bc_to_point(bcs)
m = tri_mesh.entity_measure('cell')

nbcs = nmesh.point_to_bc(pts) # (NC, NQ, DOF)
phi = bspace.basis(nbcs)
gphi = bspace.grad_basis(nbcs, variable="u") # (DOF,)
R = nmesh.grad_lambda()

print(gphi.shape, R.shape)

M = bm.einsum("c, q, cqf, cqg -> fg", m, ws, gphi, gphi)
A = bm.einsum("fd, fg, gd -> fg", R, M, R)

src_val = source(pts) # (NC, NQ)
F = bm.einsum("c, q, cqf, cq -> f", m, ws, phi, src_val)

print(A)
print(F)