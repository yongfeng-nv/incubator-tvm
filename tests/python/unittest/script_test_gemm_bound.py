import tvm
from tvm import te
import numpy as np
import time


nn = 1024
n = tvm.runtime.convert(nn)
A = te.placeholder((n, n), name='A')
B = te.placeholder((n, n), name='B')
k = te.reduce_axis((0, n), name='k')
C = te.compute(
    (n, n),
    lambda ii, jj: te.sum(A[ii, k] * B[jj, k], axis=k),
    name='CC')
# schedule
s = te.create_schedule(C.op)
xtile, ytile = 32, 32
scale = 8
num_thread = 8
block_factor = scale * num_thread
block_x = te.thread_axis("blockIdx.x")
thread_x = te.thread_axis("threadIdx.x")
block_y = te.thread_axis("blockIdx.y")
thread_y = te.thread_axis("threadIdx.y")
CC = s.cache_write(C, "local")
AA = s.cache_read(A, "shared", [CC])
BB = s.cache_read(B, "shared", [CC])
by, yi = s[C].split(C.op.axis[0], factor=block_factor)
bx, xi = s[C].split(C.op.axis[1], factor=block_factor)
s[C].reorder(by, bx, yi, xi)
s[C].bind(by, block_y)
s[C].bind(bx, block_x)
ty, yi = s[C].split(yi, nparts=num_thread)
tx, xi = s[C].split(xi, nparts=num_thread)
s[C].reorder(ty, tx, yi, xi)
s[C].bind(ty, thread_y)
s[C].bind(tx, thread_x)
yo, xo = CC.op.axis
s[CC].reorder(k, yo, xo)
s[CC].compute_at(s[C], tx)
s[AA].compute_at(s[CC], k)
s[BB].compute_at(s[CC], k)
# s[AA].compute_at(s[C], bx)
# s[BB].compute_at(s[C], bx)
ty, xi = s[AA].split(s[AA].op.axis[0], nparts=num_thread)
tx, xi = s[AA].split(xi, nparts=num_thread)
s[AA].bind(ty, thread_y)
s[AA].bind(tx, thread_x)
ty, xi = s[BB].split(s[BB].op.axis[0], nparts=num_thread)
tx, xi = s[BB].split(xi, nparts=num_thread)
s[BB].bind(ty, thread_y)
s[BB].bind(tx, thread_x)
ctx = tvm.context("cuda", 0)
func = tvm.build(s, [A, B, C], target="cuda", name='tid')
assert func
print(func.imported_modules[0].get_source())

from tvm.contrib import tedd
tedd.viz_dataflow_graph(s, False, '/tmp/dfg.dot')
tedd.viz_schedule_tree(s, False, '/tmp/scheduletree.dot')
tedd.viz_itervar_relationship_graph(s, False, '/tmp/itervar.dot')
    
# Random generated tensor for testing
dtype = "float32"
a = tvm.nd.array(np.random.rand(A.shape[0].value, A.shape[1].value).astype(dtype), ctx)
b = tvm.nd.array(np.random.rand(B.shape[0].value, B.shape[1].value).astype(dtype), ctx)
c = tvm.nd.array(np.random.rand(C.shape[0].value, C.shape[1].value).astype(dtype), ctx)

func(a, b, c)
result = c.asnumpy()
answer = np.matmul(a.asnumpy(), b.asnumpy().transpose())
tvm.testing.assert_allclose(result, answer, rtol=1e-2)
# evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
# print(name+': %f ms' % (evaluator(a, b, d).mean * 1e3))