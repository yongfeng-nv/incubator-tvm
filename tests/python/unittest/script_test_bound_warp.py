import tvm
from tvm import te
import numpy as np
import time



# m = te.var('m')
# l = te.var('l')

m = 3
l = 1


A = te.placeholder((m, l), name='A')
A1 = te.compute((m, l), lambda i, j: A[i, j], name='A1')
A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')
s = te.create_schedule(A2.op)
s[A1].set_scope("warp")
xo, xi = s[A2].split(A2.op.axis[0], 32)
xi0, xi1 = s[A2].split(xi, factor=16)
tx = te.thread_axis("threadIdx.x")
s[A2].bind(xi1, tx)
s[A2].bind(xi0, te.thread_axis("threadIdx.y"))
y = s[A2].op.axis[1]
s[A1].compute_at(s[A2], y)
xo, xi = s[A1].split(s[A1].op.axis[0], factor=16)
s[A1].bind(xi, tx)

from tvm.contrib import tedd
tedd.viz_dataflow_graph(s, False, '/tmp/dfg.dot')
tedd.viz_schedule_tree(s, False, '/tmp/scheduletree.dot')
tedd.viz_itervar_relationship_graph(s, False, '/tmp/itervar.dot')
ctx = tvm.context("cuda", 0)
func = tvm.build(s, [A, A2], target="cuda", name='tid')
assert func
print(func.imported_modules[0].get_source())
    
# Random generated tensor for testing
dtype = "float32"
a = tvm.nd.array(np.random.rand(A.shape[0].value, A.shape[1].value).astype(dtype), ctx)
b = tvm.nd.array(np.random.rand(A2.shape[0].value, A2.shape[1].value).astype(dtype), ctx)
# c = tvm.nd.array(np.random.rand(C.shape[0].value, C.shape[1].value).astype(dtype), ctx)

func(a, b)
result = b.asnumpy()
answer = a.asnumpy() + 3
tvm.testing.assert_allclose(result, answer, rtol=1e-2)

