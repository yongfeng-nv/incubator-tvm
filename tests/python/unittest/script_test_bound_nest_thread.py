import tvm
from tvm import te
import numpy as np
import time



# m = te.var('m')

m = 64

A = te.placeholder((m,), name='A')
A1 = te.compute((m,), lambda i: A[i], name='A1')
A2 = te.compute((m,), lambda i: A1[i] + 0.2, name='A2')
A3 = te.compute((m,), lambda i: A2[i] + 0.3, name='A3')
s = te.create_schedule(A3.op)
s[A2].set_scope("shared")
s[A1].set_scope("local")
block_x = te.thread_axis("blockIdx.x")
thread_x = te.thread_axis("threadIdx.x")
bx, tx = s[A3].split(A3.op.axis[0], factor=32)
s[A3].bind(bx, block_x)
s[A3].bind(tx, thread_x)
s[A2].compute_at(s[A3], tx)
_, xi = s[A2].split(A2.op.axis[0], nparts=1)
# s[A2].bind(xi, thread_x)
s[A1].compute_at(s[A3], tx)

from tvm.contrib import tedd
tedd.viz_dataflow_graph(s, False, '/tmp/dfg.dot')
tedd.viz_schedule_tree(s, False, '/tmp/scheduletree.dot')
tedd.viz_itervar_relationship_graph(s, False, '/tmp/itervar.dot')
ctx = tvm.context("cuda", 0)
func = tvm.build(s, [A, A3], target="cuda", name='tid')
assert func
print(func.imported_modules[0].get_source())
    
# Random generated tensor for testing
dtype = "float32"
a = tvm.nd.array(np.random.rand(A.shape[0].value, ).astype(dtype), ctx)
b = tvm.nd.array(np.random.rand(A3.shape[0].value, ).astype(dtype), ctx)

func(a, b)
result = b.asnumpy()
answer = a.asnumpy() + 0.2 + 0.3
tvm.testing.assert_allclose(result, answer, rtol=1e-5)
print(result)
print("\n")
print(answer)

