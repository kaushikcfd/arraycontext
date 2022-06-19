"""
.. currentmodule:: arraycontext
.. autoclass:: PyCUDAArrayContext
"""
__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from functools import partial, reduce
import operator
import numpy as np

from arraycontext.fake_numpy import \
        BaseFakeNumpyNamespace, BaseFakeNumpyLinalgNamespace
from arraycontext.container import NotAnArrayContainerError, serialize_container
from arraycontext.container.traversal import (
        rec_multimap_array_container, rec_map_array_container,
        rec_map_reduce_array_container,
        )

import pycuda 
import pycuda.cumath as cumath

try:
    import pycuda.gpuarray as gpuarray
except ImportError:
    pass


# {{{ fake numpy

class PyCUDAFakeNumpyNamespace(BaseFakeNumpyNamespace):

    def _get_fake_numpy_linalg_namespace(self):
        return _PyCUDAFakeNumpyLinalgNamespace(self._array_context)

    def __getattr__(self, name):
        c_to_numpy_arc_functions = {
            "atan": "arctan",
            "atan2": "arctan2",
            }

        pycuda_funcs = ["fabs", "ceil", "floor", "exp", "log", "log10", "sqrt",
                "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh"]
        if name in pycuda_funcs:
            from functools import partial
            return partial(rec_map_array_container, getattr(cumath, name))
        
        return super().__getattr__(c_to_numpy_arc_functions.get(name, name))

    # {{{ comparisons

    # FIXME: This should be documentation, not a comment.
    # These are here mainly because some arrays may choose to interpret
    # equality comparison as a binary predicate of structural identity,
    # i.e. more like "are you two equal", and not like numpy semantics.
    # These operations provide access to numpy-style comparisons in that
    # case.
            
    def array_equal(self, x, y):
        actx = self._array_context
        false = actx.from_numpy(np.int8(False))
        def rec_equal(x, y):
            if type(x) != type(y):
                return false
            
            try:
                iterable = zip(serialize_container(x), serialize_container(y))
            except NotAnArrayContainerError:
                if x.shape != y.shape:
                    return false
                else:
                    return (x == y).get().all()
            else:
                return reduce(
                    partial(gpuarray.minimum),
                    [rec_equal(ix, iy) for (_, ix),(_, iy) in iterable]
                )

        return rec_equal(x, y)

    def equal(self, x, y):
        return rec_multimap_array_container(operator.eq, x, y)

    def not_equal(self, x, y):
        return rec_multimap_array_container(operator.ne, x, y)

    def greater(self, x, y):
        return rec_multimap_array_container(operator.gt, x, y)

    def greater_equal(self, x, y):
        return rec_multimap_array_container(operator.ge, x, y)

    def less(self, x, y):
        return rec_multimap_array_container(operator.lt, x, y)

    def less_equal(self, x, y):
        return rec_multimap_array_container(operator.le, x, y)

    def maximum(self, x, y):
        return rec_multimap_array_container(gpuarray.maximum,x, y)

    def minimum(self, x, y):
        return rec_multimap_array_container(gpuarray.minimum,x, y)

    def where(self, criterion, then, else_):
        def where_inner(inner_crit, inner_then, inner_else):
            if isinstance(inner_crit, bool):
                return inner_then if inner_crit else inner_else
            return gpuarray.if_positive(inner_crit != 0, inner_then, inner_else)

        return rec_multimap_array_container(where_inner, criterion, then, else_)

    def sum(self, a, dtype=None):
        def _gpuarray_sum(ary):
            if dtype not in [ary.dtype, None]:
                raise NotImplementedError

            return gpuarray.sum(ary)

        return rec_map_reduce_array_container(sum, _gpuarray_sum, a)

    def min(self, a):
        return rec_map_reduce_array_container(
                partial(reduce, partial(gpuarray.minimum)),partial(gpuarray.min),a)

    def max(self, a):
        return rec_map_reduce_array_container(
                partial(reduce, partial(gpuarray.maximum)), partial(gpuarray.max), a)

    def stack(self, arrays, axis=0):
         return rec_multimap_array_container(
                lambda *args: gpuarray.stack(arrays=args, axis=axis),
                *arrays)

    def ones_like(self, ary):
        return self.full_like(ary, 1)

    def full_like(self, ary, fill_value):
        def _full_like(subary):
            ones = self._array_context.empty_like(subary)
            ones.fill(fill_value)
            return ones

        return self._new_like(ary, _full_like)

    def reshape(self, a, newshape, order="C"):
        return rec_map_array_container(
            lambda ary : gpuarray.reshape(ary, newshape, order=order),
            a)

    def concatenate(self, arrays, axis=0):
        return  rec_multimap_array_container( 
                    lambda *args: gpuarray.concatenate(arrays=args, axis=axis,
                    allocator=self._array_context.allocator),
                    *arrays)

    def ravel(self, a, order="C"):
        def _rec_ravel(a):
            if order in "FC":
                return gpuarray.reshape(a, -1, order=order)
            elif order == "A":
                if a.flags.f_contiguous:
                    return gpuarray.reshape(a, -1, order="F")
                elif a.flags.c_contiguous:
                    return gpuarray.reshape(a, -1, order="C")
                else:
                    raise ValueError("For `order='A'`, array should be either"
                                     " F-contiguous or C-contiguous.")
            elif order == "K":
                raise NotImplementedError("PyCUDAArrayContext.np.ravel not "
                                          "implemented for 'order=K'")
            else:
                raise ValueError("`order` can be one of 'F', 'C', 'A' or 'K'. "
                                 f"(got {order})")
        return rec_map_array_container(_rec_ravel, a)


# }}}


# {{{ fake np.linalg

class _PyCUDAFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    pass

# }}}


# vim: foldmethod=marker
