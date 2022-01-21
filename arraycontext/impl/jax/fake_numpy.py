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

from arraycontext.fake_numpy import (
        BaseFakeNumpyNamespace, BaseFakeNumpyLinalgNamespace,
        )
from arraycontext.container.traversal import (
        rec_multimap_array_container, rec_map_array_container,
        rec_map_reduce_array_container,
        )
from arraycontext.container import NotAnArrayContainerError, serialize_container
import numpy
import jax.numpy as jnp


class JAXFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    # Everything is implemented in the base class for now.
    pass


class JAXFakeNumpyNamespace(BaseFakeNumpyNamespace):
    """
    A :mod:`numpy` mimic for :class:`JAXArrayContext`.
    """
    def _get_fake_numpy_linalg_namespace(self):
        return JAXFakeNumpyLinalgNamespace(self._array_context)

    _jax_funcs = {"abs", "sin", "cos", "tan", "arcsin", "arccos",
                  "arctan", "sinh", "cosh", "tanh", "exp", "log", "log10",
                  "isnan", "sqrt", "exp", "conj"}

    def __getattr__(self, name):
        if name in self._jax_funcs:
            from functools import partial
            return partial(rec_map_array_container, getattr(jnp, name))

        raise AttributeError(f"{type(self)} object has no attribute '{name}'")

    def reshape(self, a, newshape, order="C"):
        return rec_multimap_array_container(
            lambda ary: jnp.reshape(ary, newshape, order=order),
            a)

    def transpose(self, a, axes=None):
        return rec_multimap_array_container(jnp.transpose, a, axes)

    def concatenate(self, arrays, axis=0):
        return rec_multimap_array_container(jnp.concatenate, arrays, axis)

    def ones_like(self, ary):
        return jnp.ones_like(ary)

    def maximum(self, x, y):
        return rec_multimap_array_container(jnp.maximum, x, y)

    def minimum(self, x, y):
        return rec_multimap_array_container(jnp.minimum, x, y)

    def where(self, criterion, then, else_):
        return rec_multimap_array_container(jnp.where, criterion, then, else_)

    def sum(self, a, axis=None, dtype=None):
        return rec_map_reduce_array_container(sum,
                                              partial(jnp.sum,
                                                      axis=axis,
                                                      dtype=dtype),
                                              a)

    def min(self, a, axis=None):
        return rec_map_reduce_array_container(
                partial(reduce, jnp.minimum), partial(jnp.amin, axis=axis), a)

    def max(self, a, axis=None):
        return rec_map_reduce_array_container(
                partial(reduce, jnp.maximum), partial(jnp.amax, axis=axis), a)

    def stack(self, arrays, axis=0):
        return rec_multimap_array_container(
            lambda *args: jnp.stack(arrays=args, axis=axis),
            *arrays)

    # {{{ relational operators

    def equal(self, x, y):
        return rec_multimap_array_container(jnp.equal, x, y)

    def not_equal(self, x, y):
        return rec_multimap_array_container(jnp.not_equal, x, y)

    def array_equal(self, a, b):
        actx = self._array_context

        # NOTE: not all backends support `bool` properly, so use `int8` instead
        false = actx.from_numpy(numpy.int8(False))

        def rec_equal(x, y):
            if type(x) != type(y):
                return false

            try:
                iterable = zip(serialize_container(x), serialize_container(y))
            except NotAnArrayContainerError:
                if x.shape != y.shape:
                    return false
                else:
                    return jnp.all(jnp.equal(x, y))
            else:
                return reduce(
                        jnp.logical_and,
                        [rec_equal(ix, iy) for (_, ix), (_, iy) in iterable]
                        )

        return rec_equal(a, b)

    def greater(self, x, y):
        return rec_multimap_array_container(jnp.greater, x, y)

    def greater_equal(self, x, y):
        return rec_multimap_array_container(jnp.greater_equal, x, y)

    def less(self, x, y):
        return rec_multimap_array_container(jnp.less, x, y)

    def less_equal(self, x, y):
        return rec_multimap_array_container(jnp.less_equal, x, y)

    # }}}

    def arctan2(self, y, x):
        return rec_multimap_array_container(jnp.arctan2, y, x)

    def ravel(self, a, order="C"):
        def _rec_ravel(a):
            if order in "FC":
                return jnp.ravel(a, order=order)
            elif order in "AK":
                # flattening in a C-order
                # memory layout is assumed to be "C"
                return jnp.ravel(a, order="C")
            else:
                raise ValueError("`order` can be one of 'F', 'C', 'A' or 'K'. "
                                 f"(got {order})")

        return rec_map_array_container(_rec_ravel, a)

    def vdot(self, x, y, dtype=None):
        from arraycontext import rec_multimap_reduce_array_container

        def _rec_vdot(ary1, ary2):
            if dtype not in [None, numpy.find_common_type((ary1.dtype,
                                                           ary2.dtype),
                                                          ())]:
                raise NotImplementedError("JAXArrayContext cannot take dtype in"
                                          " vdot.")

            return jnp.vdot(ary1, ary2)

        return rec_multimap_reduce_array_container(sum, _rec_vdot, x, y)

    def any(self, a):
        return rec_map_reduce_array_container(
                partial(reduce, partial(jnp.logical_or)),
                lambda subary: subary.any(),
                a)

    def all(self, a):
        return rec_map_reduce_array_container(
                partial(reduce, partial(jnp.logical_and)),
                lambda subary: subary.all(),
                a)

    def broadcast_to(self, array, shape):
        return rec_map_array_container(partial(jnp.broadcast_to, shape=shape), array)
