"""
.. autoclass:: TaggableCLArray
.. autoclass:: Axis

.. autofunction:: to_tagged_cl_array
"""

import pyopencl.array as cla
from typing import FrozenSet, Union, Sequence, Optional, Tuple
from pytools.tag import Taggable, Tag
from dataclasses import dataclass
from pytools import memoize


@dataclass(frozen=True, eq=True)
class Axis(Taggable):
    """
    Records the tags corresponding to a dimensions of :class:`TaggableCLArray`.
    """
    tags: FrozenSet[Tag]

    def copy(self, **kwargs):
        from dataclasses import replace
        return replace(self, **kwargs)


@memoize
def _construct_untagged_axes(ndim: int) -> Tuple[Axis, ...]:
    return tuple(Axis(frozenset()) for _ in range(ndim))


class TaggableCLArray(cla.Array, Taggable):
    """
    A :class:`pyopencl.array.Array` with additional attributes record

    .. attribute:: axes

       A :class:`tuple` of instances of :class:`Axis`, with one :class:`Axis`
       for each dimension of the array.

    .. attribute:: tags

        A :class:`frozenset` of :class:`pytools.tag.Tag`. Typically intended to
        record application-specific metadata to drive the optimizations in
        :meth:`arraycontext.PyOpenCLArrayContext.transform_loopy_program`.
    """
    def __init__(self, cq, shape, dtype, order="C", allocator=None,
                 data=None, offset=0, strides=None, events=None, _flags=None,
                 _fast=False, _size=None, _context=None, _queue=None,
                 axes=None, tags=frozenset()):

        super().__init__(cq=cq, shape=shape, dtype=dtype,
                         order=order, allocator=allocator,
                         data=data, offset=offset,
                         strides=strides, events=events,
                         _flags=_flags, _fast=_fast,
                         _size=_size, _context=_context,
                         _queue=_queue)

        self.tags = tags
        axes = axes if axes is not None else _construct_untagged_axes(len(self
                                                                          .shape))
        self.axes = axes

    def copy(self, queue=cla._copy_queue, tags=None, axes=None):
        if tags is not None or axes is not None:
            if queue is not cla._copy_queue:
                raise ValueError("Cannot change both 'tags'/'axes' and 'queue'"
                                 " at once.")
            tags = self.tags if tags is None else tags
            axes = self.axes if axes is None else axes
            return self.__class__(None, self.shape, self.dtype,
                                  allocator=self.allocator,
                                  strides=self.strides, data=self.base_data,
                                  offset=self.offset, events=self.events,
                                  _fast=True, _context=self.context,
                                  _queue=self.queue, _size=self.size,
                                  tags=tags, axes=axes)
        else:
            new_with_queue = super().copy(queue=queue)
            return self.__class__(None, new_with_queue.shape,
                                  new_with_queue.dtype,
                                  allocator=new_with_queue.allocator,
                                  strides=new_with_queue.strides,
                                  data=new_with_queue.base_data,
                                  offset=new_with_queue.offset,
                                  events=new_with_queue.events, _fast=True,
                                  _context=new_with_queue.context,
                                  _queue=new_with_queue.queue,
                                  _size=new_with_queue.size,
                                  tags=self.tags,
                                  axes=self.axes,
                                  )

    def with_tagged_axis(self, iaxis: int,
                         tags: Union[Sequence[Tag], Tag]) -> "TaggableCLArray":
        """
        Returns a copy of *self* with *iaxis*-th axis tagged with *tags*.
        """
        new_axes = (self.axes[:iaxis]
                    + (self.axes[iaxis].tagged(tags),)
                    + self.axes[iaxis+1:])
        return self.copy(axes=new_axes)


def to_tagged_cl_array(ary: cla.Array,
                       axes: Optional[Tuple[Axis, ...]],
                       tags: FrozenSet[Tag]) -> TaggableCLArray:
    """
    Returns a :class:`TaggableCLArray` that is constructed from the data in
    *ary* along with the metadata from *axes* and *tags*.

    :arg axes: An instance of :class:`Axis` for each dimension of the
        array. If passed *None*, then initialized to a :class:`pytato.Axis`
        with no tags attached for each dimension.
    """
    return TaggableCLArray(None, ary.shape,
                           ary.dtype,
                           allocator=ary.allocator,
                           strides=ary.strides,
                           data=ary.base_data,
                           offset=ary.offset,
                           events=ary.events, _fast=True,
                           _context=ary.context,
                           _queue=ary.queue, _size=ary.size,
                           axes=axes,
                           tags=tags)
