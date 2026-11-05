from __future__ import annotations
from typing import Callable, Generic, TypeVar

T=TypeVar("T")


class SegmentReductionTree(Generic[T]):
    """
    Dynamic exact reduction for any associative binary operation.

    Point replacement: O(log n) operator applications.
    Full aggregate: O(1) from the root.
    Range aggregate: O(log n).
    """
    def __init__(self, values: list[T], op: Callable[[T,T],T], identity: T):
        self.op=op
        self.identity=identity
        self.n=len(values)
        size=1
        while size < self.n:
            size*=2
        self.size=size
        self.tree=[identity for _ in range(2*size)]
        for i,v in enumerate(values):
            self.tree[size+i]=v
        for i in range(size-1,0,-1):
            self.tree[i]=op(self.tree[2*i],self.tree[2*i+1])

    @property
    def aggregate(self) -> T:
        return self.tree[1]

    def replace(self, index: int, value: T) -> None:
        if index<0 or index>=self.n:
            raise IndexError(index)
        p=self.size+index
        self.tree[p]=value
        p//=2
        while p:
            self.tree[p]=self.op(self.tree[2*p],self.tree[2*p+1])
            p//=2

    def range_reduce(self, start: int, stop: int) -> T:
        if start<0 or stop<start or stop>self.n:
            raise IndexError((start,stop))
        l=start+self.size; r=stop+self.size
        left=self.identity; right=self.identity
        while l<r:
            if l&1:
                left=self.op(left,self.tree[l]); l+=1
            if r&1:
                r-=1; right=self.op(self.tree[r],right)
            l//=2; r//=2
        return self.op(left,right)
