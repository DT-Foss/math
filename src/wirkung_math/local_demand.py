from __future__ import annotations
from dataclasses import dataclass
from .dirty2d import Rect

def intersect_rect(a:Rect,b:Rect)->Rect|None:
    y0=max(a.y0,b.y0); x0=max(a.x0,b.x0)
    y1=min(a.y1,b.y1); x1=min(a.x1,b.x1)
    if y0>=y1 or x0>=x1:
        return None
    return Rect(y0,x0,y1,x1)

@dataclass(frozen=True)
class LayerDemandRegion:
    layer:int
    forward_dirty:Rect
    backward_demand:Rect
    required:Rect|None

def local_chain_delta_demand(
    input_edit:Rect,
    output_request:Rect,
    height:int,
    width:int,
    radii:tuple[int,...],
)->tuple[LayerDemandRegion,...]:
    """
    Exact geometry for a same-resolution local operator chain.

    At layer l:
      forward dirty = input edit expanded by prefix receptive radius;
      backward demand = requested outputs expanded by suffix receptive radius;
      required changed state = intersection of both.

    This is the 2-D specialization of Desc(edit) intersect Anc(request).
    """
    L=len(radii)
    prefix=[0]
    for r in radii:
        prefix.append(prefix[-1]+r)
    suffix=[0]*(L+1)
    for i in range(L-1,-1,-1):
        suffix[i]=suffix[i+1]+radii[i]

    out=[]
    for l in range(L+1):
        fd=input_edit.expand(prefix[l],height,width)
        bd=output_request.expand(suffix[l],height,width)
        out.append(LayerDemandRegion(l,fd,bd,intersect_rect(fd,bd)))
    return tuple(out)

def required_area_sum(plan:tuple[LayerDemandRegion,...])->int:
    return sum(0 if x.required is None else x.required.area for x in plan[1:])
