from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .consumer_ir import ConsumerProgram
from .linear_quotient import LinearConsumerQuotient


@dataclass(frozen=True)
class LinearNodeQuotient:
    node:str
    state_dimension:int
    required_dimension:int
    basis:np.ndarray
    reduced_consumer:np.ndarray
    reconstruction_residual:float
    consumer_labels:tuple[str,...]


def _topological_order(program:ConsumerProgram)->tuple[str,...]:
    nodes=program.node_map
    original=[n.name for n in program.nodes]
    children={name:[] for name in original}
    indegree={name:0 for name in original}
    for node in program.nodes:
        for dep in node.inputs:
            children[dep].append(node.name)
            indegree[node.name]+=1
    queue=[name for name in original if indegree[name]==0]
    out=[]
    while queue:
        name=queue.pop(0)
        out.append(name)
        for child in children[name]:
            indegree[child]-=1
            if indegree[child]==0:
                queue.append(child)
    if len(out)!=len(original):
        raise ValueError("consumer program contains a cycle")
    return tuple(out)


def _dimensions_and_matrices(program:ConsumerProgram,order:tuple[str,...]):
    nodes=program.node_map
    dims={}
    matrices={}
    for name in order:
        node=nodes[name]
        if node.family=="source":
            if "dimension" not in node.metadata:
                raise ValueError(f"source {name!r} needs dimension metadata")
            dim=int(node.metadata["dimension"])
            if dim<0:
                raise ValueError("dimension must be nonnegative")
            dims[name]=dim
            continue
        if node.family!="linear":
            raise ValueError(
                f"linear quotient discovery does not support family {node.family!r}"
            )
        if len(node.inputs)!=1:
            raise ValueError("linear quotient nodes require exactly one input")
        if "matrix" not in node.metadata:
            raise ValueError(f"linear node {name!r} needs matrix metadata")
        M=np.asarray(node.metadata["matrix"],dtype=np.float64)
        if M.ndim!=2:
            raise ValueError("linear matrix must be two-dimensional")
        dep=node.inputs[0]
        if M.shape[1]!=dims[dep]:
            raise ValueError(f"matrix/input dimension mismatch at {name!r}")
        matrices[name]=M
        dims[name]=int(M.shape[0])
    return dims,matrices


def discover_linear_graph_quotients(
    program:ConsumerProgram,
)->tuple[LinearNodeQuotient,...]:
    """
    Back-propagate exact linear consumer requirements through a single-input
    linear DAG.

    Every requested output contributes an identity requirement. A linear child
    y=Mx maps a downstream requirement R_y to R_x=R_y M. Requirements from
    multiple requested branches are vertically stacked. The row-space quotient
    of that stack is the minimal linear information that must survive at x.
    """
    order=_topological_order(program)
    dims,matrices=_dimensions_and_matrices(program,order)
    node_index={name:i for i,name in enumerate(order)}
    requested=[
        name for name in order if name in program.requested
    ]
    # requirements[node] contains (requested-output order, label, matrix)
    requirements={name:[] for name in order}
    for ridx,name in enumerate(requested):
        requirements[name].append(
            (ridx,name,np.eye(dims[name],dtype=np.float64))
        )

    discovered={}
    for name in reversed(order):
        reqs=requirements[name]
        if not reqs:
            continue
        reqs=sorted(reqs,key=lambda x:(x[0],node_index.get(x[1],0)))
        R=np.concatenate([x[2] for x in reqs],axis=0)
        quotient=LinearConsumerQuotient.fit(R)
        labels=tuple(x[1] for x in reqs)
        discovered[name]=LinearNodeQuotient(
            node=name,
            state_dimension=dims[name],
            required_dimension=quotient.dimension,
            basis=quotient.basis,
            reduced_consumer=quotient.reduced_consumer,
            reconstruction_residual=quotient.reconstruction_residual,
            consumer_labels=labels,
        )
        node=program.node_map[name]
        if node.family=="linear":
            dep=node.inputs[0]
            M=matrices[name]
            for ridx,label,Rchild in reqs:
                requirements[dep].append((ridx,label,Rchild@M))

    return tuple(discovered[name] for name in order if name in discovered)


@dataclass(frozen=True)
class CompiledLinearGraphQuotient:
    program:ConsumerProgram
    quotients:tuple[LinearNodeQuotient,...]
    reduced_matrices:dict[str,np.ndarray]
    node_dimensions:dict[str,int]

    @property
    def quotient_map(self)->dict[str,LinearNodeQuotient]:
        return {q.node:q for q in self.quotients}

    def run(self,source_values:dict[str,np.ndarray])->dict[str,np.ndarray]:
        qmap=self.quotient_map
        order=_topological_order(self.program)
        states={}
        for name in order:
            if name not in qmap:
                continue
            node=self.program.node_map[name]
            q=qmap[name]
            if node.family=="source":
                if name not in source_values:
                    raise ValueError(f"missing source value {name!r}")
                x=np.asarray(source_values[name],dtype=np.float64)
                if x.shape!=(q.state_dimension,):
                    raise ValueError(f"source shape mismatch for {name!r}")
                states[name]=q.basis.T@x
            else:
                dep=node.inputs[0]
                states[name]=self.reduced_matrices[name]@states[dep]
        outputs={}
        for name in self.program.requested:
            if name not in states:
                raise ValueError(f"requested node {name!r} was not compiled")
            q=qmap[name]
            outputs[name]=q.basis@states[name]
        return outputs


def compile_linear_graph_quotient(
    program:ConsumerProgram,
)->CompiledLinearGraphQuotient:
    quotients=discover_linear_graph_quotients(program)
    qmap={q.node:q for q in quotients}
    _,matrices=_dimensions_and_matrices(
        program,_topological_order(program)
    )
    reduced={}
    for q in quotients:
        node=program.node_map[q.node]
        if node.family!="linear":
            continue
        dep=node.inputs[0]
        if dep not in qmap:
            raise ValueError(
                f"required linear node {q.node!r} depends on uncompiled {dep!r}"
            )
        Qout=q.basis
        Qin=qmap[dep].basis
        reduced[q.node]=Qout.T@matrices[q.node]@Qin
    return CompiledLinearGraphQuotient(
        program=program,
        quotients=quotients,
        reduced_matrices=reduced,
        node_dimensions={q.node:q.required_dimension for q in quotients},
    )
