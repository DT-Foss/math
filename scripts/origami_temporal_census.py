#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from wirkung_math.temporal_census import TemporalDeltaCensus,effective_rank


def parse_args():
    p=argparse.ArgumentParser(description="Measure exact token-to-token delta geometry in Origami/Beast traces")
    p.add_argument("trace",help="NPZ: each activation array is [tokens,width], preferably named layer/name/x")
    p.add_argument("--basis",help="optional NPZ of layer bases [width,rank]")
    p.add_argument("--out",default="origami_temporal_census.json")
    p.add_argument("--block-size",type=int,default=32)
    p.add_argument("--atol",type=float,default=0.0)
    return p.parse_args()


def activation_keys(trace):
    keys=[]
    for key in trace.files:
        arr=trace[key]
        if arr.ndim==2 and not key.endswith("/codes") and not key.endswith("/basis"):
            keys.append(key)
    return keys


def main():
    args=parse_args()
    trace=np.load(args.trace,mmap_mode="r",allow_pickle=False)
    bases=np.load(args.basis,mmap_mode="r",allow_pickle=False) if args.basis else None
    census=TemporalDeltaCensus()
    ranks={}

    for key in activation_keys(trace):
        x=trace[key]
        if len(x)<2:
            continue
        code_key=f"{key}/codes"
        codes=trace[code_key] if code_key in trace.files else None
        basis=None
        if bases is not None:
            candidates=(key,f"{key}/basis",key.rsplit("/",1)[0] if "/" in key else key)
            for candidate in candidates:
                if candidate in bases.files:
                    basis=bases[candidate];break

        deltas=[]
        for t in range(1,len(x)):
            kwargs={"atol":args.atol,"block_size":args.block_size,"basis":basis}
            if codes is not None:
                kwargs.update(previous_codes=codes[t-1],current_codes=codes[t])
            census.add(key,x[t-1],x[t],**kwargs)
            deltas.append(np.asarray(x[t]-x[t-1],dtype=np.float64))
        D=np.stack(deltas)
        ranks[key]={
            "tokens":int(len(x)),
            "delta_rank_99":effective_rank(D,.99),
            "delta_rank_999":effective_rank(D,.999),
            "delta_matrix_shape":list(D.shape),
        }

    result={
        "schema":"origami-temporal-delta-census/v1",
        "trace":str(Path(args.trace)),
        "layers":census.summary(),
        "cross_token_rank":ranks,
    }
    Path(args.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"out":args.out,"layers":len(ranks)},sort_keys=True))


if __name__=="__main__":
    main()
