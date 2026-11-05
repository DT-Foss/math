from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ColdRowFetchModel:
    rows:int
    width:int
    bytes_per_value:int=8

    @property
    def bytes(self)->int:
        return self.rows*self.width*self.bytes_per_value

    def transfer_ms(self,bandwidth_GBps:float)->float:
        return self.bytes/(bandwidth_GBps*1e9)*1e3

@dataclass(frozen=True)
class ColdBackedDeltaPlan:
    fetch:ColdRowFetchModel
    delta_compute_ms:float
    full_baseline_ms:float

    def total_delta_ms(self,bandwidth_GBps:float)->float:
        return self.fetch.transfer_ms(bandwidth_GBps)+self.delta_compute_ms

    def speedup(self,bandwidth_GBps:float)->float:
        return self.full_baseline_ms/self.total_delta_ms(bandwidth_GBps)
