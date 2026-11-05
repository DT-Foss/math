from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AdditiveStatsFootprint:
    rows:int
    width:int
    bytes_per_value:int=8

    @property
    def raw_bytes(self)->int:
        return self.rows*self.width*self.bytes_per_value

    @property
    def sufficient_bytes(self)->int:
        # sum + sumsq + full Gram
        return (2*self.width+self.width*self.width)*self.bytes_per_value

    @property
    def compression_ratio(self)->float:
        return self.raw_bytes/self.sufficient_bytes


@dataclass(frozen=True)
class StateQuotient:
    """
    Memory-side counterpart of sufficient-statistic compute replacement.

    Hot state contains only information required by future consumers.
    Raw data may remain in a cold tier or be discarded if no future consumer
    depends on it beyond the stored sufficient state.
    """
    raw_bytes:int
    hot_bytes:int

    @property
    def ratio(self)->float:
        return self.raw_bytes/self.hot_bytes
