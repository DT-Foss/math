# Source experiment map

The external CPU-box series and this repository historically reused some E-numbers for different experiments. This file prevents ID collision.

The canonical repository references are **algebra IDs Axxx + source filename**, not a bare E-number.

| CPU-box source ID / description | Canonical repository location |
|---|---|
| E168–E171 normalized / dirty / low-rank exploratory series | `results/imported_e168_e171.json` |
| CPU-box global low-rank E175 / dimension E176 | `results/imported_cpu_box_extended_2026-11-05.json`, A001/A002 |
| CPU-box lazy chain E178/E179/E181/E182 | same result file, A002 + selective materialization |
| CPU-box dirty thread/tile E183/E185/E186 | dirty compiler, A007/A008 |
| CPU-box dirty→global E187/E188 | A009 + FP64 result provenance |
| CPU-box incremental moments E189 | A010 |
| CPU-box incremental Gram E190 | A010/A011 |
| CPU-box Woodbury inverse / solve E191/E192 | A012 |
| CPU-box sparse FFT E193 | A016 |
| CPU-box logdet E194 | A012 |
| CPU-box composed statistics E195 | A010–A012 shared-delta fanout |
| CPU-box RBF pairwise update / dynamic solve (subsequent run) | A013–A015 |
| ChatGPT VM dynamic normalized bilinear cross | `results/chatgpt_vm_dynamic_bilinear_2026-11-05.json`, A020 |

Do not cite a bare E-number in architecture or algebra documentation without the source-series qualifier.
