# W20.4.5 — the CPU-elimination audit report (rg-derived)

- **Date:** 2026-10-11
- **HEAD:** `b9289f8` (dev box — CPU-only, no CUDA device present)
- **Baseline tree:** tag `w20-baseline` → commit `e9ea357` (the W20.0.5 close; the pre-rewrite tree the 103-touchpoint audit measured)
- **Law:** PROMPT.md §1 — no test run, no CPU test authored, **no repo module imported or
  executed on this box**. Every number below is a literal ripgrep (`rg`) result over
  source text at HEAD. This file is the wave's static evidence artifact;
  **behavioral verification (parity, determinism, speed) is G20.P1's, on the A10G box**
  (env/W20_GPU_PROTOCOL.md) — see §7.

---

## 1. Methodology

The two audit patterns mirror `scripts/w20_static_audit.py`'s existing CHECKS entries
(W20.3.4's module-wide `absent` checks on turboquant.py; W20.2.1's `.item()` absent
check on tq_codes.py — the same regexes, generalized to every runtime module):

| id | command (per file, repo-relative) | meaning |
|----|-----------------------------------|---------|
| **NP**   | `rg -c "np\.|numpy" <file>` | the numpy surface (imports, calls, annotations, mentions) |
| **SYNC** | `rg -c "\.item\(\)|\.cpu\(\)|\.numpy\(\)|from_numpy" <file>` | device→host sync touchpoints |

- `rg -c` counts **matching lines**; rc=1 (no matches) is recorded as **0**.
- Scope: the runtime modules of `src/rag/` (13 audited per the W20.4.5 brief + 4
  in-package extras for completeness = 17 rows). `src/rag/tests/` is out of scope —
  tests are not runtime, and PROMPT.md §1 forbids test authoring on this box anyway.
- Zero confirmation: files absent from the directory-wide `rg -c` listing have zero
  matches; the four headline zeros (turboquant, evals, m1m2, tq_cache) were
  additionally re-run per file — "No matches found" (rc=1) on both patterns.
- Nothing under `src/` was imported, executed, or byte-run. `git` was used read-only
  (`git rev-parse`, `git log`). The counts are re-derivable by anyone with the two
  commands above at HEAD `b9289f8`.

---

## 2. The per-module audit table (17 rows)

| # | module (src/rag/) | NP lines | SYNC lines | verdict |
|---|-------------------|---------:|-----------:|---------|
| 1 | turboquant.py    | 0  | 0 | **CLEAN** — the 103-touchpoint baseline module reads zero on both patterns |
| 2 | tq_codes.py      | 33 | 1 | allowlisted **DISK-EDGE** — numpy exists only inside `to_arrays`/`from_arrays` |
| 3 | tq_cache.py      | 0  | 0 | CLEAN |
| 4 | install.py       | 0  | 0 | CLEAN |
| 5 | index.py         | 4  | 4 | allowlisted **FAISS-EDGE** — the 4 marked hops; both patterns hit the *same* 4 lines |
| 6 | query.py         | 0  | 0 | CLEAN |
| 7 | ingest.py        | 0  | 0 | CLEAN |
| 8 | m1m2.py          | 0  | 0 | CLEAN — the baseline `:345` `.item()` site is gone |
| 9 | evals.py         | 0  | 0 | CLEAN — the baseline ×5 stat sites are gone (REPORT-EDGE markers remain, §3.7) |
| 10 | snapshot.py     | 69 | 3 | allowlisted **DISK-EDGE** — the module *is* the npz codec |
| 11 | codebooks.py    | 90 | 0 | allowlisted **OFFLINE-ISLAND** — zero syncs; never runtime math |
| 12 | hooks.py        | 1  | 0 | **REPORT-EDGE** — one comment-only mention; no executable numpy |
| 13 | lut_export.py   | 28 | 10 | **DISK-EDGE, offline artifact codec — NOT runtime-imported** (§3.6) |
| 14 | _paths.py       | 0  | 0 | CLEAN (in-package extra) |
| 15 | __init__.py     | 0  | 0 | CLEAN (in-package extra) |
| 16 | finetune.py     | 0  | 1 | **OFFLINE-ISLAND** — train-loop `.item()` in the offline fine-tune trainer (§3.8, §6) |
| 17 | m1m2_finetune.py| 7  | 5 | DISK-EDGE gates-npz codec + OFFLINE-ISLAND train logging (offline trainer, §3.8, §6) |

The brief's expected picture holds exactly for the 13 audited runtime modules:
**zero everywhere except the four allowlisted edge modules** (tq_codes, snapshot,
codebooks, index). Rows 14–17 are in-package modules outside the brief's list,
audited anyway and disclosed — nothing omitted.

---

## 3. Hit-line enumeration + classification (nonzero modules only)

Legend: **DISK-EDGE** (the npz file boundary) · **FAISS-EDGE** (the faiss host
boundary) · **OFFLINE-ISLAND** (offline-only code, never the serving path) ·
**REPORT-EDGE** (textual mention in comment/docstring/report — not a touchpoint).

### 3.1 tq_codes.py — 33 NP lines, 1 SYNC line → all DISK-EDGE

numpy is imported **only inside the two disk-edge methods** — the deferred imports
carry the source comments "the disk edge" (:237) and "the disk edge (inverse
direction)" (:281). Full enumeration (33 lines):

- `to_arrays` (export side, 24 lines): 229 (signature `Dict[str, np.ndarray]`),
  237 (`import numpy as np`), 240 (`t.detach().to("cpu").numpy()` — the module's
  **only** SYNC line), 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
  255, 259, 261, 262, 264, 265, 267, 268, 270 (the `np.array`/`np.ascontiguousarray`
  payload build).
- `from_arrays` (import side, 9 lines): 274 (signature), 281 (`import numpy as np`),
  292, 298, 300, 306, 309, 312, 315 (the `torch.as_tensor(np.asarray(...))` rebuilds).

The single SYNC line (240) is among the 33. No numpy, no sync, anywhere else in the
module — consistent with the auditor's W20.2.1 absent-check (`\.item\(\)` → 0).

### 3.2 snapshot.py — 69 NP lines, 3 SYNC lines → all DISK-EDGE

The module docstring declares it (line 85: "This module is the CODEC AND THE DISK
EDGE: numpy exists here by design"). Full enumeration (69 lines), by block:

- docstring contract (7): 28, 48, 49, 51, 52, 62, 85
- module imports (2): 109 (`import numpy as np`), 110 (`import numpy.lib.format`)
- row byte-serializer (9): 283, 284, 285, 288, 289, 295, 296, 297, 298
- `_edge_f32` + annotations (6): 348, 353, 354, 359, 377, 403
- save path (5): 458, 472, 484, 485, 488 — the `np.savez` writes, each source-marked
  `# DISK-EDGE`
- NpzFile reader / member validators (20): 494, 501, 574, 585, 601, 634, 660, 672,
  673, 681, 687, 688, 693, 694, 701, 702, 706, 707, 715, 716
- mmap reader doc + impl (11): 731, 736, 737, 739, 743, 745, 747, 749, 759, 762, 819
- loaders (9): 867, 871, 888, 896, 897, 898, 899, 911, 917 — `np.load` reads and the
  `torch.from_numpy` inbound hops

The 3 SYNC lines (353, 896, 898) are among the 69, and each carries the source
comment `# DISK-EDGE (the hop)`. The byte serializer (283–298) is digest/pack-only
— no host-array math.

### 3.3 codebooks.py — 90 NP lines, 0 SYNC lines → all OFFLINE-ISLAND

The module carries the `OFFLINE-NUMPY-ISLAND` marker at line 52 (the exact string
the W20.2.2 auditor entry greps) and the contract "NEVER runtime math. Runtime
consumers receive torch tensors, not numpy" (:55–56). Full enumeration (90 lines):

- island docstring (4): 35, 45, 55, 56
- numpy import (1): 67
- the pure-numpy fixed-point solve (22): 103, 135, 144, 145, 146, 147, 153, 160,
  163, 165, 166, 168, 170, 172, 173, 174, 175, 177, 182, 185, 189, 190
- Codebook encode/decode, np.searchsorted (11): 226, 233, 234, 237, 238, 239, 240,
  242, 244, 252, 255
- solve validation + centroids (13): 277, 279, 285, 288, 294, 300, 317, 318, 338,
  352, 353, 359, 360
- `get_codebook` router docstring (4): 382, 383, 387, 391
- the island's npz cache save/load/verify (14): 406, 408, 409, 412, 419, 426, 427,
  431, 436, 441, 443, 445, 447, 449
- device-mirror notes + the one-time mirror (9): 464, 480, 496, 497, 502, 512, 513,
  523, 526 — 523/526 are `torch.as_tensor(np.ascontiguousarray(...))`, the documented
  single mirror into torch tensors
- DeviceCodebook wrapper (12): 539, 542, 551, 558, 561, 574, 578, 582, 602, 603,
  607, 615

**Zero SYNC hits** — the island never syncs a device tensor.

### 3.4 index.py — 4 NP lines = 4 SYNC lines (the same 4 lines) → all FAISS-EDGE

- :869 `index.add(mat.detach().to("cpu", torch.float32)` / `.contiguous().numpy())`
  — `# FAISS-EDGE (the host hop)` — the add
- :949 `train_host = (train_mat.detach().to("cpu", torch.float32)` /
  `.contiguous().numpy())` — `# FAISS-EDGE (the host hop)` — the one-time train crossing
- :1087 `q_host = (q_hat.detach().to("cpu", torch.float32)` /
  `.contiguous().numpy())` — `# FAISS-EDGE (the host hop)` — the query
- :1089 `ids = torch.from_numpy(labels.reshape(-1))` — `# FAISS-EDGE (inbound hop)`

3 outbound `.numpy()` hops + 1 inbound `from_numpy` hop = **4, exactly at the ≤4
ceiling**, every line source-marked (rg -c "FAISS-EDGE" index.py = 9 marker mentions
including the comment blocks). There is **no `import numpy` in index.py** — the 4 NP
hits are the `.numpy()`/`from_numpy` calls themselves.

### 3.5 hooks.py — 1 NP line, 0 SYNC lines → REPORT-EDGE (comment-only)

:172 `# popcount over the packed channel mask (np.packbits zero-pads)` — a comment
documenting the zero-padding semantics of the historical numpy packbits, retained as
byte-layout documentation. hooks.py has no numpy import and no executable numpy
(the NP count would exceed 1 if it did).

### 3.6 lut_export.py — 28 NP lines, 10 SYNC lines (34 distinct) → DISK-EDGE; runtime status: NOT runtime

The brief asked to audit "tq_cache.py's helper lut_export.py if it is runtime."
Determination: **it is not runtime-imported** — the only in-tree importer is
`src/rag/tests/test_finetune.py:64`; tq_cache.py's imports are torch,
transformers.cache_utils, _paths, turboquant (rg -n "^import |^from " tq_cache.py).
It is the §7/§11 fine-tuned-LUT **artifact codec** (export at fine-tune time,
import of the artifact later) — an offline export/import edge, same class as
to_arrays/from_arrays. Full enumeration:

- NP (28): 29 (docstring), 64 (`import numpy as np`), 113, 118, 120 (digest
  helpers), 214, 215, 218, 219, 225, 230, 232, 234, 235, 236, 239 (export: the
  `mod.snapped_lut().detach().cpu().numpy()` hops and the `np.savez` payload),
  262, 263, 267 (`np.load`), 273, 274, 341, 344, 348, 349, 352 (import/validation),
  362, 366 (`torch.from_numpy(...).to(...)` inbound hops)
- SYNC (10): 215, 219 (the export `.cpu().numpy()` hops), 275, 276, 278, 280, 284,
  285 (the `.item()` header/meta reads on the loaded artifact), 362, 366 (inbound
  `from_numpy` hops)
- Overlap {215, 219, 362, 366} → 34 distinct lines. All DISK-EDGE.

### 3.7 The zero-hit serving modules — confirmations

- **turboquant.py**: rc=1 on both patterns. This is the module that carried the
  baseline's **103** numpy/sync touchpoints (PROPOSAL.md §2); the same patterns now
  find nothing.
- **evals.py**: 0/0 — the former ×5 stat sites compute in torch; the report floats
  go through plain `float(...)` at the REPORT-EDGE-marked comment sites (:38, :99,
  :127, :158, :179, :190, :299; e.g. `float(mse_t)` :130, `float(spread_t)` :180).
  `float()` is not `.item()`/`.cpu()`/`.numpy()`/`from_numpy`, so the SYNC pattern
  legitimately records 0 — the design W20.4.4 intended ("report formatting may print
  floats — formatting, not math").
- **m1m2.py**: 0/0 — the `:345` `.item()` min-position check became a tensor-space
  assertion (W20.4.4's DO-D: `rg -n "\.item\(\)" src/rag/m1m2.py` == 0 — confirmed).
- **tq_cache.py, install.py, query.py, ingest.py, _paths.py, __init__.py**: 0/0.

### 3.8 In-package offline trainers (outside the brief's list — disclosed)

- **finetune.py** — 1 SYNC line: :170 `history.append((step, float(loss.item())))`
  — train-loop history logging → **OFFLINE-ISLAND**.
- **m1m2_finetune.py** — 7 NP + 5 SYNC (10 distinct lines): :119 (docstring
  mention → REPORT-EDGE), :131 (`import numpy as np`), :166 (the
  `.detach().cpu().numpy()` gates export), :175 (`np.savez`), :185 (`np.load`),
  :189 (`z[k].item()` meta read), :191 (`np.asarray` read), :215
  (`torch.from_numpy` inbound) → **DISK-EDGE** (the trained-gates npz codec; the
  artifact is `m1m2_gates.npz`); :703, :709 (`loss.item()` train-loop logging) →
  **OFFLINE-ISLAND**.
- Both modules are imported in-tree only by tests and by
  `scripts/gpu/finetune_m1m2.py` (the GPU-box trainer) — never the serving path.

### 3.9 Classification totals (distinct matching lines)

| class | lines | derivation |
|-------|------:|------------|
| DISK-EDGE | 143 | tq_codes 33 + snapshot 69 + lut_export 34 + m1m2_finetune 7 |
| FAISS-EDGE | 4 | index 4 |
| OFFLINE-ISLAND | 93 | codebooks 90 + m1m2_finetune 2 + finetune 1 |
| REPORT-EDGE | 2 | hooks 1 + m1m2_finetune 1 (docstring) |
| **total nonzero lines** | **242** | 33+69+90+4+1+34+1+10 (per-module distinct) |

(A `.numpy()` line matches both patterns, so NP+SYNC double-counts it; the totals
above are de-duplicated to distinct lines. The per-module table keeps the raw
`rg -c` numbers as run.)

---

## 4. Before/after arithmetic (PROPOSAL.md §2, lines 75–77)

The baseline, quoted from PROPOSAL.md §2 ("The measured problem"):

> "The audit counted **103 numpy/sync touchpoints in turboquant.py** plus the
> serving-side sites (index.py:372/473/510, evals.py ×5, m1m2.py:345)."

| audited site | baseline (tag `w20-baseline`, e9ea357) | after (HEAD `b9289f8`) |
|---|---:|---|
| turboquant.py | **103** touchpoints | **0 / 0** (both patterns rc=1) |
| index.py:372/473/510 | 3 sync sites | 0 unmarked sites; **4 lines remain, all FAISS-EDGE-marked** (3 outbound `.numpy()` + 1 inbound `from_numpy`) |
| evals.py | 5 sites | **0 / 0** |
| m1m2.py:345 | 1 site | **0 / 0** |
| serving-side subtotal | 9 sites | 0 outside the allowlist; 4 allowlisted FAISS-EDGE lines |
| **audited total** | **112** (103 + 9) | **0 outside the allowlist** |

**Stated honestly:** 103 → 0 in turboquant.py; the serving-side 9 sites → 0 except
the 4 allowlisted FAISS-EDGE hops in index.py (the sanctioned faiss boundary,
PROPOSAL.md §3.6/§3.7). Metric caveat, stated equally honestly: the baseline 103
was the rev-1 audit's *touchpoint* count on the pre-rewrite tree; this report's
after-counts are `rg -c` *matching-line* counts at HEAD. The two numbers are not a
line-for-line comparable metric — but the direction is unambiguous: the exact
patterns that located the baseline sites return zero matches at HEAD.

---

## 5. The two foreign edges + the island, restated

**The npz disk edge.** numpy survives in the serving tree exactly at the snapshot
codec: `tq_codes.to_arrays/from_arrays` (the codes value type's numpy exists only
inside those two methods — deferred imports at :237/:281, "the disk edge") and
`snapshot.py` (the whole module is the codec: the npz save path, the eager and
memmap readers, and the three marked hops :353/:896/:898). Writes ride
`codes.to_arrays()` into `np.savez`; reads rebuild torch tensors via
`torch.as_tensor`/`torch.from_numpy` at the loader. Every sync in these two modules
is a marked DISK-EDGE hop at the file boundary, and the only host-array work is
digest/pack — no math on host arrays.

**The faiss boundary.** index.py crosses to host exactly 4 marked times — the add
(:869), the one-time train (:949), and the query (:1087), each a
`.detach().to("cpu", torch.float32).contiguous().numpy()` one-hop crossing, plus
the inbound `torch.from_numpy(labels.reshape(-1))` (:1089). faiss-cpu stays by
design (PROPOSAL.md §3.6: "the faiss add receives GPU-produced vectors once at its
boundary — faiss-cpu stays: foreign library, tiny index, few queries"); the vectors
are produced on GPU and everything between crossings is torch on device.

**The offline solve island.** codebooks.py is the OFFLINE-NUMPY-ISLAND (marker :52,
checked by the W20.2.2 auditor entry): the (2^b)-centroid fixed-point solve is pure
numpy + stdlib, executed once per (b, d), cached to npz (lines 406–449); runtime
consumers receive `DeviceCodebook`'s torch tensors through the one-time mirror
(:523/:526, `torch.as_tensor(np.ascontiguousarray(...))` — zero-copy on host, copy
to device), with an id-keyed cache (:496–502) so freshly-loaded numpy tables can
never alias a served mirror. Never runtime math; zero SYNC hits in the module.

---

## 6. FINDINGS (loud — nothing omitted)

- **Un-classifiable hits: NONE.** Every one of the 242 distinct nonzero lines
  classifies into DISK-EDGE / FAISS-EDGE / OFFLINE-ISLAND / REPORT-EDGE, and every
  hit in a *serving-path* module sits on a source-marked allowlist line.
- **Disclosure 1 — the offline trainers still sync their logs.** finetune.py:170
  and m1m2_finetune.py:703/:709 use `.item()` for train-loop history/prints. These
  modules are offline-only (imported in-tree by tests and the GPU-box trainer
  script, never the serving path), so they do not violate the serving contract —
  but if the law is read strictly as "no `.item()` anywhere under src/rag outside
  the four allowlisted edge modules," these 3 lines are the residue. Disposition
  for the orchestrator: an explicit offline-train allowlist, or a later offline
  sweep. **Not hidden.**
- **Disclosure 2 — lut_export.py carries 34 disk-edge lines and is NOT
  runtime-imported** (only tests/test_finetune.py imports it in-tree; tq_cache.py
  does not). If it ever becomes runtime-imported, this audit must be re-run for it.
- **Disclosure 3 — hooks.py:172 mentions `np.packbits` in a comment only.** No
  executable numpy exists in hooks.py.
- The brief's expected result is confirmed for all 13 audited runtime modules:
  zero everywhere except the four allowlisted edge modules (tq_codes 33+1,
  snapshot 69+3, codebooks 90+0, index 4+4).

---

## 7. What this report is NOT (the G20.P1 handoff)

This report is **rg-derived static evidence at HEAD `b9289f8`, 2026-10-11**. No
module was imported or executed (PROMPT.md §1); no test was run or authored. It
says nothing about parity, determinism, or speed — those are **G20.P1**'s gates on
the A10G box (env/W20_GPU_PROTOCOL.md; the gate harness of W20.4.6). The static
backstop already lives in `scripts/w20_static_audit.py` (W20.3.4 pins turboquant.py
to zero on both patterns; W20.2 pins the tq_codes `.item()` absence and the
codebooks OFFLINE-NUMPY-ISLAND marker); this wave's CHECKS entry is appended at
wave close per the auditor's own protocol. Anyone can re-derive every number above
with the two commands of §1.
