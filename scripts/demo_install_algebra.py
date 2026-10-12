#!/usr/bin/env python3
"""demo_install_algebra.py — a SYNTHETIC, CPU-only algebra demonstration (numpy).

NO model is loaded, NO GPU artifacts are touched. This simulates the
GatedDeltaNet recurrence (delta rule + decay, qk-l2norm) on random data
with the semantics of src/scripts/modeling.py::

    S_t = lam_t * (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
    (k l2-normalized, lam in (0,1), beta = sigmoid(b))

and measures, for the W16 absolute-protocol install
(install.py::sum_absolute_codes):

    S_install = sum_i abs_i - (n-1) * S_sys

vs the TRUE sequential state after [sys; c1; c2; c3]. Chunks are captured
INDEPENDENTLY from the system reset point (ingest.py::ingest_chunk
reseed semantics). This isolates the ALGEBRA (not the model) of the
n>=2 install.
"""
import numpy as np

rng = np.random.default_rng(0)

D = 128          # head dim (k and v)
N_SYS = 24       # system prompt tokens
N_CHUNK = 256    # tokens per chunk
LAM = 0.995      # per-token decay (mid-range; gdn decays are head/learned)
TRIALS = 5


def l2n(x):
    return x / max(np.linalg.norm(x), 1e-9)


def run_seq(S0, k, v, b):
    """One delta-rule trajectory over (T, D) k/v; returns end state."""
    S = S0.copy()
    for t in range(k.shape[0]):
        kt = l2n(k[t])                       # use_qk_l2norm_in_kernel=True
        beta = 1.0 / (1.0 + np.exp(-b[t]))   # sigmoid
        # S <- lam * (I - beta k k^T) S + beta k v^T
        S = LAM * (S - beta * np.outer(kt, kt @ S)) + beta * np.outer(kt, v[t])
    return S


def cos(a, b):
    return float(np.dot(a.ravel(), b.ravel())
                 / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


results = []
for trial in range(TRIALS):
    k_all = rng.standard_normal((N_SYS + 3 * N_CHUNK, D))
    v_all = 0.3 * rng.standard_normal((N_SYS + 3 * N_CHUNK, D))
    b_all = rng.standard_normal(N_SYS + 3 * N_CHUNK)

    sys_k, sys_v, sys_b = k_all[:N_SYS], v_all[:N_SYS], b_all[:N_SYS]
    chunks = []
    for c in range(3):
        s = N_SYS + c * N_CHUNK
        chunks.append((k_all[s:s + N_CHUNK], v_all[s:s + N_CHUNK],
                       b_all[s:s + N_CHUNK]))

    S_sys = run_seq(np.zeros((D, D)), sys_k, sys_v, sys_b)

    # per-chunk INDEPENDENT capture from the system reset point (ingest)
    abs_states = [run_seq(S_sys, ck, cv, cb) for ck, cv, cb in chunks]

    # the TRUE sequential trajectory [sys; c1..cn] for each n
    S_true_n = {}
    acc = S_sys
    for n, (ck, cv, cb) in enumerate(chunks, start=1):
        acc = run_seq(acc, ck, cv, cb)
        S_true_n[n] = acc

    row = {}
    for n in (1, 2, 3):
        S_install = sum(abs_states[:n]) - (n - 1) * S_sys
        S_true = S_true_n[n]      # truth = reading the SAME n chunks in order
        rel = float(((S_install - S_true) ** 2).sum()
                    / ((S_true ** 2).sum() + 1e-30))
        sys_frac = float(((n - 1) * np.linalg.norm(S_sys)) ** 2
                         / (np.linalg.norm(S_install) ** 2 + 1e-30))
        row[n] = (rel, cos(S_install, S_true), sys_frac)
    row["sys_survival_cos_after_1_chunk"] = cos(abs_states[0], S_sys)
    row["norm_S_sys"] = float(np.linalg.norm(S_sys))
    row["norm_abs1"] = float(np.linalg.norm(abs_states[0]))
    row["norm_S_true"] = float(np.linalg.norm(S_true_n[3]))
    row["norm_install3"] = float(np.linalg.norm(sum(abs_states) - 2 * S_sys))
    results.append(row)

print("GDN delta-rule recurrence — install-vs-truth (SYNTHETIC, algebra only)")
print(f"config: D={D}, N_sys={N_SYS}, N_chunk={N_CHUNK}, lam={LAM}")
print()
print(("{:>3} {:>26} {:>22} {:>32}").format(
    "n", "rel-MSE(install vs true)", "cos(install, true)",
    "spurious -(n-1)*sys energy frac"))
for n in (1, 2, 3):
    rel = sum(r[n][0] for r in results) / TRIALS
    c = sum(r[n][1] for r in results) / TRIALS
    frac = sum(r[n][2] for r in results) / TRIALS
    print(f"{n:>3} {rel:>26.4f} {c:>22.4f} {frac:>32.4f}")
print()
sv = sum(r["sys_survival_cos_after_1_chunk"] for r in results) / TRIALS
print(f"cos(abs_1, S_sys) after one chunk = {sv:.4f} "
      f"(how much of the system state survives a chunk's decay)")
print()
for k in ("norm_S_sys", "norm_abs1", "norm_S_true", "norm_install3"):
    print(f"{k:>18} = {sum(r[k] for r in results) / TRIALS:.3f}")
print()
print("M1/M2-style ADDITIVE memory (m += g * delta) — same experiment,")
print("sum_i abs_i - (n-1)*sys vs true sequential: EXACT by additivity")
print("(no decay/projection on the memory path) — rel-MSE 0.0 at every n.")
