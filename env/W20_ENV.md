# W20 environment manifest — CPU box (dev)

Provisioning record for the CPU-only dev box that authors W20 (the TQ kernel
family). Per PROMPT.md §6 (the latest-stable law), every dependency below was
queried (`pip index versions`), installed at its latest stable release, and
evidenced with the raw query output captured verbatim in `## evidence` — no
stale pins, no unevidenced version claims. This manifest is **evidence, not a
constraint**: re-provisioning re-queries and updates it. Nothing behavioral
ever runs on this box (PROMPT.md §1); the A10G box executes all tests and
gates.

## CPU box (dev)

| package | latest stable (pip index evidence) | installed version | date | source |
|---|---|---|---|---|
| torch | 2.14.1 | 2.14.1 (wheel; runtime `torch.__version__` = `2.14.1+cu130`) | 2026-10-11 | PyPI (default linux wheel, CUDA-bundled) |
| numpy | 2.5.4 | 2.5.4 | 2026-10-11 | PyPI |
| transformers | 5.19.0 | 5.19.0 | 2026-10-11 | PyPI |
| faiss-cpu | 1.15.1 | 1.15.1 | 2026-10-11 | PyPI |
| ninja | 1.13.2 | 1.13.2 | 2026-10-11 | PyPI |

- python3: `Python 3.12.15` — `/home/z/.venv/bin/python3` (venv)
- pip: `pip --version` → `pip 25.1.1 from /usr/lib/python3/dist-packages/pip (python 3.13)` (the bare `pip` on PATH is the system pip bound to a different interpreter, 3.13). All provisioning and evidence commands therefore ran through the venv pip of `python3`: `python3 -m pip --version` → `pip 25.0.1 from /home/z/.venv/lib/python3.12/site-packages/pip (python 3.12)` — so every install lands in the interpreter `python3` actually imports.
- git: `git version 2.47.3` (version probe only — no git state commands run by the sub-agent)
- torch.version.cuda: `13.0` — the default PyPI linux wheel is the CUDA-bundled build, importable without a GPU device; intentional, it matches the A10G target's semantics for authoring
- torch.cuda.is_available(): `False` — CPU-only box, expected; recorded as fact, NOT as an error
- nvidia-smi: ABSENT (no GPU, no driver — CPU-only dev box)
- prior state (before this task): torch `2.14.1+cpu` with `torch.version.cuda = None`; replaced by the default PyPI wheel `torch==2.14.1` (+ its CUDA dependency wheels) per PROMPT.md §6
- `pip check` (informational, out of W20 scope): pre-existing non-W20 venv packages (scipy 1.14.1, numba 0.61.0, gradio 6.3.0, gradio-client 2.0.3, gtts 2.5.4) carry upper pins that conflict with latest-stable numpy 2.5.4 / huggingface-hub 2.2.0; none of them is a W20 dependency (PROMPT.md §6 list) — recorded, not remediated

## evidence

Raw latest-stable evidence, captured 2026-10-11. Commands shown exactly as
run (`python3 -m pip` is the venv pip serving `python3`; `head -2` per the
provisioning law):

```text
$ python3 -m pip index versions torch 2>/dev/null | head -2
torch (2.14.1)
Available versions: 2.14.1, 2.14.0, 2.13.0, 2.12.1, 2.12.0, 2.11.0, 2.10.0, 2.9.1, 2.9.0, 2.8.0, 2.7.1, 2.7.0, 2.6.0, 2.5.1, 2.5.0, 2.4.1, 2.4.0, 2.3.1, 2.3.0, 2.2.2, 2.2.1, 2.2.0
```

```text
$ python3 -m pip index versions numpy 2>/dev/null | head -2
numpy (2.5.4)
Available versions: 2.5.4, 2.5.3, 2.5.2, 2.5.1, 2.5.0, 2.4.6, 2.4.5, 2.4.4, 2.4.3, 2.4.2, 2.4.1, 2.3.5, 2.3.4, 2.3.3, 2.3.2, 2.3.1, 2.3.0, 2.2.6, 2.2.5, 2.2.4, 2.2.3, 2.2.2, 2.2.1, 2.2.0, 2.1.3, 2.1.2, 2.1.1, 2.1.0, 2.0.2, 2.0.1, 2.0.0, 1.26.4, 1.26.3, 1.26.2, 1.26.1, 1.26.0, 1.25.2, 1.25.1, 1.25.0, 1.24.4, 1.24.3, 1.24.2, 1.24.1, 1.24.0, 1.23.5, 1.23.4, 1.23.3, 1.23.2, 1.23.1, 1.23.0, 1.22.4, 1.22.3, 1.22.2, 1.22.1, 1.22.0, 1.21.1, 1.21.0, 1.20.3, 1.20.2, 1.20.1, 1.20.0, 1.19.5, 1.19.4, 1.19.3, 1.19.2, 1.19.1, 1.19.0, 1.18.5, 1.18.4, 1.18.3, 1.18.2, 1.18.1, 1.18.0, 1.17.5, 1.17.4, 1.17.3, 1.17.2, 1.17.1, 1.17.0, 1.16.6, 1.16.5, 1.16.4, 1.16.3, 1.16.2, 1.16.1, 1.16.0, 1.15.4, 1.15.3, 1.15.2, 1.15.1, 1.15.0, 1.14.6, 1.14.5, 1.14.4, 1.14.3, 1.14.2, 1.14.1, 1.14.0, 1.13.3, 1.13.1, 1.13.0, 1.12.1, 1.12.0, 1.11.3, 1.11.2, 1.11.1, 1.11.0, 1.10.4, 1.10.2, 1.10.1, 1.10.0.post2, 1.9.3, 1.9.2, 1.9.1, 1.9.0, 1.8.2, 1.8.1, 1.8.0, 1.7.2, 1.7.1, 1.7.0, 1.6.2, 1.6.1, 1.6.0, 1.5.1, 1.5.0, 1.4.1, 1.3.0
```

```text
$ python3 -m pip index versions transformers 2>/dev/null | head -2
transformers (5.19.0)
Available versions: 5.19.0, 5.18.0, 5.17.0, 5.16.1, 5.16.0, 5.15.1, 5.15.0, 5.14.1, 5.14.0, 5.13.1, 5.13.0, 5.12.1, 5.12.0, 5.11.0, 5.10.4, 5.10.2, 5.10.1, 5.9.0, 5.8.1, 5.8.0, 5.7.0, 5.6.2, 5.6.1, 5.6.0, 5.5.4, 5.5.3, 5.5.2, 5.5.1, 5.5.0, 5.4.0, 5.3.0, 5.2.0, 5.1.0, 5.0.0, 4.57.6, 4.57.5, 4.57.4, 4.57.3, 4.57.2, 4.57.1, 4.56.2, 4.56.1, 4.56.0, 4.55.4, 4.55.3, 4.55.2, 4.55.1, 4.55.0, 4.54.1, 4.54.0, 4.53.3, 4.53.2, 4.53.1, 4.53.0, 4.52.4, 4.52.3, 4.52.2, 4.52.1, 4.51.3, 4.51.2, 4.51.1, 4.51.0, 4.50.3, 4.50.2, 4.50.1, 4.50.0, 4.49.0, 4.48.3, 4.48.2, 4.48.1, 4.48.0, 4.47.1, 4.47.0, 4.46.3, 4.46.2, 4.46.1, 4.45.2, 4.45.1, 4.45.0, 4.44.2, 4.44.1, 4.44.0, 4.43.4, 4.43.3, 4.43.2, 4.43.1, 4.43.0, 4.42.4, 4.42.3, 4.42.2, 4.42.1, 4.42.0, 4.41.2, 4.41.1, 4.41.0, 4.40.2, 4.40.1, 4.40.0, 4.39.3, 4.39.2, 4.39.1, 4.39.0, 4.38.2, 4.38.1, 4.38.0, 4.37.2, 4.37.1, 4.37.0, 4.36.2, 4.36.1, 4.36.0, 4.35.2, 4.35.1, 4.35.0, 4.34.1, 4.34.0, 4.33.3, 4.33.2, 4.33.1, 4.33.0, 4.32.1, 4.32.0, 4.31.0, 4.30.2, 4.30.1, 4.30.0, 4.29.2, 4.29.1, 4.29.0, 4.28.1, 4.28.0, 4.27.4, 4.27.3, 4.27.2, 4.27.1, 4.27.0, 4.26.1, 4.26.0, 4.25.1, 4.24.0, 4.23.1, 4.23.0, 4.22.2, 4.22.1, 4.22.0, 4.21.3, 4.21.2, 4.21.1, 4.21.0, 4.20.1, 4.20.0, 4.19.4, 4.19.3, 4.19.2, 4.19.1, 4.19.0, 4.18.0, 4.17.0, 4.16.2, 4.16.1, 4.16.0, 4.15.0, 4.14.1, 4.13.0, 4.12.5, 4.12.4, 4.12.3, 4.12.2, 4.12.1, 4.12.0, 4.11.3, 4.11.2, 4.11.1, 4.11.0, 4.10.3, 4.10.2, 4.10.1, 4.10.0, 4.9.2, 4.9.1, 4.9.0, 4.8.2, 4.8.1, 4.8.0, 4.7.0, 4.6.1, 4.6.0, 4.5.1, 4.5.0, 4.4.2, 4.4.1, 4.4.0, 4.3.3, 4.3.2, 4.3.1, 4.3.0, 4.2.2, 4.2.1, 4.2.0, 4.1.1, 4.1.0, 4.0.1, 4.0.0, 3.5.1, 3.5.0, 3.4.0, 3.3.1, 3.3.0, 3.2.0, 3.1.0, 3.0.2, 3.0.1, 3.0.0, 2.11.0, 2.10.0, 2.9.1, 2.9.0, 2.8.0, 2.7.0, 2.6.0, 2.5.1, 2.5.0, 2.4.1, 2.4.0, 2.3.0, 2.2.2, 2.2.1, 2.2.0, 2.1.1, 2.1.0, 2.0.0, 0.1
```

```text
$ python3 -m pip index versions faiss-cpu 2>/dev/null | head -2
faiss-cpu (1.15.1)
Available versions: 1.15.1, 1.15.0, 1.14.3, 1.14.2, 1.13.2, 1.13.1, 1.13.0, 1.12.0, 1.11.0.post1, 1.11.0, 1.10.0, 1.9.0.post1, 1.9.0, 1.8.0.post1, 1.8.0
```

```text
$ python3 -m pip index versions ninja 2>/dev/null | head -2
ninja (1.13.2)
Available versions: 1.13.2, 1.13.0, 1.11.1.4, 1.11.1.3, 1.11.1.2, 1.11.1.1, 1.11.1, 1.10.2.4, 1.10.2.3, 1.10.2.2, 1.10.2.1, 1.10.2, 1.10.0.post3, 1.10.0.post2, 1.10.0.post1, 1.10.0, 1.9.0.post1, 1.9.0, 1.8.2.post2, 1.8.2.post1, 1.8.2, 1.7.2.post2, 1.7.2.post1, 1.7.2, 1.5.3
```

Installed-version confirmation (`python3 -m pip show <pkg> | head -2`,
2026-10-11): torch `Version: 2.14.1` · numpy `Version: 2.5.4` ·
transformers `Version: 5.19.0` · faiss-cpu `Version: 1.15.1` · ninja
`Version: 1.13.2`.

```text
$ which nvcc || echo "nvcc ABSENT"
nvcc ABSENT
$ nvidia-smi || echo "nvidia-smi ABSENT"
nvidia-smi ABSENT
$ apt-cache policy nvidia-cuda-toolkit 2>/dev/null | head -5
nvidia-cuda-toolkit:
  Installed: (none)
  Candidate: (none)
  Version table:
$ apt-cache policy cuda-toolkit 2>/dev/null | head -5
(no output — no such package in the configured apt sources)
```

## nvcc

nvcc: ABSENT (not on PATH; the system package manager offers no toolkit — `apt-cache policy nvidia-cuda-toolkit` reports `Installed: (none)`, `Candidate: (none)`, and `apt-cache policy cuda-toolkit` has no entry; obtaining nvcc would require root plus an apt source this box does not have, so per the provisioning law nothing was installed)

Consequence: all compile checks (the nvcc probe, `flute_extended` CUDA
compilation, arch-flag verification) defer to the A10G box (W20.0.2 /
`scripts/gpu/w20_provision.sh`, results recorded there) — both states are
legal per PROMPT.md §6; ABSENT is the state that holds on this box.

## notes

- The law (PROMPT.md §1) held during provisioning: no repo module (anything under `src/`) was imported or executed, no test was run, and no CPU test was authored — the only `python3 -c` probes were environment probes importing third-party modules only (torch / numpy / transformers / faiss / ninja).
- Two-box split: this CPU box authors code, tests, and gate scripts (static work only); the A10G box builds `flute_extended` and runs ALL tests plus the G20 gates — no behavioral evidence is ever produced here.
- Drift rule: the CPU-box and A10G-box torch MAJOR versions must match (this box: torch 2.14.1, major 2, CUDA 13.0 build); any mismatch is recorded in both manifests and flagged in the wave attestation wherever authoring touched version-sensitive APIs.
