# e-means ROCm probe scripts

Offline compile checks: no GPU needed, only `pip install triton` (tested with upstream 3.9.0).
Put all files in one directory and run from there. `SRC` is the path to
`faiss/contrib/e_means/triton_assignment.py` (tested at faiss commit `14392c9`).

Run in this order:

```bash
python probe.py gfx942 gfx950 gfx1100 gfx1200          # minimal fusion on/off probe
python port.py   SRC gfx942 gfx950 gfx1100 gfx1200     # swaps the 10 PTX sites, compiles; writes ported_kernels.py
python attrib.py gfx942 gfx1200                        # needs ported_kernels.py; fused ops per source line
python nv.py                                           # needs ported_kernels.py and probe_kernels.py; NVIDIA PTX comparison
python assign.py SRC gfx942 gfx950 gfx1100 gfx1200     # real assignment kernels: matrix instruction + LDS per arch
```

Notes:
- `port.py` and `assign.py` read the e-means source as text and generate `ported_kernels.py` / `assign_kernels.py` next to themselves.
- "error: could not allocate output register for constraint 'f'" printed by `probe.py` and `nv.py` is expected: it is the PTX snippet being compiled for an AMD target.
- `probe.py` passes wavefront size 64 for every target; Triton overrides it to 32 on gfx10 and later, so results are unaffected.
- `lds_sweep.py`: offline LDS check of the kernels added by the b14b3d69 sync, at the block sizes their launchers pick, for gfx942/gfx950/gfx1100/gfx1200 (no GPU; about 3 min on one core). The assignment kernels are checked by `tests/torch_test_e_means_launch_config.py`.
