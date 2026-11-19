# e-means-rocm

AMD ROCm support for faiss `contrib/e_means` (E-Means mini-batch k-means), developed as a
standalone repo. Targets: gfx942, gfx950, gfx1100, gfx1200 via the Triton backend.

Based on [facebookresearch/faiss](https://github.com/facebookresearch/faiss) `main` at commit
`14392c952788ca1d6980361cf01d1c8736afc4db` (MIT license, see `LICENSE`). The first commit is an
unmodified snapshot of `contrib/e_means`, `benchs/e_means` and the e_means tests; paths mirror
faiss so changes can be cherry-picked upstream.

Layout: `contrib/e_means/` source, `benchs/e_means/`, `tests/`, `tools/rocm_probe/` (offline
Triton compile probes), `tools/w0_smoke.py` (precision x backend smoke matrix).
