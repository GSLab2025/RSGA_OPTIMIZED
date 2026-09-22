# Reciprocal Space Gated Attention Optimized

This repository contains the optimized standalone Reciprocal Space Gated
Attention (RSGA) modules that correspond to the production MACERSGA code used
for the 2026 In2Se3 4x4x1 and 9x9x1 molecular-dynamics runs.

RSGA is a reciprocal-space, periodic long-range correction block designed to be
coupled to a short-range atomistic model. The optimized implementation preserves
the original RSGA physics while improving production reliability and GPU
efficiency in strict float64 workflows.

## Hardware Scope

The optimized implementation is written at the PyTorch tensor-operation level;
it does not require a custom NVIDIA-only CUDA extension for correctness. The
main algorithmic changes--shared geometry context, reciprocal-grid caching,
chunked large-graph evaluation, and safer `torch.compile` fallback behavior--are
therefore general and should run on CPU or any PyTorch-supported accelerator
with the required operations.

The performance path, however, was designed and validated for strict float64
production runs on NVIDIA CUDA GPUs, specifically A100 and H100-class hardware.
The default documented runtime settings disable TF32 and optional fp32 fast
evaluation to preserve long-range physics. CPU, AMD GPU, Apple silicon, or other
non-CUDA backends should be considered portable but not performance-validated;
benchmark and numerically validate those platforms before production use.

## Files

- `k_frequencies_triclinic.py`: builds triclinic reciprocal-space modes,
  Ewald-like spectral weights, and integer reciprocal indices for the
  cell-invariant phase identity `r . k(n) = 2*pi*(f . n)`.
- `rsga.py`: implements `ReciprocalSpaceGatedAttention` with shared geometry
  context support, node-level SR/LR mixing, cached reciprocal grids for repeated
  cells, chunked large-graph evaluation, and strict TF32 controls.

## Production Defaults

The implementation is intended for float64 production usage. The environment
flags used in the validated production workflow are:

```bash
export MACE_RSGA_ALLOW_TF32=0
export MACE_RSGA_FAST_EVAL_FP32=0
export MACE_RSGA_CHUNK_MB=1024
```

`MACE_RSGA_ALLOW_TF32=0` disables TF32 tensor-core matmul paths. This preserves
the float64/strict-numeric behavior needed for long-range physics.

`MACE_RSGA_FAST_EVAL_FP32=0` disables optional fp32 eval shortcuts. The
validated production path keeps this off.

`MACE_RSGA_CHUNK_MB=1024` sets the target memory budget for chunked large-graph
RSGA evaluation. This reduces peak memory while keeping the calculation
deterministic for the strict dtype path.

`MACE_RSGA_DISABLE_TORCH_COMPILE=1` can be set to force eager reciprocal-cell
helpers if a runtime has a `torch.compile` backend issue. By default the helper
uses `torch.compile` and falls back to eager mode automatically if compilation
fails.

## Relation To MACERSGA

The full MACE-integrated implementation is maintained in
[`GSLab2025/RSGA_MACE_OPT`](https://github.com/GSLab2025/RSGA_MACE_OPT).
That repository embeds these optimized RSGA modules into the MACE interaction
stack as a layerwise embedding correction, not as a separate additive energy
model. Example entry point in the modified MACE stack:

```bash
mace_run_train --model="MACERSGA"  --max_L=0  --pair_repulsion  --distance_transform="Agnesi" --forces_weight=1000 --energy_weight=100 --swa_forces_weight=10 --swa_energy_weight=1000 ...
```

## Provenance

This optimized code descends from the published RSGA implementation in
[`GSLab2025/RSGA`](https://github.com/GSLab2025/RSGA) and the MACE-integrated
MACERSGA implementation in
[`GSLab2025/MACE_RSGA`](https://github.com/GSLab2025/MACE_RSGA).

## How to Cite?

If you are using the RSGA and MACERSGA workflow in your research, please cite us as:

* **Boundary-driven phase transformation in layered 3R In2Se3 captured by long range machine learning molecular dynamics** Igor Evangelista, Michaela Cohen, Atul C. Thakur, et al.  *ChemRxiv. 22 September 2026. DOI: [10.26434/chemrxiv.15009227/v1](https://doi.org/10.26434/chemrxiv.15009227/v1)

* **Reciprocal Space Attention for Learning Long-Range Interactions** H. Ramasubramanian, A. Vazquez-Mayagoitia, G. Sivaraman, and A. C. Thakur.  
  *Poster in AI4Mat-NeurIPS-2025: NeurIPS 2025 Workshop on AI for Accelerated Materials Design.* Preprint: [arXiv:2510.13055](https://arxiv.org/abs/2510.13055).

* **Neutron and X-ray Diffraction Reveal the Limits of Long-Range Machine Learning Potentials for Medium-Range Order in Silica Glass** Balantrapu, S. H., Thakur, A. C., Benmore, C. J., & Sivaraman, G. (2026).  
  *Journal of Physics: Materials*. DOI: [10.1088/2515-7639/ae8643](https://doi.org/10.1088/2515-7639/ae8643)

* **RSGA: Reciprocal Space Gated Attention** Thakur, Atul C., Alvaro Vazquez-Mayagoitia, and Ganesh Sivaraman.  
  *Zenodo*, April 20, 2026. DOI: [10.5281/zenodo.19673766](https://doi.org/10.5281/zenodo.19673766)

### BibTeX

```bibtex
@article{
doi:10.26434/chemrxiv.15009227/v1,
author = {Igor Evangelista  and Michaela Cohen  and Atul C. Thakur  and Anderson Janotti  and Chris Benmore  and Tingyi Gu  and Ganesh Sivaraman },
title = {Boundary-driven phase transformation in layered 3R In2Se3 captured by long range machine learning molecular dynamics},
journal = {ChemRxiv},
volume = {2026},
number = {0922},
pages = {},
year = {2026},
doi = {10.26434/chemrxiv.15009227/v1},
URL = {https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15009227/v1},
eprint = {https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.15009227/v1}
}

@misc{ramasubramanian2025rsga,
  title={Reciprocal Space Attention for Learning Long-Range Interactions},
  author={Ramasubramanian, H. and Vazquez-Mayagoitia, A. and Sivaraman, G. and Thakur, A. C.},
  year={2025},
  eprint={2510.13055},
  archivePrefix={arXiv},
  note={Poster in AI4Mat-NeurIPS-2025: NeurIPS 2025 Workshop on AI for Accelerated Materials Design}
}

@article{Balantrapu_2026,
doi = {10.1088/2515-7639/ae8643},
url = {https://doi.org/10.1088/2515-7639/ae8643},
year = {2026},
month = {jul},
publisher = {IOP Publishing},
volume = {9},
number = {3},
pages = {035013},
author = {Balantrapu, Sai Harshit and Thakur, Atul and Benmore, Chris and Sivaraman, Ganesh},
title = {Neutron and x-ray diffraction reveal the limits of long-range machine learning potentials for medium-range order in silica glass},
journal = {Journal of Physics: Materials}
}

@misc{thakur2026rsgacode,
  title={{RSGA}: Reciprocal Space Gated Attention},
  author={Thakur, Atul C. and Vazquez-Mayagoitia, Alvaro and Sivaraman, Ganesh},
  year={2026},
  month={April},
  publisher={Zenodo},
  doi={10.5281/zenodo.19673766},
  url={[https://doi.org/10.5281/zenodo.19673766](https://doi.org/10.5281/zenodo.19673766)}
}
