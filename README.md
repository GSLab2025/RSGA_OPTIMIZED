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
mace_run_train --model="MACERSGA" --pair_repulsion  --distance_transform="Agnesi"  ...
```

## Provenance

This optimized code descends from the published RSGA implementation in
[`GSLab2025/RSGA`](https://github.com/GSLab2025/RSGA) and the MACE-integrated
MACERSGA implementation in
[`GSLab2025/MACE_RSGA`](https://github.com/GSLab2025/MACE_RSGA).
