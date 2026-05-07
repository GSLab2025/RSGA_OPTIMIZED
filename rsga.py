# -------------  mace/modules/rsga.py  -----------------
from __future__ import annotations

import math
import os
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
from e3nn.o3 import Irreps
from torch import nn

from k_frequencies_triclinic import EwaldPotentialTriclinic
#from line_profiler import profile
#from mace.tools.scatter import scatter_sum
_ALLOW_TF32 = os.getenv("MACE_RSGA_ALLOW_TF32", "0") != "0"
torch.backends.cuda.matmul.allow_tf32 = _ALLOW_TF32
torch.backends.cudnn.allow_tf32 = _ALLOW_TF32
torch.set_float32_matmul_precision("high" if _ALLOW_TF32 else "highest")  # PyTorch 2.x

_KGRID_CACHE: Dict[Tuple[int, str, str, float, Tuple[float, ...]], Tuple[torch.Tensor, torch.Tensor]] = {}
_KGRID_CACHE_ORDER: List[Tuple[int, str, str, float, Tuple[float, ...]]] = []
_KGRID_CACHE_MAXSIZE = 16


def _default_phase_cache_threshold() -> int:
    """
    Return the large-graph phase-cache threshold used for checkpoint fallbacks.

    Older trained MACERSGA checkpoints do not serialize this optimization knob,
    so the runtime default must remain explicit and centrally defined.
    """

    return int(os.getenv("MACE_RSGA_PHASE_CACHE_THRESHOLD", "1000000"))


def _default_pairwise_eval_threshold() -> int:
    """
    Return the large-graph pairwise threshold used for checkpoint fallbacks.

    The validated large-graph silica path activates near one thousand atoms.
    Keeping the fallback here preserves the optimized behavior for older frozen
    production checkpoints that predate this attribute.
    """

    return int(os.getenv("MACE_RSGA_PAIRWISE_THRESHOLD", "1024"))

# ---------- helper: slice that contains ONLY the 0e channels --------------

def scalar_slice(irreps: Irreps) -> slice:
    """
    Returns a slice that grabs *all* 0e channels at the front of `irreps`,
    assuming they are stored first (default MACE ordering).
    Works with both new and old e3nn iterators.  
    """
    start = 0
    for mul_ir in irreps:
        # new API: _MulIr;  old API: Irrep
        ir   = mul_ir.ir if hasattr(mul_ir, "ir") else mul_ir
        mul  = mul_ir.mul if hasattr(mul_ir, "mul") else 1

        if ir.l != 0 or ir.p != 1:     # not a scalar-even channel
            break
        start += mul * ir.dim          # each copy contributes ir.dim dims
    return slice(0, start)             # [:start] are scalars


class RSGABatchContext(NamedTuple):
    """
    Geometry-only tensors reused across every reciprocal-space layer.

    MACERSGA applies RS-LGA after every message-passing block, but the
    fractional coordinates and triclinic reciprocal grid depend only on the
    input geometry, not on the layer weights. Building them once per forward
    removes redundant work without changing the learned physics or checkpoint
    layout.
    """

    fractional_positions: torch.Tensor
    node_offsets: torch.Tensor
    n_vectors: torch.Tensor
    weights: torch.Tensor
    mode_offsets: torch.Tensor
    phase_cos_blocks: Tuple[Optional[torch.Tensor], ...]
    phase_sin_blocks: Tuple[Optional[torch.Tensor], ...]


@torch.jit.unused
def _get_cached_kgrid(
    kspace_freq: EwaldPotentialTriclinic,
    pos_g: torch.Tensor,
    cell_g: torch.Tensor,
    dtype: torch.dtype,
    r_cut: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cache reciprocal grids for repeated evaluations of the same simulation cell.

    In production MD the cell is often fixed for long stretches, so rebuilding
    the integer k-grid every force call is pure overhead. The cache is keyed by
    the exact cell entries plus the RS-LGA module identity so it stays strictly
    backward-compatible with the current weighting scheme.
    """

    key = (
        id(kspace_freq),
        str(cell_g.device),
        str(dtype),
        float(r_cut),
        tuple(float(x) for x in cell_g.detach().reshape(-1).cpu().tolist()),
    )
    cached = _KGRID_CACHE.get(key)
    if cached is not None:
        return cached

    n_g, w_g = kspace_freq(pos_g, cell_g, r_cut=r_cut, return_n=True)
    cached_value = (n_g.to(dtype=dtype), w_g.to(dtype=dtype))
    _KGRID_CACHE[key] = cached_value
    _KGRID_CACHE_ORDER.append(key)

    if len(_KGRID_CACHE_ORDER) > _KGRID_CACHE_MAXSIZE:
        old_key = _KGRID_CACHE_ORDER.pop(0)
        _KGRID_CACHE.pop(old_key, None)

    return cached_value


def build_rsga_batch_context(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch: torch.Tensor,
    dtype: torch.dtype,
    kspace_freq: EwaldPotentialTriclinic,
    r_cut: float,
    exact_eval_fast_path: bool = False,
    phase_cache_threshold: Optional[int] = None,
    pairwise_eval_threshold: Optional[int] = None,
) -> RSGABatchContext:
    """
    Precompute the geometry shared by every RS-LGA layer in one forward pass.

    The existing training, ASE, and LAMMPS paths keep atoms grouped by graph,
    so graph slices can be reused directly through prefix offsets rather than
    rebuilding boolean masks inside every long-range layer.
    """

    if phase_cache_threshold is None:
        phase_cache_threshold = _default_phase_cache_threshold()
    if pairwise_eval_threshold is None:
        pairwise_eval_threshold = _default_pairwise_eval_threshold()

    num_graphs = int(cell.shape[0])
    node_counts = torch.bincount(batch.to(torch.long), minlength=num_graphs)
    node_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=batch.device)
    node_offsets[1:] = torch.cumsum(node_counts, dim=0)

    frac_blocks: List[torch.Tensor] = []
    n_blocks: List[torch.Tensor] = []
    w_blocks: List[torch.Tensor] = []
    phase_cos_blocks: List[Optional[torch.Tensor]] = []
    phase_sin_blocks: List[Optional[torch.Tensor]] = []
    mode_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=batch.device)
    total_modes = 0
    two_pi = 2.0 * math.pi

    for g in range(num_graphs):
        start = int(node_offsets[g].item())
        end = int(node_offsets[g + 1].item())

        pos_g = positions[start:end].to(dtype)
        cell_g = cell[g].to(dtype)

        frac_g = torch.linalg.solve(cell_g.mT, pos_g.mT).mT
        frac_g = frac_g - torch.floor(frac_g)
        frac_blocks.append(frac_g)

        if torch.jit.is_scripting():
            n_g, w_g = kspace_freq(pos_g, cell_g, r_cut=r_cut, return_n=True)
            n_blocks.append(n_g.to(dtype=dtype))
            w_blocks.append(w_g.to(dtype=dtype))
        else:
            n_g, w_g = _get_cached_kgrid(
                kspace_freq=kspace_freq,
                pos_g=pos_g,
                cell_g=cell_g,
                dtype=dtype,
                r_cut=r_cut,
            )
            n_blocks.append(n_g)
            w_blocks.append(w_g)

        num_nodes = end - start
        num_modes = int(w_g.shape[0])
        cache_phase = exact_eval_fast_path and num_modes > 0 and (
            num_nodes >= pairwise_eval_threshold
            or (num_nodes * num_modes) >= phase_cache_threshold
        )
        if cache_phase:
            phase_t = (two_pi * (frac_g @ n_g.T)).transpose(0, 1).contiguous()
            phase_cos_blocks.append(phase_t.cos().unsqueeze(-1))
            phase_sin_blocks.append(phase_t.sin().unsqueeze(-1))
        else:
            phase_cos_blocks.append(None)
            phase_sin_blocks.append(None)

        total_modes += int(w_g.shape[0])
        mode_offsets[g + 1] = total_modes

    fractional_positions = torch.cat(frac_blocks, dim=0)
    if total_modes > 0:
        n_vectors = torch.cat(n_blocks, dim=0)
        weights = torch.cat(w_blocks, dim=0)
    else:
        n_vectors = positions.new_zeros((0, 3), dtype=dtype)
        weights = positions.new_zeros((0,), dtype=dtype)

    return RSGABatchContext(
        fractional_positions=fractional_positions,
        node_offsets=node_offsets,
        n_vectors=n_vectors,
        weights=weights,
        mode_offsets=mode_offsets,
        phase_cos_blocks=tuple(phase_cos_blocks),
        phase_sin_blocks=tuple(phase_sin_blocks),
    )


# -----------------------------------------------------------------------------
# Reciprocal-Space Linear Gated Attention (RS-LGA) with Fractional Fourier Phase Encoding
#
# Goal:
#   Implement a physically rigorous reciprocal-space attention mechanism that is:
#   (i) Periodicity-aware via Ewald summation logic.
#   (ii) Correct for Triclinic geometries (handling skewed lattice symmetries).
#   (iii) Invariant to cell definitions (supercells) via fractional coordinates.
#   (iv) Expressive via Gated Linear Attention (GLA) mechanisms.
#
# Key Idea (Fractional Fourier Encoding):
#   For a periodic cell with lattice matrix A (3x3), reciprocal vectors are:
#       k(n) = 2π * n * A^{-T},   where n ∈ Z^3 (integer triplets)
#   Atomic positions r are mapped to fractional coordinates f:
#       r = A f  =>  f = r A^{-1}
#
#   The Fourier phase satisfies the exact identity:
#       r · k(n) = (A f) · (2π n A^{-T}) = 2π (f · n)
#
#   Consequence:
#   The phase basis (cos/sin) depends solely on fractional positions f and
#   integer indices n, making it invariant to cell deformations. Geometry enters
#   only through the spectral weights w(k) derived from the Ewald kernel.
#
# Architecture (RS-LGA):
#   1. Input Gating ("Source Filter"):
#      - An element-wise sigmoid gate filters the Value vectors v before summation.
#      - Allows atoms to broadcast specific physical features (chemical identity)
#        while suppressing others based on local context.
#        v_gated = v * σ(W_in x)
#
#   2. Triclinic-Correct Phase Encoding:
#      - We generate a symmetric integer grid n ∈ [-N, N] to capture all
#        reflections in skewed triclinic lattices.
#      - Q and K are phase-encoded (rotated) by exp(i 2π f·n).
#
#   3. Linear Attention Aggregation:
#      - We compute the "global field" S_m for each frequency mode m:
#        S_m = Σ_t K_enc[m,t] ⊗ V_gated[t]      (Linear complexity)
#      - The unweighted update is retrieved via query projection:
#        β_m[i] = Q_enc[m,i]^T S_m
#
#   4. Weighted Accumulation:
#      - Modes are summed using Ewald spectral weights w_m:
#        update[i] = Σ_m w_m β_m[i]
#      - This is implemented as a highly optimized BLAS GEMV operation.
#
#   5. Output Gating ("Result Scaling"):
#      - A head-wise (scalar) tanh gate scales the accumulated field.
#      - Uses an identity-centered update rule for stability:
#        Output = Update * (1.0 + 0.2 * tanh(W_out x))
#
# Practical Notes:
#   - Fractional coordinates are wrapped to [0,1) to handle unwrapped MD trajectories.
#   - Processing is chunked over modes (Mc) to maintain constant memory footprint.
#   - Critical Float64 precision is enforced for grid generation and weight accumulation.
# -----------------------------------------------------------------------------
class ReciprocalSpaceGatedAttention(nn.Module): 
    def __init__(self, node_irreps, r_max: float, hidden: int = None, Mc: int = 128):
        super().__init__()
        self.scalar_sl = scalar_slice(node_irreps)
        S = self.scalar_sl.stop                      # #scalar channels

        if hidden is None:                               # ← NEW
            hidden = S                                   # ← NEW
        assert hidden % 2 == 0, "hidden must be even"    # ← keep existing check
        assert S % 2 == 0, "scalar channel count must be even (real+imag)."

        self.H = int(hidden) 
        self.Mc = int(Mc)

        # project scalar slice to hidden if needed (important if S != hidden)
        self.in_proj = nn.Identity() if S == hidden else nn.Linear(S, hidden, bias=False)

        self.qkv = nn.Linear(hidden, 3*hidden, bias=False)
        self.act    = nn.SiLU() 
        self.scale_q = 1 / math.sqrt(self.H)
        #self.norm   = nn.RMSNorm(hidden)

        # 1. Input Gate: Element-wise [Inspired by Gated Linear Attention (GLA) ]
        # We keep this element-wise (N, H) to allow "Source Filtering"
        # If this also decreases accuracy, we will downgrade it to (N, 1) later.
        self.val_gate = nn.Linear(hidden, hidden, bias=True)
        nn.init.zeros_(self.val_gate.weight)
        nn.init.constant_(self.val_gate.bias, 2.0) # Sigmoid(2) ≈ 0.88 (Open Valve)

        #Define the output and mixing  gates
        # 2. Output Gate: Head-wise (The Winner)
        # Reverted to (N, 1) as per P benchmarks
        self.head_gate = nn.Linear(hidden, 1, bias=True)   # headwise gate (N,1); Recommend option by https://arxiv.org/pdf/2505.06708
        # ZERO INITIALIZATION IS CRITICAL
        # This ensures tanh(0) = 0, so the multiplier starts at exactly 1.0
        nn.init.zeros_(self.head_gate.weight)
        nn.init.zeros_(self.head_gate.bias)          # tanh(0)=0  -> scale starts at 1.0
        
        #self.elem_gate =  nn.Linear(hidden, hidden, bias=True)  # elementwise gate (N,H) #Not recommended option by https://arxiv.org/pdf/2505.06708
        # ZERO INITIALIZATION IS CRITICAL
        # This ensures tanh(0) = 0, so the multiplier starts at exactly 1.0
        #nn.init.zeros_(self.elem_gate.weight)
        #nn.init.zeros_(self.elem_gate.bias)

        #self.mix_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.get_default_dtype())) #global scalar
        self._install_mix_gate_node()


        self.kspace_freq = EwaldPotentialTriclinic(
            auto_sigma=True, 
            eps_real=1e-4, # TIGHT tolerance for smooth handover
            auto_cut=True, 
            eps_k=1e-3, # Minimum tolerance 
            eps_mass=1e-6, # TIGHT tolerance (crucial for accurate tail)
            normalize_weights=False, # <--- CRITICAL: Disable artificial normalization
            M_cap=1024 # Sufficient capacity
        )

        self.r_cut = r_max   # use your SR cutoff as r_c for auto-sigma
        self.fast_eval_fp32 = os.getenv("MACE_RSGA_FAST_EVAL_FP32", "0") != "0"
        self.chunk_target_mb = int(os.getenv("MACE_RSGA_CHUNK_MB", "1024"))
        self.phase_cache_threshold = _default_phase_cache_threshold()
        self.pairwise_eval_threshold = _default_pairwise_eval_threshold()

    def _install_mix_gate_node(
        self,
        bias_value: Optional[float] = None,
        ref_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        gate = nn.Linear(self.H, 1, bias=True)
        nn.init.zeros_(gate.weight)
        nn.init.constant_(gate.bias, -5.0 if bias_value is None else float(bias_value))
        if ref_tensor is not None:
            gate = gate.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        self.mix_gate_node = gate

    def _ensure_mix_gate_node(self, ref_tensor: Optional[torch.Tensor] = None) -> None:
        if hasattr(self, "mix_gate_node"):
            return

        legacy_bias = None
        if hasattr(self, "mix_gate"):
            legacy_bias = float(self.mix_gate.detach().item())
            del self.mix_gate

        self._install_mix_gate_node(bias_value=legacy_bias, ref_tensor=ref_tensor)

    def __setstate__(self, state):
        super().__setstate__(state)
        self._ensure_mix_gate_node()

    def get_compute_dtype(self, node_feat: torch.Tensor) -> torch.dtype:
        """
        Use a faster CUDA eval path for MD/inference while keeping training on the
        original model dtype.
        """

        fast_eval_fp32 = getattr(
            self,
            "fast_eval_fp32",
            os.getenv("MACE_RSGA_FAST_EVAL_FP32", "0") != "0",
        )
        if node_feat.is_cuda and not self.training and fast_eval_fp32:
            return torch.float32
        return node_feat.dtype

    def _use_exact_eval_fast_path(self) -> bool:
        """
        Enable exact eval-only kernel rewrites for inference-style execution.

        MACECalculator commonly runs frozen checkpoints without toggling every
        submodule into eval mode first. Keep the fast path available there so
        older production checkpoints still reach the validated large-graph path.
        """

        qkv_weight = getattr(self.qkv, "weight", None)
        is_frozen = qkv_weight is not None and not qkv_weight.requires_grad
        return ((not self.training) and (not torch.is_grad_enabled())) or is_frozen

    def _effective_chunk_size(self, num_nodes: int, dtype: torch.dtype) -> int:
        """
        Choose a mode chunk size that limits the RS-LGA working set on large cells.

        The original fixed chunk of 128 is too large for 3000-atom silica and
        similar MD cells because `q_rot`, `k_rot`, and `beta_blk` dominate memory
        traffic. Keeping the block within a target memory budget improves GPU
        throughput by reducing allocator pressure and bandwidth thrash.
        """

        if num_nodes <= 0:
            return self.Mc

        elem_size = torch.empty((), dtype=dtype).element_size()
        bytes_per_mode = max(1, num_nodes * (3 * self.H + 4) * elem_size)
        chunk_target_mb = getattr(
            self,
            "chunk_target_mb",
            int(os.getenv("MACE_RSGA_CHUNK_MB", "1024")),
        )
        target_bytes = max(1, chunk_target_mb) * 1024 * 1024
        chunk = max(1, target_bytes // bytes_per_mode)
        chunk = min(self.Mc, int(chunk))
        if chunk >= 8:
            chunk = max(8, (chunk // 8) * 8)
        return max(1, chunk)

    @staticmethod
    def _rotate_from_phase(a: torch.Tensor, b: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        a,b: (N_g, H/2)
        phase: (N_g, M_blk)   (NOT transposed)
        returns: (M_blk, N_g, H)  with the same "cat" convention as previous code
        """
        # optional stability: keep phase small for trig, use if not float64
        #phase = torch.remainder(phase, 2 * math.pi)

        # a,b -> (1,N_g,H/2) broadcast against (M_blk,N_g,1)
        rot_a = a.unsqueeze(0) * cos - b.unsqueeze(0) * sin   # (M_blk,N_g,H/2)
        rot_b = a.unsqueeze(0) * sin + b.unsqueeze(0) * cos   # (M_blk,N_g,H/2)

        return torch.cat([rot_a, rot_b], dim=-1)              # (M_blk,N_g,H)

    @staticmethod
    def _rotate_feature_major_from_phase(
        a: torch.Tensor,
        b: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rotate keys directly into feature-major layout so the dominant
        reciprocal contraction can run as a batched GEMM without an extra
        transpose-contiguous step in the hot loop.
        """

        top = (a.unsqueeze(0) * cos - b.unsqueeze(0) * sin).transpose(1, 2)
        bottom = (a.unsqueeze(0) * sin + b.unsqueeze(0) * cos).transpose(1, 2)
        return torch.cat([top, bottom], dim=1)  # (M_blk, H, N_g)

    def _should_use_pairwise_eval(self, num_nodes: int) -> bool:
        """
        Decide whether the exact dense pairwise contraction should be used.

        This remains an evaluation-only optimization. For large graphs it
        replaces many repeated reciprocal-mode `bmm` launches with a smaller
        number of dense GEMMs while preserving the exact finite reciprocal sum.
        """

        return self._use_exact_eval_fast_path() and (
            num_nodes
            >= getattr(
                self,
                "pairwise_eval_threshold",
                _default_pairwise_eval_threshold(),
            )
        )

    def _pairwise_eval_update(
        self,
        a_qg: torch.Tensor,
        b_qg: torch.Tensor,
        a_kg: torch.Tensor,
        b_kg: torch.Tensor,
        v_g: torch.Tensor,
        w_g: torch.Tensor,
        cos_full: torch.Tensor,
        sin_full: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the exact finite reciprocal-space sum as a dense pairwise kernel
        for very large eval graphs. This changes contraction order only.
        """

        phase_re = cos_full.squeeze(-1).transpose(0, 1).contiguous()
        phase_im = sin_full.squeeze(-1).transpose(0, 1).contiguous()
        w_row = w_g.reshape(1, -1)
        phase_re_w = phase_re * w_row
        phase_im_w = phase_im * w_row

        phase_kernel_re = phase_re_w @ phase_re.T
        phase_kernel_re.addmm_(phase_im_w, phase_im.T, beta=1.0, alpha=1.0)
        phase_kernel_im = phase_im_w @ phase_re.T
        phase_kernel_im.addmm_(phase_re_w, phase_im.T, beta=1.0, alpha=-1.0)

        a_qs = a_qg * self.scale_q
        b_qs = b_qg * self.scale_q
        feature_kernel_re = a_qs @ a_kg.T
        feature_kernel_re.addmm_(b_qs, b_kg.T, beta=1.0, alpha=1.0)
        feature_kernel_im = b_qs @ a_kg.T
        feature_kernel_im.addmm_(a_qs, b_kg.T, beta=1.0, alpha=-1.0)

        feature_kernel_re.mul_(phase_kernel_re)
        feature_kernel_re.addcmul_(feature_kernel_im, phase_kernel_im, value=-1.0)
        feature_kernel_re.div_(float(v_g.shape[0]))
        return feature_kernel_re @ v_g

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        node_feat: torch.Tensor,
        rsga_ctx: Optional[RSGABatchContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sl = self.scalar_sl
        scalars = node_feat[:, sl]                 # (N,S)
        scalars = self.in_proj(scalars)            # (N,H)
        compute_dtype = self.get_compute_dtype(node_feat)

        if rsga_ctx is None:
            positions = data["positions"].to(node_feat.dtype)
            cell = data["cell"].view(-1, 3, 3).to(node_feat.dtype)
            if data["batch"] is None:
                batch = torch.zeros(
                    positions.shape[0], dtype=torch.long, device=positions.device
                )
            else:
                batch = data["batch"]
            rsga_ctx = build_rsga_batch_context(
                positions=positions,
                cell=cell,
                batch=batch,
                dtype=compute_dtype,
                kspace_freq=self.kspace_freq,
                r_cut=self.r_cut,
            )

        N = node_feat.shape[0]
        G = int(rsga_ctx.node_offsets.numel()) - 1
        H = self.H #Placeholder var
        two_pi = 2.0 * math.pi

        # Q,K,V per node
        q, k, v = self.qkv(scalars).chunk(3, dim=-1)         # (N, H) each
        q, k    = self.act(q), self.act(k)       # (N,H)

        # ---------------------------------------------------------
        # GLA INSERTION (Input Valve)
        # ---------------------------------------------------------
        # Gate the Values *before* they enter the summation.
        # This allows the atom to "filter" what it broadcasts to the grid.
        g_in = torch.sigmoid(self.val_gate(scalars)) # (N, H)
        q = q.to(compute_dtype)
        k = k.to(compute_dtype)
        v = (v * g_in).to(compute_dtype)
        
        # Pre-split real/imag pairs once
        a_q, b_q = q[..., 0::2], q[..., 1::2]           # (N,H/2)
        a_k, b_k = k[..., 0::2], k[..., 1::2]           # (N,H/2)

        update = torch.zeros((N, H), device=q.device, dtype=q.dtype)

        # Graphwise (no padding): each graph has its own M_g, but the geometry
        # has already been assembled once per batch in build_rsga_batch_context.
        for g in range(G):
            start = int(rsga_ctx.node_offsets[g].item())
            end = int(rsga_ctx.node_offsets[g + 1].item())
            if end <= start:
                continue

            N_g_t = v.new_tensor(end - start)
            f_g = rsga_ctx.fractional_positions[start:end]  # (N_g,3)
            v_g = v[start:end]                              # (N_g,H)
            # Stabilizer 1: This prevents the k=0 singularity from leaking into low-k modes.
            #v_g = v_g - v_g.mean(dim=0, keepdim=True)
            #print("position", pos_g)

            a_qg, b_qg = a_q[start:end], b_q[start:end]             # (N_g,H/2)
            a_kg, b_kg = a_k[start:end], b_k[start:end]             # (N_g,H/2)

            mode_start = int(rsga_ctx.mode_offsets[g].item())
            mode_end = int(rsga_ctx.mode_offsets[g + 1].item())
            M_g = mode_end - mode_start
            if M_g <= 0:
                continue

            n_vecs_g = rsga_ctx.n_vectors[mode_start:mode_end]
            w_g = rsga_ctx.weights[mode_start:mode_end]
            cos_full = rsga_ctx.phase_cos_blocks[g]
            sin_full = rsga_ctx.phase_sin_blocks[g]

            if self._should_use_pairwise_eval(end - start):
                if cos_full is None or sin_full is None:
                    phase_t = (two_pi * (f_g @ n_vecs_g.T)).transpose(0, 1).contiguous()
                    cos_full = phase_t.cos().unsqueeze(-1)
                    sin_full = phase_t.sin().unsqueeze(-1)
                update[start:end] = self._pairwise_eval_update(
                    a_qg=a_qg,
                    b_qg=b_qg,
                    a_kg=a_kg,
                    b_kg=b_kg,
                    v_g=v_g,
                    w_g=w_g,
                    cos_full=cos_full,
                    sin_full=sin_full,
                )
                continue

            upd_g = torch.zeros((end - start, H), device=q.device, dtype=q.dtype)
            Mc_eff = self._effective_chunk_size(end - start, q.dtype)
            v_batch = v_g.unsqueeze(0)

            # chunk over k-modes to control memory
            for m0 in range(0, M_g, Mc_eff):
                m1 = min(m0 + Mc_eff, M_g)
                w_blk = w_g[m0:m1]                     # (M_blk,)
                if cos_full is None or sin_full is None:
                    n_blk = n_vecs_g[m0:m1]                # (M_blk,3)
                    phase_t = (two_pi * (f_g @ n_blk.T)).transpose(0, 1)
                    cos = phase_t.cos().unsqueeze(-1)
                    sin = phase_t.sin().unsqueeze(-1)
                else:
                    cos = cos_full[m0:m1]
                    sin = sin_full[m0:m1]

                # rotate q,k for this block: (M_blk,N_g,H)
                q_rot = self._rotate_from_phase(a_qg, b_qg, cos, sin) * self.scale_q
                k_rot_t = self._rotate_feature_major_from_phase(a_kg, b_kg, cos, sin)

                S_blk = torch.bmm(k_rot_t, v_batch.expand(m1 - m0, -1, -1))
                S_blk = (S_blk / N_g_t) * w_blk.view(-1, 1, 1)
                upd_g.add_(torch.bmm(q_rot, S_blk).sum(dim=0))


            update[start:end] = upd_g

        # ---------------------------------------------------------
        # OUTPUT GATING (The Winner)
        # ---------------------------------------------------------
        update = update.to(scalars.dtype)
        # Back to Head-wise (N, 1) based on P benchmarks.
        attn_gate = torch.tanh(self.head_gate(scalars)  )      # (N,1) head-wise gate beat element-wise for P benchmark. So we keep this!
        #attn_gate = torch.tanh(self.elem_gate(scalars))      # (N,H) ; element-wise leads to (N, H) * (N, H) element-wise multiplication for update!

        update = update * (1.0 + 0.2 * attn_gate )                        # Broadcast (N, H) * (N, 1)
        #print(attn_gate.mean(), attn_gate.min(), attn_gate.max())

        self._ensure_mix_gate_node(ref_tensor=scalars)
        gate_sr_lr = torch.sigmoid(self.mix_gate_node(scalars))  # (N,1)
        return update, gate_sr_lr
