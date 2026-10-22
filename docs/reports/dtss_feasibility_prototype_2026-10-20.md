# DTSS Feasibility Prototype

Date: 2026-10-20

## Scope

This isolated prototype implements Section III-A of Datta et al., *Spiking
Neural Networks with Dynamic Time Steps for Vision Transformers*
([arXiv:2311.16456](https://arxiv.org/abs/2311.16456)). It does not modify or
invoke any DINOv3, YOLO, FCOS, dataset, or checkpoint code.

## Implemented

- A trainable length-`T_max` timestep parameter vector.
- Non-increasing cumulative scores equivalent to multiplication by the paper's
  lower-triangular coefficient matrix.
- Binary masks with threshold 1 and the paper's rectangular surrogate gradient
  (`d mask / d score = 1` for `0 < score < 2`).
- A non-negative `softplus` parameterization of timestep increments. This is a
  safety refinement beyond the paper's stated non-negative initialization: it
  guarantees masks always retain a consecutive initial prefix and only remove
  trailing timesteps after optimizer updates.
- Initial active scores are placed above, rather than exactly on, threshold 1
  so CPU and MPS floating-point rounding produce the same initial mask.
- LIF linear layers with soft membrane reset and the triangular spike surrogate
  described in the paper's preliminaries.
- A two-layer synthetic spiking MLP with `T_max=4` and differentiable mask loss.
- One minimal Spiking Transformer block with DTSS-filtered spiking Q/K/V,
  scaled `(QK^T)V` attention without softmax, an attention LIF stage, spiking
  output projection, two residual branches, and a spiking MLP.
- Batch normalization after token-wise projections, with no layer normalization,
  dropout, or drop path, following the paper's architectural modifications.

## Explicit Simplifications

- Token projections use `nn.Linear`; this is the token-wise equivalent needed
  for the prototype, while the paper's full compact ViT implementation replaces
  linear layers with convolutional layers.
- The attention scale is `head_dim^-0.5`. The paper specifies scaled Q/K/V
  dot-product attention without softmax but does not give a different constant
  in Section III.
- Each of the seven LIF sites has an independent DTSS mask. The paper analyzes
  and regularizes layer categories (`qkv`, `attn`, and `mlp`) at network scale;
  this single block does not implement category-specific loss selection.
- Eager PyTorch still evaluates all four loop iterations and masks their
  outputs. Actual trailing-step compute skipping needs compiler/hardware support,
  as the paper also notes.

## Not Implemented

- A full multi-block Spiking Transformer, spiking patch splitter, positional
  encoding, classifier, or network-level mask-loss schedule.
- Dataset training, accuracy/latency/energy evaluation, hardware-aware early
  termination, or layer-group-specific mask regularization.
- Integration with DINOv3. The existing DINOv3 encoder is an ANN and cannot be
  called a Spiking Transformer merely by attaching this module.

## Validation

Focused tests cover mask monotonicity and prefix structure, `T_max=4` forward
shapes, finite/nonzero timestep gradients, linear-weight gradients, invalid
time dimensions, and CPU/MPS forward/backward compatibility.

`PYTHONPATH=src:. .venv-dinov3/bin/python -m pytest -q tests/test_dtss.py`
completed with **8 passed**. The Transformer smoke used synthetic
`[T=4, B=2, N=5, C=16]` tokens and produced the same shape on CPU and MPS.
All seven masks were `[1, 1, 0, 0]`; outputs and backward gradients were finite,
and every timestep parameter had a nonzero gradient (minimum norm `0.001695`
on CPU and `0.001896` on MPS). The block contains 2,412 parameters.

Step 3's controlled learning experiment is reported in
`docs/reports/dtss_learning_validation_2026-10-20.md`.

