# DTSS Learning Validation

Date: 2026-10-20

## Protocol

A deterministic balanced binary task used 256 synthetic training and 128 test
token sequences (`6 x 16`), direct-encoded by repeating each sequence over four
timesteps. Both arms used the same 2,412-parameter Spiking Transformer block,
initial state, data, batch order, AdamW optimizer (`lr=0.002`), batch size 32,
and 20 epochs.

- **Fixed T=4:** all seven timestep parameters frozen with every mask active.
- **Learned DTSS:** timestep parameters trainable with normalized mask-loss
  weight 0.2. All masks also began at four active steps.

## Results

| Device / arm | Test accuracy | Test loss | Mean active steps | Runtime | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU fixed T=4 | 100% | 0.00956 | 4.000 | 0.545 s | 306.3 MiB process HWM |
| CPU learned DTSS | 100% | 0.01266 | 3.286 | 0.634 s | 308.3 MiB process HWM |
| MPS fixed T=4 | 100% | 0.00957 | 4.000 | 1.890 s | 0.02618 GiB MPS driver |
| MPS learned DTSS | 100% | 0.01277 | 3.286 | 2.623 s | 0.02620 GiB MPS driver |

At epoch 10, Q, K, V, attention LIF, and MLP-hidden masks changed from
`[1,1,1,1]` to `[1,1,1,0]`; projection and MLP-output masks remained at four
steps. This reduced active layer-time slots from 28 to 23 (17.9%) with no
accuracy loss on this simple task, although learned-DTSS test loss was 0.00310
higher. Timestep parameter delta norms were 0.322-0.625 and final aggregate
gradient norm was 0.0378, confirming actual optimization rather than a static
mask. CPU and MPS produced the same learned masks and accuracy.

## Interpretation

DTSS learned valid trailing-step removal without harming synthetic accuracy.
This is a mechanism validation, not evidence for vision accuracy or energy
savings. Eager PyTorch still ran all four iterations and merely zeroed masked
outputs; consequently DTSS was slower here because it added mask gradients.
True compute/latency reduction requires compiler or hardware support that skips
the inactive trailing iterations. CPU RSS is a process-wide high-water mark and
is not an isolated per-arm allocator measurement.

Artifacts:

- `src/forestpersons/dtss_experiment.py`
- `scripts/run_dtss_synthetic_classification.py`
- `tests/test_dtss_experiment.py`
- `outputs/dtss_synthetic_classification_20261020/cpu_metrics.json`
- `outputs/dtss_synthetic_classification_20261020/mps_metrics.json`

