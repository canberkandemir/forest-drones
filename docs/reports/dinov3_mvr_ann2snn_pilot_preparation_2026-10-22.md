# DINOv3 MVR ANN2SNN Pilot Preparation

## Completed MVR result

The matched frozen-DINOv3 train5000/val600 comparison is complete. The
baseline FCOS obtained AP 0.3267, AP50 0.7915, AP75 0.2022, and AR@100 0.4550.
MVR + FCOS obtained AP 0.3623, AP50 0.8162, AP75 0.2675, and AR@100 0.4900.
At the exploratory FP/image budget 0.15, MVR recall/R20 were 0.7210/0.4885,
versus 0.6916/0.4425 for the matched baseline.

## Conversion boundary

The pilot keeps cached frozen DINOv3 block 3/6/9/12 features as the analog
feature extractor. Those features are signed, so they are not treated as
binary spikes. Six trained ReLU sites in the MVR readout are converted:

- four 1x1 layer-projection ReLUs;
- fine-context depthwise-convolution ReLU;
- coarse-context depthwise-convolution ReLU.

Thresholds are scalar 99.9th-percentile positive activations calibrated only
from a deterministic prefix of the train5000 manifest. Deterministic
soft-reset integrate-and-fire neurons emit at most one spike per neuron per
timestep. Their decoded rate is `spike_count * threshold / T`.

The MVR residual projection and L2 normalization remain analog. The complete
FCOS tower also remains analog because its GroupNorm/GELU sequence is not a
direct ReLU ANN2SNN conversion and changing it without retraining would not be
faithful. Classification, centerness, bounding-box regression, decoding, and
NMS are unchanged and analog.

## Pilot protocol

- Existing trained MVR checkpoint and cached train/val features only.
- ANN and hybrid SNN use the same evaluation images and decoder settings.
- Timesteps: 4, 8, and 16.
- Fixed confidence 0.63, inherited from the completed MVR comparison; no pilot
  threshold search.
- Metrics: AP50, IoU-0.50 recall, R20, precision, FP/image, model latency, and
  per-site/overall spike activity.
- Default bounded command: 100 train calibration images and 100 val images.
- Output directory is overwrite-protected.

## Interpretation limits

This is a hybrid, layerwise rate-coded conversion study. Dense convolutions
still execute on MPS, so measured Mac latency is not neuromorphic latency.
Spike activity is only an activity proxy and cannot establish energy savings.
A full detector conversion would require a justified treatment of signed
features, GroupNorm, GELU, and analog regression, likely followed by
conversion-aware calibration or fine-tuning.
