import numpy as np
from PIL import Image

from forestpersons.rx_anomaly import fit_rx_model, rgb_patch_features, rx_scores


def test_rgb_patch_features_match_patch_grid() -> None:
    image = Image.new("RGB", (32, 16), color=(128, 64, 32))

    features, info = rgb_patch_features(image, short_side=16)

    assert features.shape == (1, 2, 6)
    assert info.grid_height == 1
    assert info.grid_width == 2
    assert np.allclose(features[..., :3], np.array([128, 64, 32]) / 255.0, atol=1e-6)
    assert np.allclose(features[..., 3:], 0.0, atol=1e-6)


def test_rx_scores_are_low_near_background_and_high_for_outlier() -> None:
    background = np.array(
        [
            [0.1, 0.2, 0.3, 0.01, 0.01, 0.01],
            [0.11, 0.19, 0.31, 0.01, 0.02, 0.01],
            [0.09, 0.21, 0.29, 0.02, 0.01, 0.01],
            [0.10, 0.20, 0.30, 0.01, 0.01, 0.02],
        ],
        dtype=np.float32,
    )
    model = fit_rx_model(background, regularization=1e-2)
    features = np.array(
        [
            [[0.1, 0.2, 0.3, 0.01, 0.01, 0.01]],
            [[0.8, 0.8, 0.1, 0.2, 0.2, 0.2]],
        ],
        dtype=np.float32,
    )

    scores = rx_scores(features, model)

    assert scores.shape == (2, 1)
    assert scores[1, 0] > scores[0, 0]


def test_fit_rx_model_rejects_too_few_features() -> None:
    try:
        fit_rx_model(np.ones((1, 6), dtype=np.float32))
    except ValueError as exc:
        assert "At least two" in str(exc)
    else:
        raise AssertionError("fit_rx_model should reject a single feature")
