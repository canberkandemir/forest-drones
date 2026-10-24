from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.evaluate_dinov3_mvr_label_efficiency import (
    checkpoint_initialization,
    checkpoint_specs,
)
from scripts.prepare_dinov3_mvr_label_efficiency import (
    VISIBILITIES,
    build_nested_order,
    validate_no_leakage,
)
from scripts.train_dinov3_mvr_fcos import validate_sample_count


def item(visibility: int, index: int) -> dict:
    folder = f"train_seq_{index % 4}"
    return {
        "image_id": visibility * 100 + index,
        "file_name": f"{folder}/{visibility}_{index}.jpg",
        "folder": folder,
        "target_visibility": visibility,
        "annotation_count": 1,
        "visibility_counts": {str(visibility): 1},
        "season": "summer",
        "camera": "camera",
        "place": "forest",
        "weather": "clear",
    }


def test_nested_order_is_deterministic_balanced_and_preserves_pool() -> None:
    items = [item(visibility, index) for visibility in VISIBILITIES for index in range(8)]
    first = build_nested_order(items, seed=23)
    second = build_nested_order(list(reversed(items)), seed=23)
    assert [entry["file_name"] for entry in first] == [entry["file_name"] for entry in second]
    assert {entry["file_name"] for entry in first} == {entry["file_name"] for entry in items}
    for size in (4, 8, 16, 32):
        counts = {visibility: 0 for visibility in VISIBILITIES}
        for entry in first[:size]:
            counts[entry["target_visibility"]] += 1
        assert set(counts.values()) == {size // 4}


def test_leakage_uses_filenames_and_sequences_not_numeric_ids() -> None:
    train = [item(20, 1)]
    val = [
        {
            **item(40, 1),
            "image_id": train[0]["image_id"],
            "folder": "val_seq",
            "file_name": "val_seq/frame.jpg",
        }
    ]
    result = validate_no_leakage(train, val)
    assert result["filename_overlap"] == 0
    assert result["sequence_folder_overlap"] == 0
    assert result["numeric_image_id_overlap_ignored"] == 1

    val[0]["folder"] = train[0]["folder"]
    with pytest.raises(ValueError, match="Train/val leakage"):
        validate_no_leakage(train, val)


def test_training_sample_count_is_explicit() -> None:
    validate_sample_count(400, 400)
    with pytest.raises(ValueError, match="Expected 1000 training samples"):
        validate_sample_count(400, 1000)


def test_checkpoint_specs_are_sorted_and_reject_duplicates() -> None:
    assert checkpoint_specs(["5000=/b.pt", "100=/a.pt"]) == [
        (100, Path("/a.pt")),
        (5000, Path("/b.pt")),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        checkpoint_specs(["100=/a.pt", "100=/b.pt"])


def test_checkpoint_metadata_guards_subset_size_and_initialization(tmp_path) -> None:
    path = tmp_path / "model.pt"
    torch.save(
        {
            "arm": "mvr",
            "training_samples": 400,
            "fcos_initial_sha256": "matched_fcos",
            "model_initial_sha256": "matched_model",
        },
        path,
    )
    assert checkpoint_initialization(path, 400) == ("matched_fcos", "matched_model")
    with pytest.raises(ValueError, match="expected 1000"):
        checkpoint_initialization(path, 1000)
