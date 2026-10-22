from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from forestpersons.deimv2_training import (
    CHECKPOINT_ARCHITECTURE,
    PilotFlatCosineScheduler,
    normalized_person_boxes,
    official_parameter_groups,
    samples_from_manifest,
    validate_training_checkpoint,
)


def _load_selector():
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / "select_deimv2_calibration_threshold.py"
    spec = importlib.util.spec_from_file_location("select_deimv2_calibration_threshold", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalized_person_boxes_clip_and_convert_xywh() -> None:
    boxes = normalized_person_boxes(
        [
            {"bbox": [-10.0, 10.0, 30.0, 20.0]},
            {"bbox": [90.0, 40.0, 20.0, 20.0]},
            {"bbox": [20.0, 20.0, -2.0, 5.0]},
        ],
        width=100,
        height=50,
    )
    assert boxes.shape == (2, 4)
    assert torch.allclose(boxes[0], torch.tensor([0.10, 0.40, 0.20, 0.40]))
    assert torch.allclose(boxes[1], torch.tensor([0.95, 0.90, 0.10, 0.20]))


def test_manifest_resolution_uses_image_id_with_duplicate_filenames(tmp_path: Path) -> None:
    (tmp_path / "annotations").mkdir()
    (tmp_path / "images").mkdir()
    Image.new("RGB", (20, 10)).save(tmp_path / "images" / "duplicate.jpg")
    coco = {
        "images": [
            {"id": 11, "file_name": "duplicate.jpg", "width": 20, "height": 10},
            {"id": 12, "file_name": "duplicate.jpg", "width": 20, "height": 10},
        ],
        "annotations": [
            {"id": 101, "image_id": 11, "category_id": 1, "bbox": [1, 1, 2, 2]},
            {"id": 102, "image_id": 12, "category_id": 1, "bbox": [5, 2, 3, 4]},
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    (tmp_path / "annotations" / "train.json").write_text(json.dumps(coco), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "images": [
                    {"image_id": 11, "file_name": "duplicate.jpg"},
                    {"image_id": 12, "file_name": "duplicate.jpg"},
                ]
            }
        ),
        encoding="utf-8",
    )

    samples = samples_from_manifest(tmp_path, manifest, split="train")
    assert [sample.image_id for sample in samples] == [11, 12]
    assert [sample.annotations[0]["id"] for sample in samples] == [101, 102]


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.dinov3 = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.LayerNorm(2),
        )
        self.encoder = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        self.decoder = torch.nn.Linear(2, 1)


def test_official_parameter_groups_cover_each_parameter_once() -> None:
    model = _TinyModel()
    groups = official_parameter_groups(
        model,
        backbone_lr=2.5e-5,
        detector_lr=5e-4,
        weight_decay=1e-4,
    )
    grouped = [parameter for group in groups for parameter in group["params"]]
    assert len(grouped) == len(list(model.parameters()))
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    settings = {
        group["group_name"]: (group["lr"], group["weight_decay"])
        for group in groups
    }
    assert settings == {
        "backbone_decay": (2.5e-5, 1e-4),
        "backbone_no_decay": (2.5e-5, 0.0),
        "detector_decay": (5e-4, 1e-4),
        "detector_no_decay": (5e-4, 0.0),
    }


def test_pilot_flat_cosine_schedule_and_resume() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 1e-3, "initial_lr": 1e-3}])
    scheduler = PilotFlatCosineScheduler(
        optimizer,
        total_steps=200,
        warmup_steps=20,
        flat_steps=100,
        min_lr_ratio=0.5,
    )
    assert scheduler.factor(1) == pytest.approx(0.0025)
    assert scheduler.factor(20) == pytest.approx(1.0)
    assert scheduler.factor(100) == pytest.approx(1.0)
    assert scheduler.factor(200) == pytest.approx(0.5)
    scheduler.step(75)
    state = scheduler.state_dict()
    restored = PilotFlatCosineScheduler(
        optimizer,
        total_steps=200,
        warmup_steps=20,
        flat_steps=100,
        min_lr_ratio=0.5,
    )
    restored.load_state_dict(state)
    assert restored.current_step == 75
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_training_checkpoint_validation() -> None:
    valid = {
        "architecture": CHECKPOINT_ARCHITECTURE,
        "model_state": {},
        "training_config": {},
        "class_mapping": "person -> 0",
    }
    validate_training_checkpoint(valid)
    with pytest.raises(ValueError, match="architecture"):
        validate_training_checkpoint({**valid, "architecture": "wrong"})


def test_calibration_threshold_selection_rejects_validation_split() -> None:
    selector = _load_selector()
    summary = {
        "settings": {"split": "train", "manifest": "calibration.json"},
        "custom_sweep": {
            "matched_false_positive_budgets": {
                "0.425": {"confidence_threshold": 0.31, "false_positives_per_image": 0.42}
            }
        },
    }
    selected = selector.select_threshold(summary, 0.425)
    assert selected["confidence_threshold"] == pytest.approx(0.31)
    summary["settings"]["split"] = "val"
    with pytest.raises(ValueError, match="training-derived"):
        selector.select_threshold(summary, 0.425)

