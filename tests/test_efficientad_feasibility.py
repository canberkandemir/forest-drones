from scripts.analyze_efficientad_feasibility import (
    annotation_pose,
    annotation_visibility,
    scientific_decision,
    summarize_manifest,
    summarize_split,
    SplitSummary,
)


def test_annotation_fields_are_read_from_attributes() -> None:
    annotation = {
        "attributes": {
            "visibility_level": 20,
            "pose": "lying",
        }
    }

    assert annotation_visibility(annotation) == "20"
    assert annotation_pose(annotation) == "lying"


def test_summarize_manifest_counts_empty_and_positive_images() -> None:
    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg"},
            {"id": 2, "file_name": "b.jpg"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "bbox": [0, 0, 10, 10],
                "attributes": {"visibility_level": 40, "pose": "standing"},
            }
        ],
    }

    summary = summarize_manifest(coco, [{"file_name": "a.jpg"}, {"file_name": "b.jpg"}])

    assert summary["images"] == 2
    assert summary["annotations"] == 1
    assert summary["annotation_free_images"] == 1
    assert summary["visibility_counts"] == {"40": 1}
    assert summary["pose_counts"] == {"standing": 1}


def test_efficientad_decision_blocks_when_train5000_has_no_normal_images() -> None:
    official_train = SplitSummary(
        split="train",
        images=1000,
        annotations=2000,
        annotation_free_images=13,
        annotation_free_examples=[],
        visibility_counts={},
        pose_counts={},
    )
    train5000 = {"annotation_free_images": 0}

    decision = scientific_decision(train5000, official_train)

    assert decision["run_efficientad_pilot"] is False
    assert decision["decision"] == "do_not_train"
    assert "zero annotation-free images" in decision["reason"]


def test_summarize_split_reports_annotation_free_examples() -> None:
    coco = {
        "images": [
            {"id": 1, "file_name": "positive.jpg"},
            {"id": 2, "file_name": "empty.jpg"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "bbox": [0, 0, 10, 10],
                "attributes": {"visibility_level": 100, "pose": "sitting"},
            }
        ],
    }

    summary = summarize_split(coco, split="train")

    assert summary.images == 2
    assert summary.annotations == 1
    assert summary.annotation_free_images == 1
    assert summary.annotation_free_examples == ["empty.jpg"]
