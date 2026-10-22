from scripts.prepare_yolo_forestpersons import coco_bbox_to_yolo


def test_coco_bbox_to_yolo_normalizes_center_box() -> None:
    converted = coco_bbox_to_yolo([10, 20, 30, 40], image_width=100, image_height=200)

    assert converted == (0.25, 0.2, 0.3, 0.2)


def test_coco_bbox_to_yolo_clips_to_image_bounds() -> None:
    converted = coco_bbox_to_yolo([-10, 50, 30, 80], image_width=100, image_height=100)

    assert converted == (0.1, 0.75, 0.2, 0.5)


def test_coco_bbox_to_yolo_drops_empty_after_clipping() -> None:
    assert coco_bbox_to_yolo([120, 20, 10, 10], image_width=100, image_height=100) is None
