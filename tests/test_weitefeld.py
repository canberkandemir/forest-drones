from pathlib import Path

import pytest

from forestpersons.weitefeld import (
    assign_strip_track_ids,
    image_tiles,
    lower_left_bbox_to_xyxy,
    parse_data_txt,
    person_verification_tier,
)
from scripts.predict_weitefeld_pilot import cross_tile_nms, tile_owns_center


def test_lower_left_bbox_conversion_matches_official_viewer() -> None:
    assert lower_left_bbox_to_xyxy(3707, 2323, 37, 52) == (3707.0, 2286.0, 3759.0, 2323.0)


def test_parser_and_person_tiers(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text(
        '01 03143 1474 3134 3 "standing person" 01 03143 1459 3150 0040 0030 "No abnormalities, good geolocation"\n'
        '13 47493 3749 2319 3 "potential body" 13 47493 3707 2323 0037 0052 "Tree stump, good geolocation"\n',
        encoding="utf-8",
    )
    rows = assign_strip_track_ids(parse_data_txt(path))
    assert rows[0].verification_tier == "explicit_person_view"
    assert rows[1].verification_tier == "police_nonperson"
    assert rows[0].track_id != rows[1].track_id


def test_non_person_class_has_no_person_tier(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text('01 03000 100 200 2 "object" 01 03000 90 210 20 30 "barrel"\n')
    assert parse_data_txt(path)[0].verification_tier is None


def test_fully_outside_official_box_is_retained_but_invalid(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text('07 23553 7571 0000 3 "person" 07 23553 7571 0000 0020 0032 "NA"\n')
    row = parse_data_txt(path)[0]
    assert not row.bbox_valid
    assert row.bbox_xyxy is None


def test_image_tiles_cover_edges_without_duplicate_origins() -> None:
    tiles = image_tiles(8416, 6032, tile_size=2048, overlap=256)
    assert len(tiles) == 20
    assert len(set(tiles)) == len(tiles)
    assert min(tile[0] for tile in tiles) == 0
    assert min(tile[1] for tile in tiles) == 0
    assert max(tile[2] for tile in tiles) == 8416
    assert max(tile[3] for tile in tiles) == 6032


def test_invalid_tiling_rejected() -> None:
    with pytest.raises(ValueError):
        image_tiles(100, 100, tile_size=50, overlap=50)


def test_tile_ownership_uses_half_overlap_boundary() -> None:
    tile = (1792, 0, 3840, 2048)
    assert tile_owns_center([100, 100, 140, 140], tile, 8416, 6032, 256) is False
    assert tile_owns_center([130, 100, 170, 140], tile, 8416, 6032, 256) is True


def test_cross_tile_nms_keeps_higher_score() -> None:
    predictions = [
        {"bbox": [0, 0, 20, 20], "score": 0.9},
        {"bbox": [1, 1, 21, 21], "score": 0.7},
    ]
    kept = cross_tile_nms(predictions, 0.5)
    assert len(kept) == 1
    assert kept[0]["score"] == 0.9
