"""Weitefeld metadata parsing and high-resolution tiling utilities.

The official ``data.txt`` labels describe crowd-sourced anomaly categories.
Class 3 means ``person`` in that format, but it is not a guarantee that the
marked object was a person. Police comments explicitly identify several such
findings as stumps, stones, bags, or no abnormality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import shlex
from typing import Any, Iterable


WEITEFELD_WIDTH = 8416
WEITEFELD_HEIGHT = 6032
WEITEFELD_PERSON_CLASS = 3

# Table 2 of Nathan et al., Scientific Data 13:747 (2026).
PRIORITY_ZONE_RANGES: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((2856, 3240),),
    2: ((5527, 5967),),
    3: ((9067, 9582),),
    4: ((12051, 12721),),
    5: ((15865, 16667),),
    6: ((19131, 19838),),
    7: ((23133, 23885),),
    8: ((26698, 27482),),
    9: ((30509, 31279),),
    10: ((33778, 34509),),
    11: ((40407, 41175),),
    12: ((44019, 44854),),
    13: ((47264, 48023),),
    14: ((56961, 57806),),
    15: ((60972, 61854),),
}


def _normalize_comment(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def person_verification_tier(volunteer_comment: str, police_comment: str) -> str:
    """Return a conservative interpretation tier without changing class 3.

    ``explicit_person_view`` means the text explicitly describes people or a
    cyclist in the aerial image. It does not establish physical identity.
    ``police_nonperson`` means the ground report names a non-person object.
    Remaining records are ``ambiguous_person_candidate``.
    """

    volunteer = _normalize_comment(volunteer_comment)
    police = _normalize_comment(police_comment)
    nonperson_terms = (
        "tree stump", "tree stumps", "tree trunk", "uprooted tree",
        "stone", "trash bag", "hunting stand", "feeding site",
        "fallen tree", "water surface no findings", "large tree stump",
    )
    if any(term in police for term in nonperson_terms):
        return "police_nonperson"
    explicit_terms = (
        "person standing", "standing person", "walking person", "two people",
        "3 people", "few people", "cyclist",
    )
    if any(term in volunteer for term in explicit_terms) or any(
        term in police for term in ("group of people", "cyclist")
    ):
        return "explicit_person_view"
    return "ambiguous_person_candidate"


@dataclass(frozen=True)
class WeitefeldAnnotation:
    annotation_id: int
    source_strip: int
    source_image_number: int
    source_x: int
    source_y: int
    category_id: int
    volunteer_comment: str
    image_strip: int
    image_number: int
    bbox_lower_left_x: int
    bbox_lower_left_y: int
    bbox_height: int
    bbox_width: int
    police_comment: str
    bbox_xyxy: tuple[float, float, float, float] | None
    bbox_valid: bool
    verification_tier: str | None
    track_id: str | None = None

    @property
    def image_key(self) -> tuple[int, int]:
        return self.image_strip, self.image_number

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["bbox_xyxy"] = list(self.bbox_xyxy) if self.bbox_xyxy is not None else None
        return result


def lower_left_bbox_to_xyxy(
    x: int | float,
    y: int | float,
    height: int | float,
    width: int | float,
    *,
    image_width: int = WEITEFELD_WIDTH,
    image_height: int = WEITEFELD_HEIGHT,
) -> tuple[float, float, float, float]:
    """Convert official ``x, y_lower, height, width`` to top-left XYXY.

    The official viewer performs ``top = y_lower - height``. Coordinates are
    clipped to the image extent and degenerate boxes are rejected.
    """

    x1 = min(max(float(x), 0.0), float(image_width))
    y1 = min(max(float(y) - float(height), 0.0), float(image_height))
    x2 = min(max(float(x) + float(width), 0.0), float(image_width))
    y2 = min(max(float(y), 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate Weitefeld box: {(x, y, height, width)}")
    return x1, y1, x2, y2


def parse_data_txt(path: Path) -> list[WeitefeldAnnotation]:
    annotations: list[WeitefeldAnnotation] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        tokens = shlex.split(line)
        if len(tokens) != 13:
            raise ValueError(f"{path}:{line_number}: expected 13 fields, got {len(tokens)}")
        values = [int(tokens[index]) for index in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11)]
        (
            source_strip, source_image, source_x, source_y, category_id,
            image_strip, image_number, box_x, box_y, box_h, box_w,
        ) = values
        try:
            bbox = lower_left_bbox_to_xyxy(box_x, box_y, box_h, box_w)
        except ValueError:
            # A few official projected labels lie fully beyond an image edge.
            # Keep their source record but exclude them from localization use.
            bbox = None
        tier = (
            person_verification_tier(tokens[5], tokens[12])
            if category_id == WEITEFELD_PERSON_CLASS else None
        )
        annotations.append(
            WeitefeldAnnotation(
                annotation_id=line_number,
                source_strip=source_strip,
                source_image_number=source_image,
                source_x=source_x,
                source_y=source_y,
                category_id=category_id,
                volunteer_comment=tokens[5],
                image_strip=image_strip,
                image_number=image_number,
                bbox_lower_left_x=box_x,
                bbox_lower_left_y=box_y,
                bbox_height=box_h,
                bbox_width=box_w,
                police_comment=tokens[12],
                bbox_xyxy=bbox,
                bbox_valid=bbox is not None,
                verification_tier=tier,
            )
        )
    return annotations


def deduplicate_annotations(
    annotations: Iterable[WeitefeldAnnotation],
) -> list[WeitefeldAnnotation]:
    """Drop exact repeated labels while preserving the first official row ID."""

    result: list[WeitefeldAnnotation] = []
    seen: set[tuple[Any, ...]] = set()
    for annotation in annotations:
        key = (
            annotation.image_key,
            annotation.category_id,
            annotation.bbox_xyxy,
            _normalize_comment(annotation.volunteer_comment),
            _normalize_comment(annotation.police_comment),
        )
        if key not in seen:
            seen.add(key)
            result.append(annotation)
    return result


def assign_strip_track_ids(
    annotations: Iterable[WeitefeldAnnotation],
    *,
    max_frame_gap: int = 3,
) -> list[WeitefeldAnnotation]:
    """Assign reproducible strip-local track proxies to projected labels.

    ``data.txt`` does not expose the paper's 405 physical finding IDs. These
    tracks group contiguous frames with equal category/comments within a strip;
    they must not be interpreted as globally unique physical identities.
    """

    grouped: dict[tuple[Any, ...], list[WeitefeldAnnotation]] = {}
    for annotation in annotations:
        key = (
            annotation.image_strip,
            annotation.category_id,
            _normalize_comment(annotation.volunteer_comment),
            _normalize_comment(annotation.police_comment),
        )
        grouped.setdefault(key, []).append(annotation)

    replacements: dict[int, str] = {}
    track_number = 0
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2], item[3])):
        ordered = sorted(grouped[key], key=lambda item: (item.image_number, item.annotation_id))
        previous_image: int | None = None
        for annotation in ordered:
            if previous_image is None or annotation.image_number - previous_image > max_frame_gap:
                track_number += 1
            replacements[annotation.annotation_id] = f"strip-track-{track_number:04d}"
            previous_image = annotation.image_number

    return [
        WeitefeldAnnotation(**{**annotation.to_dict(), "bbox_xyxy": annotation.bbox_xyxy,
                               "track_id": replacements[annotation.annotation_id]})
        for annotation in annotations
    ]


def is_priority_zone_image(strip: int, image_number: int) -> bool:
    return any(start <= image_number <= end for start, end in PRIORITY_ZONE_RANGES[strip])


def tile_origins(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Require tile_size > overlap >= 0")
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    origins = list(range(0, length - tile_size + 1, step))
    final = length - tile_size
    if origins[-1] != final:
        origins.append(final)
    return origins


def image_tiles(
    width: int,
    height: int,
    *,
    tile_size: int = 2048,
    overlap: int = 256,
) -> list[tuple[int, int, int, int]]:
    return [
        (x, y, min(x + tile_size, width), min(y + tile_size, height))
        for y in tile_origins(height, tile_size, overlap)
        for x in tile_origins(width, tile_size, overlap)
    ]

