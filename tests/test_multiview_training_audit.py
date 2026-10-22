from __future__ import annotations

from scripts.audit_forestpersons_train5000_multiview import disjoint_groups, frame_number, identity


def group(sequence: str, reference: int, earlier: int, later: int) -> dict:
    def view(frame: int) -> dict:
        file_name = f"{sequence}/{sequence}-{frame:04d}.jpg"
        return {"frame": frame, "file_name": file_name, "identity": identity(file_name)}

    return {
        "sequence": sequence,
        "reference": view(reference),
        "sources": [view(earlier), view(later)],
    }


def test_identity_is_split_qualified_and_frame_parser_is_filename_based() -> None:
    file_name = "001_FPV_Person_summer_forest/001_FPV_Person_summer_forest-0042.jpg"
    assert identity(file_name) == f"train::{file_name}"
    assert frame_number(file_name) == 42


def test_disjoint_group_packing_never_reuses_a_view() -> None:
    groups = [
        group("sequence_a", 5, 2, 8),
        group("sequence_a", 8, 5, 11),
        group("sequence_a", 14, 11, 17),
        group("sequence_b", 5, 2, 8),
    ]
    selected = disjoint_groups(groups, seed=7, trials=20)
    identities = [
        view["identity"]
        for item in selected
        for view in [item["reference"], *item["sources"]]
    ]
    assert len(identities) == len(set(identities))
    assert {item["sequence"] for item in selected} == {"sequence_a", "sequence_b"}


def test_disjoint_group_packing_is_deterministic() -> None:
    groups = [group("sequence_a", frame, frame - 3, frame + 3) for frame in (5, 8, 11, 14)]
    first = disjoint_groups(groups, seed=20261017, trials=20)
    second = disjoint_groups(groups, seed=20261017, trials=20)
    assert first == second
