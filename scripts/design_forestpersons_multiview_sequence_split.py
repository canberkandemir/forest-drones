"""Design a visibility-balanced sequence split from frozen eligible groups."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LEVELS = (20, 40, 70, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/train.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-sequences", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261017)
    return parser.parse_args()


def summarize(groups: list[dict[str, Any]], by_image: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    visibility = Counter()
    people = 0
    for group in groups:
        for annotation in by_image[int(group["reference"]["image_id"])]:
            people += 1
            visibility[int(annotation["attributes"]["visibility_level"])] += 1
    return {
        "groups": len(groups),
        "sequences": len({group["sequence"] for group in groups}),
        "people": people,
        "visibility_counts": {str(level): visibility[level] for level in LEVELS},
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    groups = json.loads(args.groups.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))["annotations"]
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        by_image[int(annotation["image_id"])].append(annotation)
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_sequence[group["sequence"]].append(group)
    sequences = sorted(by_sequence)
    if args.validation_sequences >= len(sequences):
        raise ValueError("validation sequence count must be smaller than the available sequence count")

    total = summarize(groups, by_image)
    fraction = args.validation_sequences / len(sequences)
    target = {level: total["visibility_counts"][str(level)] * fraction for level in LEVELS}
    rng = random.Random(args.seed)
    best_score = float("inf")
    best_validation: set[str] = set()
    for _ in range(args.trials):
        candidate = set(rng.sample(sequences, args.validation_sequences))
        counts = Counter()
        group_count = 0
        for sequence in candidate:
            group_count += len(by_sequence[sequence])
            for group in by_sequence[sequence]:
                for annotation in by_image[int(group["reference"]["image_id"])]:
                    counts[int(annotation["attributes"]["visibility_level"])] += 1
        distribution_error = sum(((counts[level] - target[level]) / max(target[level], 1.0)) ** 2 for level in LEVELS)
        group_target = len(groups) * fraction
        score = distribution_error + 0.15 * ((group_count - group_target) / max(group_target, 1.0)) ** 2
        if score < best_score:
            best_score = score
            best_validation = candidate

    validation_groups = [group for group in groups if group["sequence"] in best_validation]
    training_groups = [group for group in groups if group["sequence"] not in best_validation]
    train_identities = {view["identity"] for group in training_groups for view in [group["reference"], *group["sources"]]}
    validation_identities = {view["identity"] for group in validation_groups for view in [group["reference"], *group["sources"]]}
    result = {
        "seed": args.seed,
        "trials": args.trials,
        "eligibility_frozen_before_annotation_stratification": True,
        "selection_uses_model_predictions": False,
        "stratification_uses_reference_visibility_annotations": True,
        "training_sequences": sorted(set(sequences) - best_validation),
        "validation_sequences": sorted(best_validation),
        "training_groups": training_groups,
        "validation_groups": validation_groups,
        "training_summary": summarize(training_groups, by_image),
        "validation_summary": summarize(validation_groups, by_image),
        "sequence_overlap": 0,
        "split_qualified_identity_overlap": len(train_identities & validation_identities),
        "physical_scene_warning": "Sequence-disjoint is verified; ForestPersons provides no global site identity for site-disjoint verification.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"training_groups", "validation_groups", "training_sequences", "validation_sequences"}}, indent=2))


if __name__ == "__main__":
    main()
