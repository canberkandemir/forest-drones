from scripts.review_weitefeld_annotations import provisional_review, target_size_bin


def test_review_does_not_auto_confirm_people() -> None:
    status, _ = provisional_review("Group of people- excluded - not checked")
    assert status == "uncertain"
    status, _ = provisional_review("Cyclist- excluded - not checked")
    assert status == "uncertain"


def test_review_prepopulates_only_explicit_nonperson_evidence() -> None:
    assert provisional_review("Tree stump- good geolocation")[0] == "non_person"
    assert provisional_review("Good geolocation- no abnormalities")[0] == "non_person"
    assert provisional_review("NA")[0] == "uncertain"
    assert provisional_review("Possibly interesting")[0] == "not_reviewed"


def test_original_resolution_size_bins_are_fixed() -> None:
    assert target_size_bin(31 * 31) == "small_lt_32sq"
    assert target_size_bin(32 * 32) == "medium_32_96sq"
    assert target_size_bin(96 * 96) == "large_ge_96sq"
