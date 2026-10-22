from __future__ import annotations

import torch

from forestpersons.dtss_experiment import (
    DTSSSyntheticClassifier,
    SyntheticExperimentConfig,
    make_synthetic_token_data,
    run_controlled_experiment,
    set_timestep_learning,
    timestep_masks,
)


def test_synthetic_token_data_is_deterministic_and_balanced() -> None:
    first = make_synthetic_token_data(16, tokens=4, embed_dim=8, seed=7)
    second = make_synthetic_token_data(16, tokens=4, embed_dim=8, seed=7)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.bincount(first[1]).tolist() == [8, 8]


def test_fixed_timestep_parameters_are_frozen_and_all_active() -> None:
    config = SyntheticExperimentConfig(embed_dim=8, num_heads=2, tokens=4)
    model = DTSSSyntheticClassifier(config)
    set_timestep_learning(model, False)
    masks = timestep_masks(model)
    assert len(masks) == 7
    assert all(not module.raw_timestep_parameters.requires_grad for module in masks.values())
    assert all(module().active_timesteps == 4 for module in masks.values())


def test_tiny_controlled_run_updates_only_learned_timestep_parameters() -> None:
    config = SyntheticExperimentConfig(
        train_samples=16,
        test_samples=8,
        tokens=4,
        embed_dim=8,
        num_heads=2,
        batch_size=8,
        epochs=2,
    )
    result = run_controlled_experiment(config, torch.device("cpu"))
    fixed, learned = result["arms"]
    assert fixed["name"] == "fixed_t4"
    assert learned["name"] == "learned_dtss"
    assert all(value == 0.0 for value in fixed["timestep_parameter_delta_norms"].values())
    assert all(value > 0.0 for value in learned["timestep_parameter_delta_norms"].values())
    assert 0.0 <= fixed["final"]["test_accuracy"] <= 1.0
    assert 0.0 <= learned["final"]["test_accuracy"] <= 1.0

