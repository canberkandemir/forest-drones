from types import SimpleNamespace

from scripts.benchmark_dinov3_vitl_feasibility import MODELS, model_card_estimate


def test_vitl_model_ids_are_architecture_matched() -> None:
    assert MODELS["web_vitl16_lvd1689m"] == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert MODELS["sat_vitl16_sat493m"] == "facebook/dinov3-vitl16-pretrain-sat493m"


def test_vitl_parameter_estimates_use_hidden_size_and_layers() -> None:
    config = SimpleNamespace(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        patch_size=16,
    )

    summary = model_card_estimate(config)

    assert summary["hidden_size"] == 1024
    assert summary["num_hidden_layers"] == 24
    assert summary["rank4_qv_lora_parameter_estimate"] == 393216
    assert summary["fcos_head_parameters"] == 627046
