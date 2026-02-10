from src.utils.config import load_config


def test_config_keys_present():
    cfg = load_config()
    expected = {
        "empty_threshold",
        "full_threshold",
        "decision_threshold",
        "horizon_shifts",
        "horizon_minutes",
        "train_fraction",
        "empty_threshold_candidates",
    }
    missing = expected - set(cfg.keys())
    assert not missing
