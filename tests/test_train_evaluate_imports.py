"""Kiểm tra tối thiểu để phát hiện lỗi cú pháp và conflict trong module chính."""

import importlib

import pytest


@pytest.mark.parametrize("module_name", ["src.train", "src.evaluate"])
def test_training_modules_import_without_errors(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None


def test_calculate_metrics_returns_expected_keys() -> None:
    from src.train import calculate_metrics

    metrics = calculate_metrics([60.0, 80.0], [61.0, 79.0])

    assert set(metrics) == {"MAE", "RMSE", "R2"}
    assert metrics["MAE"] == pytest.approx(1.0)
