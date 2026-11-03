"""Kiểm thử các thành phần quan trọng của bước chọn đặc trưng."""

import numpy as np
import pytest

from src.feature_selection import (
    mutual_info_with_random_state,
    run_feature_selection,
)


def test_mutual_information_is_reproducible() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4))
    y = 2 * X[:, 0] + rng.normal(scale=0.1, size=100)

    first_scores = mutual_info_with_random_state(X, y)
    second_scores = mutual_info_with_random_state(X, y)

    np.testing.assert_allclose(first_scores, second_scores)


def test_feature_selection_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        run_feature_selection(top_k=0)
