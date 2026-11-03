"""Kiểm thử cấu hình tuning SVR mà không chạy GridSearch đầy đủ."""

import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from src.config import CV_FOLDS
from src.train_svr import SVR_PARAM_GRID, build_svr_search


def test_svr_search_uses_shared_pipeline_and_cv() -> None:
    search = build_svr_search(n_jobs=1)

    assert isinstance(search, GridSearchCV)
    assert isinstance(search.estimator, Pipeline)
    assert isinstance(search.estimator.named_steps["model"], SVR)
    assert search.cv == CV_FOLDS
    assert search.scoring == "neg_mean_absolute_error"


def test_svr_grid_covers_required_parameters() -> None:
    keys = set().union(*(grid.keys() for grid in SVR_PARAM_GRID))

    assert {
        "model__kernel",
        "model__C",
        "model__epsilon",
        "model__gamma",
    }.issubset(keys)


def test_explain_model_rejects_invalid_repeat_count() -> None:
    from src.explain_model import explain_model

    with pytest.raises(ValueError, match="n_repeats"):
        explain_model(n_repeats=0)
