"""Kiểm thử phần tuning và đánh giá mô hình nâng cao của Người 4."""

from pathlib import Path

import numpy as np
import pandas as pd

import src.train_advanced as advanced
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _sample_data(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {column: rng.normal(10, 2, rows) for column in NUMERIC_FEATURES}
    )
    categories = {
        "Parental_Involvement": ["Low", "Medium", "High"],
        "Access_to_Resources": ["Low", "Medium", "High"],
        "Extracurricular_Activities": ["No", "Yes"],
        "Motivation_Level": ["Low", "Medium", "High"],
        "Internet_Access": ["No", "Yes"],
        "Family_Income": ["Low", "Medium", "High"],
        "Teacher_Quality": ["Low", "Medium", "High"],
        "School_Type": ["Public", "Private"],
        "Peer_Influence": ["Negative", "Neutral", "Positive"],
        "Learning_Disabilities": ["No", "Yes"],
        "Parental_Education_Level": ["High School", "College", "Postgraduate"],
        "Distance_from_Home": ["Near", "Moderate", "Far"],
        "Gender": ["Female", "Male"],
    }
    for column in CATEGORICAL_FEATURES:
        frame[column] = rng.choice(categories[column], rows)
    frame[advanced.TARGET_COLUMN] = (
        55
        + 1.5 * frame["Hours_Studied"]
        + 0.5 * frame["Attendance"]
        + rng.normal(0, 1, rows)
    ).clip(0, 100)
    return frame


def test_choose_final_model_selects_lowest_mae(tmp_path, monkeypatch):
    monkeypatch.setattr(advanced, "RESULTS_PATH", tmp_path)
    pd.DataFrame(
        [
            {"model": "SVR_Tuned", "MAE": 0.40, "RMSE": 1.80, "R2": 0.77},
            {"model": "Ridge_Tuned", "MAE": 0.45, "RMSE": 1.81, "R2": 0.76},
        ]
    ).to_csv(tmp_path / "all_models_summary.csv", index=False)
    advanced_metrics = pd.DataFrame(
        [
            {
                "model": "RandomForest_Tuned",
                "MAE": 0.90,
                "RMSE": 2.0,
                "R2": 0.70,
            }
        ]
    )

    result = advanced.choose_final_model(advanced_metrics)

    assert result.iloc[0]["model"] == "SVR_Tuned"
    assert bool(result.iloc[0]["selected"])
    assert (tmp_path / "final_model_selection.csv").exists()


def test_train_advanced_models_exports_expected_artifacts(tmp_path, monkeypatch):
    results_path = tmp_path / "results"
    figures_path = tmp_path / "figures"
    models_path = tmp_path / "models"
    monkeypatch.setattr(advanced, "RESULTS_PATH", results_path)
    monkeypatch.setattr(advanced, "FIGURES_PATH", figures_path)
    monkeypatch.setattr(advanced, "MODELS_PATH", models_path)
    monkeypatch.setattr(advanced, "CV_FOLDS", 2)
    monkeypatch.setattr(advanced, "load_data", lambda: _sample_data())
    monkeypatch.setattr(
        advanced,
        "RF_PARAM_DISTRIBUTIONS",
        {
            "model__n_estimators": [10],
            "model__max_depth": [4],
            "model__min_samples_split": [2],
            "model__min_samples_leaf": [1],
            "model__max_features": [1.0],
        },
    )
    monkeypatch.setattr(
        advanced,
        "GB_PARAM_DISTRIBUTIONS",
        {
            "model__n_estimators": [10],
            "model__learning_rate": [0.1],
            "model__max_depth": [2],
            "model__min_samples_leaf": [2],
            "model__subsample": [1.0],
        },
    )
    monkeypatch.setattr(advanced, "_write_report", lambda *_: None)

    metrics = advanced.train_advanced_models(n_iter=1)

    assert set(metrics["model"]) == {
        "RandomForest_Tuned",
        "GradientBoosting_Tuned",
    }
    expected = [
        results_path / "advanced_model_metrics.csv",
        results_path / "advanced_model_cv_results.csv",
        results_path / "advanced_overfitting_analysis.csv",
        results_path / "final_model_selection.csv",
        figures_path / "10_advanced_predicted_vs_actual.png",
        figures_path / "11_advanced_residual_distribution.png",
        figures_path / "12_advanced_overfitting_gap.png",
        figures_path / "13_all_models_comparison.png",
        models_path / "random_forest_tuned.joblib",
        models_path / "gradient_boosting_tuned.joblib",
    ]
    assert all(Path(path).exists() for path in expected)
