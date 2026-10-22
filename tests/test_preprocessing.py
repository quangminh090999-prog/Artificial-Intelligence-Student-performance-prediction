
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from src.preprocessing import (
    build_preprocessor,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def make_sample_data():
    """Create a small dataset for preprocessing tests."""
    return pd.DataFrame({
        "Hours_Studied": [5.0, None, 8.0],
        "Attendance": [80.0, 90.0, None],
        "Sleep_Hours": [7.0, 6.0, 8.0],
        "Previous_Scores": [70.0, 75.0, 85.0],
        "Tutoring_Sessions": [1.0, 0.0, 2.0],
        "Physical_Activity": [3.0, 4.0, 2.0],
        "Parental_Involvement": ["Low", "High", None],
        "Access_to_Resources": ["High", "Medium", "Low"],
        "Extracurricular_Activities": ["Yes", "No", "Yes"],
        "Motivation_Level": ["Low", "Medium", "High"],
        "Internet_Access": ["Yes", "Yes", "No"],
        "Family_Income": ["Low", "Medium", "High"],
        "Teacher_Quality": ["Medium", "High", "Low"],
        "School_Type": ["Public", "Private", "Public"],
        "Peer_Influence": ["Positive", "Neutral", "Negative"],
        "Learning_Disabilities": ["No", "No", "Yes"],
        "Parental_Education_Level": [
            "High School", "College", "High School"
        ],
        "Distance_from_Home": ["Near", "Moderate", "Near"],
        "Gender": ["Male", "Female", "Male"],
    })


def test_feature_columns_are_separated():
    assert len(NUMERIC_FEATURES) == 6
    assert len(CATEGORICAL_FEATURES) == 13
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)


def test_preprocessor_handles_missing_values():
    data = make_sample_data()
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(data)

    assert transformed.shape[0] == len(data)
    assert not pd.isna(transformed).any()


def test_preprocessor_handles_unknown_category():
    train_data = make_sample_data()
    preprocessor = build_preprocessor()
    preprocessor.fit(train_data)

    new_data = train_data.iloc[[0]].copy()
    new_data.loc[:, "Parental_Involvement"] = "NewCategory"

    transformed = preprocessor.transform(new_data)

    assert transformed.shape[0] == 1


def test_preprocessor_is_pipeline_based():
    preprocessor = build_preprocessor()

    assert hasattr(preprocessor, "fit")
    assert hasattr(preprocessor, "transform")
