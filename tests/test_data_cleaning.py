"""Kiểm thử quy trình làm sạch dữ liệu."""

import pandas as pd
import pytest

from src.data_cleaning import clean_student_data, save_clean_data


def make_dirty_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Hours_Studied": [10, 10, 20],
            "Teacher_Quality": [" High ", "High", pd.NA],
            "Gender": ["Male", "Male", " Female "],
            "Exam_Score": [101, 101, 70],
        }
    )


def test_cleaning_normalizes_text_drops_duplicates_and_clips_target() -> None:
    dirty = make_dirty_data()

    cleaned, summary = clean_student_data(dirty)

    assert len(cleaned) == 2
    assert cleaned["Exam_Score"].max() == 100
    assert cleaned.loc[0, "Teacher_Quality"] == "High"
    assert cleaned.loc[1, "Gender"] == "Female"
    assert summary.duplicates_removed == 1
    assert summary.target_values_clipped == 1


def test_cleaning_preserves_missing_features_for_model_pipeline() -> None:
    cleaned, summary = clean_student_data(make_dirty_data())

    assert cleaned["Teacher_Quality"].isna().sum() == 1
    assert summary.missing_feature_values_preserved == 1


def test_cleaning_does_not_modify_original_dataframe() -> None:
    dirty = make_dirty_data()
    original = dirty.copy(deep=True)

    clean_student_data(dirty)

    pd.testing.assert_frame_equal(dirty, original)


def test_cleaning_rejects_invalid_target() -> None:
    dirty = make_dirty_data()
    dirty["Exam_Score"] = dirty["Exam_Score"].astype(object)
    dirty.loc[0, "Exam_Score"] = "invalid"

    with pytest.raises(ValueError, match="Exam_Score"):
        clean_student_data(dirty)


def test_save_clean_data_round_trip(tmp_path) -> None:
    cleaned, _ = clean_student_data(make_dirty_data())
    output_path = tmp_path / "cleaned.csv"

    saved_path = save_clean_data(cleaned, output_path)
    reloaded = pd.read_csv(saved_path)

    assert saved_path == output_path
    assert len(reloaded) == 2
    assert reloaded["Exam_Score"].max() == 100
