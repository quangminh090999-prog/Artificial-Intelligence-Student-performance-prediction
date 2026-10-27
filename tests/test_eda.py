"""Kiểm thử tự động cho module phân tích khám phá dữ liệu (EDA) - Thành viên 2."""

from pathlib import Path
import pandas as pd
import pytest

from src.eda import (
    load_dataset,
    generate_data_summary,
    get_proposed_input_rules,
)


def test_load_dataset_returns_valid_shape():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 6000
    assert df.shape[1] == 20
    assert "Exam_Score" in df.columns


def test_generate_data_summary_structure():
    df = load_dataset()
    summary = generate_data_summary(df)
    assert len(summary) == 20
    assert "Thuộc_tính" in summary.columns
    assert "Tỷ_lệ_thiếu_%" in summary.columns
    assert "Số_giá_trị_duy_nhất" in summary.columns


def test_proposed_input_rules():
    df = load_dataset()
    rules = get_proposed_input_rules(df)
    assert "numeric_ranges" in rules
    assert "categorical_allowed" in rules
    assert len(rules["numeric_ranges"]) == 6
    assert len(rules["categorical_allowed"]) == 13
    assert rules["target_anomaly_note"]["anomaly_value"] == 101
