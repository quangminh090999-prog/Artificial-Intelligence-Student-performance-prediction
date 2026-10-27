"""Module Phân tích và Khám phá Dữ liệu (EDA) - Thành viên 2 phụ trách.

Nhiệm vụ:
- Thống kê kích thước, kiểu dữ liệu, số lượng giá trị duy nhất và tỷ lệ thiếu.
- Phân tích phân bố biến mục tiêu Exam_Score và điểm bất thường (101).
- Tính tương quan các biến số và trực quan hóa (Heatmap).
- Trực quan hóa dữ liệu thiếu, phân bố, boxplot ngoại lệ, điểm theo nhóm phân loại.
- Tự động xuất biểu đồ vào reports/figures và bảng tóm tắt vào reports/data_summary.csv.
- Đề xuất miền giá trị hợp lệ cho các thành viên khác (Người 1 và Người 5).
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import json
import sys

# Đảm bảo in tiếng Việt chuẩn trên mọi console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Thiết lập phong cách đồ thị chuyên nghiệp
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.autolayout": True,
})

# Đường dẫn mặc định
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "StudentPerformanceFactors.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Đọc dữ liệu gốc từ thư mục raw."""
    if not csv_path.exists():
        # Fallback thử tìm trong thư mục hiện tại
        fallback_path = Path("data/raw/StudentPerformanceFactors.csv")
        if fallback_path.exists():
            csv_path = fallback_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {csv_path}")
    return pd.read_csv(csv_path)


def generate_data_summary(df: pd.DataFrame, output_path: Path = None) -> pd.DataFrame:
    """Thống kê kích thước, kiểu dữ liệu, số lượng unique, tỷ lệ missing và miền giá trị."""
    summary_rows = []
    total_rows = len(df)

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isnull().sum())
        missing_pct = round((missing_count / total_rows) * 100, 2)
        unique_count = int(series.nunique(dropna=True))
        dtype = str(series.dtype)

        if pd.api.types.is_numeric_dtype(series):
            valid_series = series.dropna()
            min_val = round(float(valid_series.min()), 2)
            max_val = round(float(valid_series.max()), 2)
            mean_val = round(float(valid_series.mean()), 2)
            median_val = round(float(valid_series.median()), 2)
            std_val = round(float(valid_series.std()), 2)
            val_range = f"[{min_val}, {max_val}]"
        else:
            unique_vals = series.dropna().unique().tolist()
            min_val, max_val = None, None
            mean_val, median_val, std_val = None, None, None
            val_range = ", ".join(str(v) for v in unique_vals[:5])
            if unique_count > 5:
                val_range += f", ... ({unique_count} nhóm)"

        summary_rows.append({
            "Thuộc_tính": col,
            "Kiểu_dữ_liệu": dtype,
            "Số_lượng_thiếu": missing_count,
            "Tỷ_lệ_thiếu_%": missing_pct,
            "Số_giá_trị_duy_nhất": unique_count,
            "Miền_giá_trị_hoặc_Nhóm": val_range,
            "Min": min_val,
            "Max": max_val,
            "Mean": mean_val,
            "Median": median_val,
            "Std": std_val,
        })

    summary_df = pd.DataFrame(summary_rows)

    if output_path is None:
        output_path = REPORTS_DIR / "data_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"-> Đã lưu bảng tóm tắt dữ liệu: {output_path}")
    return summary_df


def plot_target_distribution(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Biểu đồ 1: Phân bố biến mục tiêu Exam_Score và phân tích điểm bất thường."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    exam_scores = df["Exam_Score"].dropna()
    mean_val = exam_scores.mean()
    median_val = exam_scores.median()

    # Vẽ histogram + KDE
    sns.histplot(exam_scores, bins=35, kde=True, color="#1f77b4", edgecolor="black", ax=ax)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Trung bình (Mean): {mean_val:.2f}")
    ax.axvline(median_val, color="green", linestyle="-.", linewidth=2, label=f"Trung vị (Median): {median_val:.2f}")

    # Đánh dấu điểm bất thường 101 nếu có
    outlier_101 = df[df["Exam_Score"] == 101]
    if not outlier_101.empty:
        ax.annotate(
            "Điểm bất thường: 101\n(Vượt thang điểm 100)",
            xy=(101, 5),
            xytext=(92, 180),
            arrowprops=dict(facecolor="darkred", shrink=0.05, width=1.5, headwidth=8),
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffcccc", ec="darkred", lw=1.5),
            fontweight="bold",
            color="darkred",
        )

    ax.set_title("Phân bố Điểm thi của Sinh viên (Exam Score)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Điểm thi (Thang điểm 0 - 100)", fontsize=12)
    ax.set_ylabel("Số lượng sinh viên (Tần số)", fontsize=12)
    ax.legend(loc="upper left")

    save_path = output_dir / "01_exam_score_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Đã lưu biểu đồ: {save_path.name}")
    return save_path


def plot_missing_values(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Biểu đồ 2: Trực quan hóa tỷ lệ dữ liệu thiếu của các thuộc tính."""
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"Missing_Count": missing, "Percentage": missing_pct})
    missing_df = missing_df[missing_df["Missing_Count"] > 0].sort_values("Percentage", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not missing_df.empty:
        bars = ax.bar(missing_df.index, missing_df["Percentage"], color="#e74c3c", edgecolor="black", width=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}% ({int(height*len(df)/100)} bản ghi)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax.set_ylim(0, max(missing_df["Percentage"]) * 1.3)
    else:
        ax.text(0.5, 0.5, "Không có giá trị thiếu trong tập dữ liệu!", ha="center", va="center", fontsize=12)

    ax.set_title("Tỷ lệ Dữ liệu Thiếu theo Thuộc tính (Missing Values %)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Tên thuộc tính bị khuyết", fontsize=12)
    ax.set_ylabel("Tỷ lệ khuyết (%)", fontsize=12)
    plt.xticks(rotation=15, ha="right")

    save_path = output_dir / "02_missing_values_summary.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Đã lưu biểu đồ: {save_path.name}")
    return save_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Biểu đồ 3: Heatmap ma trận tương quan Pearson giữa các biến số và Exam_Score."""
    output_dir.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[num_cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-0.2,
        center=0,
        annot=True,
        fmt=".3f",
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Hệ số tương quan Pearson (r)"},
        ax=ax,
    )

    ax.set_title(
        "Ma trận Tương quan Tuyến tính Giữa các Biến Số\n(Lưu ý: Tương quan không đồng nghĩa với quan hệ nhân quả)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    save_path = output_dir / "03_correlation_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Đã lưu biểu đồ: {save_path.name}")
    return save_path


def plot_numeric_boxplots(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Biểu đồ 4: Boxplot kiểm tra ngoại lệ (outliers) cho các biến định lượng."""
    output_dir.mkdir(parents=True, exist_ok=True)
    num_features = [
        "Hours_Studied",
        "Attendance",
        "Sleep_Hours",
        "Previous_Scores",
        "Tutoring_Sessions",
        "Physical_Activity",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, col in enumerate(num_features):
        ax = axes[idx]
        sns.boxplot(y=df[col].dropna(), color="#3498db", ax=ax, width=0.4, flierprops=dict(marker="o", color="red", alpha=0.6))
        ax.set_title(f"Phân bố & Ngoại lệ: {col}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Giá trị", fontsize=10)

    plt.suptitle("Boxplot Kiểm tra Ngoại lệ (Outlier Detection) các Biến Số Học tập", fontsize=15, fontweight="bold", y=1.02)
    save_path = output_dir / "04_numeric_features_boxplots.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Đã lưu biểu đồ: {save_path.name}")
    return save_path


def plot_key_factors_vs_score(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Biểu đồ 5: Tác động của Thời gian học và Chuyên cần đến Điểm thi (Scatter + Regplot)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Hours_Studied vs Exam_Score
    sns.regplot(
        data=df,
        x="Hours_Studied",
        y="Exam_Score",
        ax=ax1,
        color="#2980b9",
        scatter_kws={"alpha": 0.25, "s": 20},
        line_kws={"color": "darkred", "linewidth": 2.5, "label": "Đường xu hướng (Trendline)"},
    )
    corr_hours = df["Hours_Studied"].corr(df["Exam_Score"])
    ax1.set_title(f"Giờ học (Hours Studied) vs Điểm thi (r = {corr_hours:.3f})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Số giờ học mỗi tuần (giờ)", fontsize=11)
    ax1.set_ylabel("Điểm thi (Exam Score)", fontsize=11)
    ax1.legend()

    # Attendance vs Exam_Score
    sns.regplot(
        data=df,
        x="Attendance",
        y="Exam_Score",
        ax=ax2,
        color="#27ae60",
        scatter_kws={"alpha": 0.25, "s": 20},
        line_kws={"color": "darkred", "linewidth": 2.5, "label": "Đường xu hướng (Trendline)"},
    )
    corr_att = df["Attendance"].corr(df["Exam_Score"])
    ax2.set_title(f"Tỷ lệ Chuyên cần (Attendance) vs Điểm thi (r = {corr_att:.3f})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Tỷ lệ chuyên cần (%)", fontsize=11)
    ax2.set_ylabel("Điểm thi (Exam Score)", fontsize=11)
    ax2.legend()

    plt.suptitle("Quan hệ Giữa Yếu tố Nỗ lực Cá nhân và Kết quả Thi", fontsize=15, fontweight="bold", y=1.02)
    save_path = output_dir / "05_hours_attendance_vs_exam_score.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Đã lưu biểu đồ: {save_path.name}")
    return save_path


def plot_categorical_comparisons(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Biểu đồ 6: So sánh Điểm thi trung bình theo Motivation Level, Access to Resources, Teacher Quality, Parental Involvement."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cat_cols = [
        ("Motivation_Level", ["Low", "Medium", "High"], "Mức độ Động lực học tập", axes[0, 0]),
        ("Access_to_Resources", ["Low", "Medium", "High"], "Tiếp cận Tài nguyên", axes[0, 1]),
        ("Teacher_Quality", ["Low", "Medium", "High"], "Chất lượng Giảng viên", axes[1, 0]),
        ("Parental_Involvement", ["Low", "Medium", "High"], "Sự quan tâm của Gia đình", axes[1, 1]),
    ]

    palette = {"Low": "#e74c3c", "Medium": "#f39c12", "High": "#2ecc71"}

    for col, order, title, ax in cat_cols:
        sns.barplot(
            data=df,
            x=col,
            y="Exam_Score",
            order=order,
            hue=col,
            palette=palette,
            ax=ax,
            errorbar="sd",
            capsize=0.1,
            edgecolor="black",
            legend=False,
        )
        # Thêm nhãn điểm trung bình trên mỗi cột
        means = df.groupby(col)["Exam_Score"].mean()
        for i, val in enumerate(order):
            if val in means:
                ax.text(i, means[val] / 2, f"{means[val]:.2f}", ha="center", va="center", color="white", fontweight="bold", fontsize=12)

        ax.set_title(f"Điểm thi theo {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Điểm thi trung bình (± SD)", fontsize=11)
        ax.set_ylim(0, 100)

    plt.suptitle("So sánh Điểm thi Trung bình theo các Yếu tố Môi trường & Động lực", fontsize=15, fontweight="bold", y=1.02)
    save_path = output_dir / "06_categorical_factors_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Đã lưu biểu đồ: {save_path.name}")
    return save_path


def get_proposed_input_rules(df: pd.DataFrame) -> Dict[str, Any]:
    """Đề xuất miền giá trị hợp lệ cho Người 1 (data validation) và Người 5 (Streamlit schema)."""
    rules = {
        "numeric_ranges": {},
        "categorical_allowed": {},
        "target_anomaly_note": {
            "column": "Exam_Score",
            "anomaly_value": 101,
            "recommendation": "Giới hạn về 100 (clip) hoặc loại bỏ bản ghi này vì thang điểm chuẩn là [0, 100].",
        },
    }

    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "Exam_Score":
            continue
        series = df[col].dropna()
        rules["numeric_ranges"][col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "recommended_min": 0.0,
            "recommended_max": float(series.max()),
            "unit": "giờ" if "Hours" in col else ("%" if col == "Attendance" else "buổi/lần"),
        }

    for col in df.select_dtypes(exclude=[np.number]).columns:
        categories = sorted([str(x) for x in df[col].dropna().unique()])
        rules["categorical_allowed"][col] = categories

    return rules


def run_full_eda(csv_path: Path = DATA_PATH) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Hàm điều phối toàn bộ quy trình EDA: tạo bảng, vẽ 6 biểu đồ, lưu file."""
    print("=" * 60)
    print("BẮT ĐẦU QUY TRÌNH PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) - THÀNH VIÊN 2")
    print("=" * 60)

    df = load_dataset(csv_path)
    print(f"1. Đã đọc dữ liệu thành công: {df.shape[0]} dòng, {df.shape[1]} cột.")

    summary_df = generate_data_summary(df)
    print(f"2. Đã tạo bảng tổng hợp 19 đặc trưng + 1 biến mục tiêu.")

    print("3. Đang xuất các biểu đồ trực quan hóa vào reports/figures/...")
    plot_target_distribution(df)
    plot_missing_values(df)
    plot_correlation_heatmap(df)
    plot_numeric_boxplots(df)
    plot_key_factors_vs_score(df)
    plot_categorical_comparisons(df)

    rules = get_proposed_input_rules(df)
    rules_path = REPORTS_DIR / "proposed_input_rules.json"
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)
    print(f"4. Đã lưu quy tắc miền hợp lệ bàn giao (Mốc H1): {rules_path}")

    print("=" * 60)
    print("HOÀN THÀNH TOÀN BỘ QUY TRÌNH EDA XUẤT SẮC!")
    print("=" * 60)
    return summary_df, rules


if __name__ == "__main__":
    run_full_eda()
