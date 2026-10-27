
import json
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import (
    MODEL_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    RESULTS_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_data
from src.data_cleaning import clean_student_data


def evaluate_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Chua co mo hinh. Hay chay python -m src.train truoc."
        )

    data, _ = clean_student_data(load_data())
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(mean_squared_error(y_test, predictions) ** 0.5),
        "R2": float(r2_score(y_test, predictions)),
        "test_rows": int(len(y_test)),
    }

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_PATH / "evaluation_metrics.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation results:")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(f"Saved to: {output_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    evaluate_model()
