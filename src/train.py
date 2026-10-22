
"""Train and compare regression models for student exam score prediction."""

import json

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from src.config import (
    CV_FOLDS,
    MODEL_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    RESULTS_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_data
from src.preprocessing import build_preprocessor


def train_models():
    """Train regression models, compare metrics, and save the best pipeline."""
    data = load_data()

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
        "SVR": SVR(),
    }

    results = []
    best_name = None
    best_pipeline = None
    best_mae = float("inf")

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_pipeline = pipeline

    # Tune Ridge using cross-validation on the training set only.
    ridge_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", Ridge()),
    ])

    search = GridSearchCV(
        ridge_pipeline,
        param_grid={"model__alpha": [0.01, 0.1, 1, 10, 100, 1000]},
        scoring="neg_mean_absolute_error",
        cv=CV_FOLDS,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    tuned_pipeline = search.best_estimator_
    tuned_predictions = tuned_pipeline.predict(X_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    tuned_mae = mean_absolute_error(y_test, tuned_predictions)
    tuned_rmse = mean_squared_error(y_test, tuned_predictions) ** 0.5
    tuned_r2 = r2_score(y_test, tuned_predictions)

    results.append({
        "model": "Ridge_Tuned",
        "MAE": tuned_mae,
        "RMSE": tuned_rmse,
        "R2": tuned_r2,
        "best_alpha": search.best_params_["model__alpha"],
        "best_cv_MAE": -search.best_score_,
    })

    if tuned_mae < best_mae:
        best_name = "Ridge_Tuned"
        best_pipeline = tuned_pipeline

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results).sort_values("MAE")
    results_df.to_csv(RESULTS_PATH / "model_comparison.csv", index=False)

    joblib.dump(best_pipeline, MODEL_PATH)

    summary = {
        "best_model": best_name,
        "test_MAE": float(best_mae if best_name != "Ridge_Tuned" else tuned_mae),
        "test_rows": len(X_test),
        "train_rows": len(X_train),
        "best_ridge_alpha": search.best_params_["model__alpha"],
        "best_ridge_cv_MAE": float(-search.best_score_),
    }
    with open(RESULTS_PATH / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved results to: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

    return results_df


if __name__ == "__main__":
    train_models()
