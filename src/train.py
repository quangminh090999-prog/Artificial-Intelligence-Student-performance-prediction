
<<<<<<< HEAD
"""Train, tune, compare, and save regression models."""
=======
"""Train and compare regression models for student exam score prediction."""
>>>>>>> origin/main

import json

import joblib
import pandas as pd
<<<<<<< HEAD

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
=======
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
>>>>>>> origin/main
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from src.config import (
    CV_FOLDS,
    MODEL_PATH,
<<<<<<< HEAD
=======
    PROJECT_ROOT,
>>>>>>> origin/main
    RANDOM_STATE,
    RESULTS_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_data
<<<<<<< HEAD
from src.preprocessing import build_preprocessor


def calculate_metrics(y_true, y_pred):
    """Calculate regression evaluation metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def train_models():
    """Train, compare, tune, evaluate, and save models."""

    # 1. Load data
    data = load_data()
=======
from src.data_cleaning import clean_student_data
from src.preprocessing import build_preprocessor


def train_models():
    """Train regression models, compare metrics, and save the best pipeline."""
    data, cleaning_summary = clean_student_data(load_data())
>>>>>>> origin/main

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

<<<<<<< HEAD
    # 2. Split training and testing data
=======
>>>>>>> origin/main
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

<<<<<<< HEAD
    # 3. Define models
=======
>>>>>>> origin/main
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

<<<<<<< HEAD
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    best_name = None
    best_pipeline = None
    best_cv_mae = float("inf")

    # 4. Train and compare models using cross-validation
    for name, model in models.items():

=======
    results = []
    best_name = None
    best_pipeline = None
    best_mae = float("inf")

    for name, model in models.items():
>>>>>>> origin/main
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])

<<<<<<< HEAD
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="neg_mean_absolute_error",
            cv=CV_FOLDS,
            n_jobs=-1,
        )

        cv_mae = -cv_scores.mean()

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = calculate_metrics(y_test, predictions)

        results.append({
            "model": name,
            "CV_MAE": cv_mae,
            **metrics,
        })

        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best_name = name
            best_pipeline = pipeline

    # 5. Tune Ridge using GridSearchCV
=======
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
>>>>>>> origin/main
    ridge_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", Ridge()),
    ])

    search = GridSearchCV(
        ridge_pipeline,
<<<<<<< HEAD
        param_grid={
            "model__alpha": [0.01, 0.1, 1, 10, 100, 1000]
        },
=======
        param_grid={"model__alpha": [0.01, 0.1, 1, 10, 100, 1000]},
>>>>>>> origin/main
        scoring="neg_mean_absolute_error",
        cv=CV_FOLDS,
        n_jobs=-1,
    )
<<<<<<< HEAD

    search.fit(X_train, y_train)

    # 6. Save Ridge cross-validation results
    cv_results = pd.DataFrame(search.cv_results_)

    cv_results = cv_results[
        [
            "param_model__alpha",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ].copy()

    cv_results["mean_CV_MAE"] = -cv_results["mean_test_score"]
    cv_results["std_CV_MAE"] = cv_results["std_test_score"]

    cv_results = cv_results[
        [
            "param_model__alpha",
            "mean_CV_MAE",
            "std_CV_MAE",
            "rank_test_score",
        ]
    ].rename(columns={
        "param_model__alpha": "alpha",
        "rank_test_score": "rank",
    })

    cv_results.to_csv(
        RESULTS_PATH / "ridge_cv_results.csv",
        index=False,
    )

    # 7. Get the best tuned Ridge model
    tuned_pipeline = search.best_estimator_

    tuned_predictions = tuned_pipeline.predict(X_test)
    tuned_metrics = calculate_metrics(y_test, tuned_predictions)

    results.append({
        "model": "Ridge_Tuned",
        "CV_MAE": -search.best_score_,
        **tuned_metrics,
    })

    if -search.best_score_ < best_cv_mae:
        best_name = "Ridge_Tuned"
        best_pipeline = tuned_pipeline
        best_cv_mae = -search.best_score_

    # 8. Export Ridge coefficients
    preprocessor = tuned_pipeline.named_steps["preprocessor"]
    ridge_model = tuned_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = ridge_model.coef_

    coefficient_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    })

    coefficient_df["absolute_coefficient"] = (
        coefficient_df["coefficient"].abs()
    )

    coefficient_df = coefficient_df.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    coefficient_df.to_csv(
        RESULTS_PATH / "ridge_coefficients.csv",
        index=False,
    )

    # 9. Evaluate the selected model on the test set
    best_predictions = best_pipeline.predict(X_test)
    best_metrics = calculate_metrics(y_test, best_predictions)

    # 10. Save model comparison
    results_df = pd.DataFrame(results).sort_values(
        "CV_MAE",
        ascending=True,
    )

    results_df.to_csv(
        RESULTS_PATH / "model_comparison.csv",
        index=False,
    )

    # 11. Save the best model
    joblib.dump(best_pipeline, MODEL_PATH)

    # 12. Save training summary
    summary = {
        "best_model": best_name,
        "best_cv_MAE": float(best_cv_mae),
        "test_MAE": float(best_metrics["MAE"]),
        "test_RMSE": float(best_metrics["RMSE"]),
        "test_R2": float(best_metrics["R2"]),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "best_ridge_alpha": float(
            search.best_params_["model__alpha"]
        ),
        "best_ridge_cv_MAE": float(-search.best_score_),
    }

    with open(
        RESULTS_PATH / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    # 13. Display results
    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    print("\n=== RIDGE TUNING ===")
    print(f"Best alpha: {search.best_params_['model__alpha']}")
    print(f"Best CV MAE: {-search.best_score_:.4f}")

    print("\n=== BEST MODEL ===")
    print(f"Model: {best_name}")
    print(f"Test MAE: {best_metrics['MAE']:.4f}")
    print(f"Test RMSE: {best_metrics['RMSE']:.4f}")
    print(f"Test R2: {best_metrics['R2']:.4f}")

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved results to: {RESULTS_PATH}")
=======
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
        "data_cleaning": cleaning_summary.to_dict(),
    }
    with open(RESULTS_PATH / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved results to: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
>>>>>>> origin/main

    return results_df


if __name__ == "__main__":
<<<<<<< HEAD
    train_models()
=======
    train_models()
>>>>>>> origin/main
