import os
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import (
    load_dataset,
    split_features_target,
    create_train_validation_split,
    build_preprocessor
)


MODEL_PATH = os.path.join("models", "final_model.joblib")
RESULTS_PATH = os.path.join("figures", "model_results.csv")
METADATA_PATH = os.path.join("models", "training_metadata.json")


def evaluate_predictions(y_true, y_pred):
    """
    Computes regression metrics.
    """

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mae, mse, r2


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("Loading dataset...")
    df = load_dataset()
    X, y = split_features_target(df)

    print(f"Dataset shape: {df.shape}")
    print(f"Number of input features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")

    X_train, X_val, y_train, y_val = create_train_validation_split(X, y)

    preprocessor = build_preprocessor(X_train)

    models = {
        "DummyRegressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DummyRegressor(strategy="mean"))
            ]
        ),

        "Ridge_alpha_1": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Ridge(alpha=1.0))
            ]
        ),

        "Ridge_alpha_10": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Ridge(alpha=10.0))
            ]
        ),

        "RandomForest_depth_10": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ))
            ]
        ),

        "RandomForest_depth_20_small": Pipeline(
        steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=30,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1))
        ])}

    results = []

    best_model = None
    best_model_name = None
    best_validation_mae = float("inf")

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        validation_predictions = model.predict(X_val)

        train_mae, train_mse, train_r2 = evaluate_predictions(y_train, train_predictions)
        val_mae, val_mse, val_r2 = evaluate_predictions(y_val, validation_predictions)

        results.append({
            "model": model_name,
            "train_mae": train_mae,
            "validation_mae": val_mae,
            "train_mse": train_mse,
            "validation_mse": val_mse,
            "train_r2": train_r2,
            "validation_r2": val_r2
        })

        print(f"Train MAE: {train_mae:.2f}")
        print(f"Validation MAE: {val_mae:.2f}")
        print(f"Validation MSE: {val_mse:.2f}")
        print(f"Validation R2: {val_r2:.4f}")

        if val_mae < best_validation_mae:
            best_validation_mae = val_mae
            best_model = model
            best_model_name = model_name

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        "best_model": best_model_name,
        "best_validation_mae": best_validation_mae,
        "model_path": MODEL_PATH,
        "results_path": RESULTS_PATH,
        "target_column": "power_consumption",
        "random_state": 42
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nFinal model comparison:")
    print(results_df)

    print("\nBest model:")
    print(best_model_name)
    print(f"Best validation MAE: {best_validation_mae:.2f}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()