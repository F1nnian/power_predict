import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import (
    load_dataset,
    split_features_target,
    create_train_validation_split
)


MODEL_PATH = os.path.join("models", "final_model.joblib")
RESULTS_PATH = os.path.join("figures", "model_results.csv")


def main():
    os.makedirs("figures", exist_ok=True)

    print("Loading dataset...")
    df = load_dataset()
    X, y = split_features_target(df)

    X_train, X_val, y_train, y_val = create_train_validation_split(X, y)

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("Making validation predictions...")
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print("\nFinal validation performance:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2: {r2:.4f}")

    # plot 1 (MAE)
    if os.path.exists(RESULTS_PATH):
        results_df = pd.read_csv(RESULTS_PATH)

        plt.figure(figsize=(10, 6))
        plt.bar(results_df["model"], results_df["validation_mae"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Validation MAE")
        plt.title("Validation MAE for Different Models")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "model_comparison_mae.png"), dpi=300)
        plt.close()

        print("Saved figure: figures/model_comparison_mae.png")

    # plot 2 (True vs Predicted)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_val, y_pred, alpha=0.4)
    plt.xlabel("True power consumption")
    plt.ylabel("Predicted power consumption")
    plt.title("True vs Predicted Power Consumption")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "true_vs_predicted.png"), dpi=300)
    plt.close()

    print("Saved figure: figures/true_vs_predicted.png")

    # plot 3 (Residual distribution)
    residuals = y_val - y_pred

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=40)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "residuals_histogram.png"), dpi=300)
    plt.close()

    print("Saved figure: figures/residuals_histogram.png")

    # save final metrics to CSV
    final_metrics = pd.DataFrame([
        {
            "mae": mae,
            "mse": mse,
            "r2": r2
        }
    ])

    final_metrics.to_csv(os.path.join("figures", "final_validation_metrics.csv"), index=False)
    print("Saved metrics: figures/final_validation_metrics.csv")


if __name__ == "__main__":
    main()