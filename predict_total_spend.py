from pathlib import Path
import json
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_features(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    numeric_means: dict,
    numeric_stds: dict,
    feature_columns: list[str],
):
    x = df[categorical_cols + numeric_cols].copy()

    for col in numeric_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x = pd.get_dummies(x, columns=categorical_cols, drop_first=False)

    for col in numeric_cols:
        std = float(numeric_stds[col]) if float(numeric_stds[col]) != 0 else 1.0
        x[col] = (x[col] - float(numeric_means[col])) / std

    x = x.fillna(0.0)
    x = x.reindex(columns=feature_columns, fill_value=0.0)

    logger.info("Przygotowano cechy testowe. Liczba kolumn: %d", len(feature_columns))
    logger.info("Rozmiar macierzy cech testowych: %s", x.shape)
    return x


def main():
    script_dir = Path(__file__).resolve().parent
    artifacts_dir = script_dir / os.getenv("ARTIFACTS_DIR", "revenue_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    preprocess_path = artifacts_dir / "preprocessing.json"
    test_raw_path = artifacts_dir / "test_raw.csv"
    predictions_path = artifacts_dir / "test_predictions.csv"
    metrics_path = artifacts_dir / "evaluation_metrics.json"
    summary_path = artifacts_dir / "evaluation_summary.txt"

    logger.info("Katalog artefaktów: %s", artifacts_dir)

    if not preprocess_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku preprocessingu: {preprocess_path}")
    if not test_raw_path.exists():
        raise FileNotFoundError(f"Nie znaleziono danych testowych: {test_raw_path}")

    with open(preprocess_path, "r", encoding="utf-8") as f:
        prep = json.load(f)
    logger.info("Wczytano konfigurację preprocessingu")

    model_path = artifacts_dir / prep.get("model_path", "revenue_model.keras")
    if not model_path.exists():
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")

    df_test = pd.read_csv(test_raw_path)
    logger.info("Wczytano dane testowe. Rozmiar: %s", df_test.shape)

    x_test = build_features(
        df_test,
        categorical_cols=prep["categorical_cols"],
        numeric_cols=prep["numeric_cols"],
        numeric_means=prep["numeric_means"],
        numeric_stds=prep["numeric_stds"],
        feature_columns=prep["feature_columns"],
    )

    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(x_test.astype("float32").to_numpy(), verbose=0).reshape(-1)
    logger.info("Wykonano predykcję dla %d rekordów", len(predictions))

    result_df = pd.DataFrame({"predicted_total_spend": predictions})

    target_col = prep.get("target_col", "total_spend")
    metrics = {"n_test": int(len(predictions))}

    if target_col in df_test.columns:
        y_true = pd.to_numeric(df_test[target_col], errors="coerce").astype("float32").to_numpy()
        mask = ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = predictions[mask]

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))

        metrics.update(
            {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": rmse,
            }
        )
        result_df[target_col] = df_test[target_col]
        logger.info("Metryki ewaluacji: MAE=%.4f, MSE=%.4f, RMSE=%.4f", mae, mse, rmse)
    else:
        logger.warning("Brak kolumny targetowej %s w danych testowych - zapisuję tylko predykcje.", target_col)

    result_df.to_csv(predictions_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}={value}\n")

    logger.info("Zapisano predykcje do: %s", predictions_path)
    logger.info("Zapisano metryki do: %s", metrics_path)
    logger.info("Przykładowe predykcje:\n%s", result_df.head())


if __name__ == "__main__":
    main()
