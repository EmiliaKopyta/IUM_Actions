from pathlib import Path
import json
import logging
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SEED = 42
TARGET_COL = "total_spend"

CATEGORICAL_CANDIDATES = {
    "time_of_day": ["time_of_day", "order_time_of_day"],
    "store_location_type": ["store_location_type", "location_type", "store_type"],
}

NUMERIC_CANDIDATES = {
    "num_customizations": ["num_customizations", "customization_count"],
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info("Ustawiono seed na %s", seed)


def resolve_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Nie znaleziono kolumny dla '{label}'. "
        f"Sprawdzane nazwy: {candidates}. "
        f"Dostępne kolumny: {list(df.columns)}"
    )


def load_dataset(script_dir: Path) -> pd.DataFrame:
    data_path = Path(os.getenv("DATA_PATH", "starbucks_customer_ordering_patterns.csv"))
    if not data_path.is_absolute():
        data_path = script_dir / data_path

    if not data_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku CSV: {data_path}")

    df = pd.read_csv(data_path)
    logger.info("Wczytano dane z: %s", data_path)
    logger.info("Rozmiar zbioru: %s", df.shape)
    return df


def add_time_of_day_column(df: pd.DataFrame) -> pd.DataFrame:
    """Tworzy time_of_day z order_time, jeśli time_of_day nie ma jeszcze w danych."""
    df = df.copy()

    if "time_of_day" in df.columns:
        logger.info("Kolumna time_of_day już istnieje - nie tworzę jej ponownie.")
        return df

    if "order_time" not in df.columns:
        raise ValueError(
            "Brakuje kolumny 'time_of_day' lub 'order_time'. "
            "Dodaj jedną z nich do CSV albo zmień przygotowanie danych."
        )

    parsed_time = pd.to_datetime(
        df["order_time"].astype(str).str.strip(),
        format="%H:%M",
        errors="coerce",
    )
    hours = parsed_time.dt.hour

    def map_hour_to_period(hour):
        if pd.isna(hour):
            return None
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 21:
            return "evening"
        return "night"

    df["time_of_day"] = hours.apply(map_hour_to_period)
    return df


def prepare_dataframe(df: pd.DataFrame):
    df = add_time_of_day_column(df)

    time_col = resolve_column(df, CATEGORICAL_CANDIDATES["time_of_day"], "time_of_day")
    location_col = resolve_column(
        df,
        CATEGORICAL_CANDIDATES["store_location_type"],
        "store_location_type",
    )
    custom_col = resolve_column(
        df,
        NUMERIC_CANDIDATES["num_customizations"],
        "num_customizations",
    )

    used_columns = [time_col, location_col, custom_col, TARGET_COL]

    work_df = df[used_columns].copy()
    work_df[custom_col] = pd.to_numeric(work_df[custom_col], errors="coerce")
    work_df[TARGET_COL] = pd.to_numeric(work_df[TARGET_COL], errors="coerce")
    work_df = work_df.dropna(subset=used_columns).reset_index(drop=True)
    work_df[time_col] = work_df[time_col].astype(str)
    work_df[location_col] = work_df[location_col].astype(str)

    logger.info("Użyte kolumny:")
    logger.info("time_of_day: %s", time_col)
    logger.info("store_location_type: %s", location_col)
    logger.info("num_customizations: %s", custom_col)
    logger.info("target: %s", TARGET_COL)
    logger.info("Rozmiar po czyszczeniu: %s", work_df.shape)

    return work_df, time_col, location_col, custom_col


def build_features(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    numeric_means: dict | None = None,
    numeric_stds: dict | None = None,
    feature_columns: list[str] | None = None,
):
    x = df[categorical_cols + numeric_cols].copy()

    for col in numeric_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x = pd.get_dummies(x, columns=categorical_cols, drop_first=False)

    if numeric_means is None or numeric_stds is None:
        numeric_means = {}
        numeric_stds = {}
        for col in numeric_cols:
            mean = float(x[col].mean())
            std = float(x[col].std())
            if std == 0 or np.isnan(std):
                std = 1.0
            numeric_means[col] = mean
            numeric_stds[col] = std

    for col in numeric_cols:
        x[col] = (x[col] - numeric_means[col]) / numeric_stds[col]

    x = x.fillna(0.0)

    if feature_columns is None:
        feature_columns = x.columns.tolist()
    else:
        x = x.reindex(columns=feature_columns, fill_value=0.0)

    logger.info("Przygotowano cechy. Liczba kolumn wejściowych: %d", len(feature_columns))
    return x, numeric_means, numeric_stds, feature_columns


def create_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    logger.info("Utworzono model z input_dim=%d", input_dim)
    return model


def main():
    set_seed()

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / os.getenv("ARTIFACTS_DIR", "revenue_artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(os.getenv("EPOCHS", "10"))
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    test_size = float(os.getenv("TEST_SIZE", "0.2"))
    validation_split = float(os.getenv("VALIDATION_SPLIT", "0.2"))

    logger.info(
        "Parametry treningu: epochs=%s, batch_size=%s, test_size=%s, validation_split=%s",
        epochs,
        batch_size,
        test_size,
        validation_split,
    )
    logger.info("Katalog artefaktów: %s", output_dir)

    df = load_dataset(script_dir)
    work_df, time_col, location_col, custom_col = prepare_dataframe(df)

    train_df, test_df = train_test_split(
        work_df,
        test_size=test_size,
        random_state=SEED,
    )
    logger.info("Podział danych: train=%s, test=%s", train_df.shape, test_df.shape)

    categorical_cols = [time_col, location_col]
    numeric_cols = [custom_col]

    x_train, numeric_means, numeric_stds, feature_columns = build_features(
        train_df,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )

    y_train = train_df[TARGET_COL].astype("float32").to_numpy()
    x_train_np = x_train.astype("float32").to_numpy()

    model = create_model(input_dim=x_train_np.shape[1])

    logger.info("Rozpoczynam trening modelu")
    history = model.fit(
        x_train_np,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    train_loss, train_mae = model.evaluate(x_train_np, y_train, verbose=0)
    final_val_loss = float(history.history["val_loss"][-1])
    final_val_mae = float(history.history["val_mae"][-1])

    model_path = output_dir / "revenue_model.keras"
    model.save(model_path)
    logger.info("Zapisano model lokalnie do: %s", model_path)

    test_raw_path = output_dir / "test_raw.csv"
    test_df.to_csv(test_raw_path, index=False)
    logger.info("Zapisano zbiór testowy do: %s", test_raw_path)

    history_path = output_dir / "training_history.csv"
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    logger.info("Zapisano historię treningu do: %s", history_path)

    preprocess_path = output_dir / "preprocessing.json"
    preprocess_data = {
        "target_col": TARGET_COL,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "feature_columns": feature_columns,
        "numeric_means": numeric_means,
        "numeric_stds": numeric_stds,
        "model_path": "revenue_model.keras",
        "seed": SEED,
    }
    with open(preprocess_path, "w", encoding="utf-8") as f:
        json.dump(preprocess_data, f, indent=2)
    logger.info("Zapisano preprocessing do: %s", preprocess_path)

    summary_path = output_dir / "train_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"epochs={epochs}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"test_size={test_size}\n")
        f.write(f"validation_split={validation_split}\n")
        f.write(f"train_loss={float(train_loss)}\n")
        f.write(f"train_mae={float(train_mae)}\n")
        f.write(f"val_loss_final={final_val_loss}\n")
        f.write(f"val_mae_final={final_val_mae}\n")
        f.write(f"model_path={model_path}\n")
        f.write(f"test_raw_path={test_raw_path}\n")
    logger.info("Zapisano podsumowanie treningu do: %s", summary_path)


if __name__ == "__main__":
    main()
