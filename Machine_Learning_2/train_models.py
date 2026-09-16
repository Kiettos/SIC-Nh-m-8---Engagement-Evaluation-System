
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


BASE_DIR = Path(r"C:\Users\KIET PC\Desktop\Focus_Guard")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "outputs" / "models"
RESULTS_DIR = BASE_DIR / "outputs" / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]

# Cac cot feature -- 20 cot da nen tu extract_features.py (khong lay ClipID/label/QA columns)
FEATURE_PREFIXES = ["ear_", "mar_", "yaw_", "pitch_"]


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]


def load_split_from_folder(folder: Path) -> pd.DataFrame:
    """
    Doc TOAN BO file CSV nho (moi file = 1 video, output cua run_extraction.py moi)
    trong 1 folder, gop lai thanh 1 DataFrame duy nhat de train.
    """
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Khong tim thay file CSV nao trong {folder}")

    rows = [pd.read_csv(f) for f in csv_files]
    return pd.concat(rows, ignore_index=True)


def load_data():
    # THAY DOI: gio doc tu 3 FOLDER (moi folder chua nhieu file CSV nho, 1 file/video)
    # thay vi doc 1 file CSV lon duy nhat nhu truoc.
    train_df = load_split_from_folder(PROCESSED_DIR / "train")
    val_df = load_split_from_folder(PROCESSED_DIR / "val")
    test_df = load_split_from_folder(PROCESSED_DIR / "test")

    # Loai bo dong bi NaN (video khong detect duoc frame nao -> feature rong)
    feature_cols = get_feature_columns(train_df)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        before = len(df)
        df.dropna(subset=feature_cols, inplace=True)
        after = len(df)
        if before != after:
            print(f"  {name}: bo {before - after} dong bi NaN (khong detect duoc face)")

    return train_df, val_df, test_df, feature_cols


def train_one_label(model_name: str, model_class, model_kwargs: dict,
                     X_train, y_train, X_val, y_val, X_test, y_test, label_name: str):
    """Train + danh gia 1 model cho 1 label (vd Random Forest cho Boredom)."""
    model = model_class(**model_kwargs)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    result = {
        "model": model_name,
        "label": label_name,
        "val_accuracy": accuracy_score(y_val, val_pred),
        "val_f1_macro": f1_score(y_val, val_pred, average="macro", zero_division=0),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_f1_macro": f1_score(y_test, test_pred, average="macro", zero_division=0),
        "test_confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "test_classification_report": classification_report(y_test, test_pred, zero_division=0, output_dict=True),
    }

    return model, result


def main():
    print("Loading data...")
    train_df, val_df, test_df, feature_cols = load_data()
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}\n")

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # 3 thuat toan can train, moi cai co hyperparameter rieng
    algorithms = {
        "DecisionTree": (
            DecisionTreeClassifier,
            {"max_depth": 8, "min_samples_leaf": 10, "class_weight": "balanced", "random_state": 42},
        ),
        "RandomForest": (
            RandomForestClassifier,
            {"n_estimators": 200, "max_depth": 12, "class_weight": "balanced", "random_state": 42, "n_jobs": -1},
        ),
        "XGBoost": (
            XGBClassifier,
            {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
             "eval_metric": "mlogloss", "random_state": 42},
        ),
    }

    all_results = []

    for model_name, (model_class, model_kwargs) in algorithms.items():
        print(f"\n{'='*60}\nTraining {model_name}\n{'='*60}")

        for label_name in LABEL_COLUMNS:
            y_train = train_df[label_name].values
            y_val = val_df[label_name].values
            y_test = test_df[label_name].values

            model, result = train_one_label(
                model_name, model_class, model_kwargs,
                X_train, y_train, X_val, y_val, X_test, y_test, label_name
            )

            print(f"  [{label_name}] val_acc={result['val_accuracy']:.3f}  "
                  f"val_f1={result['val_f1_macro']:.3f}  "
                  f"test_acc={result['test_accuracy']:.3f}  "
                  f"test_f1={result['test_f1_macro']:.3f}")

            # Luu model
            model_path = MODELS_DIR / f"{model_name}_{label_name}.pkl"
            joblib.dump(model, model_path)

            all_results.append(result)

    # Luu toan bo ket qua ra 1 file json duy nhat -- dung cho notebook so sanh sau nay
    results_path = RESULTS_DIR / "all_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n\nDa luu toan bo model vao: {MODELS_DIR}")
    print(f"Da luu ket qua vao: {results_path}")

    # In bang tong ket ngan gon
    print(f"\n{'='*60}\nTONG KET (test_f1_macro)\n{'='*60}")
    summary_df = pd.DataFrame(all_results)[["model", "label", "test_f1_macro", "test_accuracy"]]
    pivot = summary_df.pivot(index="label", columns="model", values="test_f1_macro")
    print(pivot.round(3))


if __name__ == "__main__":
    main()
