

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, make_scorer

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE


BASE_DIR = Path(r"C:\Users\KIET PC\Desktop\Focus_Guard")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "outputs" / "models_tuned3"
RESULTS_DIR = BASE_DIR / "outputs" / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]
FEATURE_PREFIXES = ["ear_", "mar_", "yaw_", "pitch_"]

N_ITER_SEARCH = 25   # so to hop ngau nhien thu cho MOI model -- tang len neu co du thoi gian

#CAU HINH THU NGHIEM (bat/tat de so sanh ket qua)
USE_BINARY_LABELS = False   # True = gop 4 muc (0-3) thanh 2 muc (Low/High)
USE_SMOTE = True            # True = ap dung SMOTE can bang lop hiem (chi tren Train)
SMOTE_K_NEIGHBORS = 5        # so hang xom dung de noi suy (tu dong giam neu lop hiem qua it)


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]


def load_split_from_folder(folder: Path) -> pd.DataFrame:
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Khong tim thay file CSV nao trong {folder}")
    rows = [pd.read_csv(f) for f in csv_files]
    return pd.concat(rows, ignore_index=True)


def load_data():
    train_df = load_split_from_folder(PROCESSED_DIR / "train")
    val_df = load_split_from_folder(PROCESSED_DIR / "val")
    test_df = load_split_from_folder(PROCESSED_DIR / "test")

    feature_cols = get_feature_columns(train_df)
    for df in [train_df, val_df, test_df]:
        df.dropna(subset=feature_cols, inplace=True)
    if USE_BINARY_LABELS:
        for df in [train_df, val_df, test_df]:
            for label in LABEL_COLUMNS:
                df[label] = (df[label] >= 2).astype(int)  # 0,1 -> 0 (Low) | 2,3 -> 1 (High)

    return train_df, val_df, test_df, feature_cols


# SEARCH SPACE -- pham vi hyperparameter can thu cho moi thuat toan
SEARCH_SPACES = {
    "DecisionTree": {
        "class": DecisionTreeClassifier,
        "fixed_params": {"class_weight": "balanced", "random_state": 42},
        "search_space": {
            "max_depth": [3, 5, 8, 10, 15, None],
            "min_samples_leaf": [1, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10, 20],
            "criterion": ["gini", "entropy"],
        },
    },
    "RandomForest": {
        "class": RandomForestClassifier,
        "fixed_params": {"class_weight": "balanced", "random_state": 42, "n_jobs": -1},
        "search_space": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [8, 12, 16, 20, None],
            "min_samples_leaf": [1, 5, 10, 20],
            "max_features": ["sqrt", "log2", None],
        },
    },
    "XGBoost": {
        "class": XGBClassifier,
        "fixed_params": {"eval_metric": "mlogloss", "random_state": 42},
        "search_space": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        },
    },
}


def make_predefined_split(n_train: int, n_val: int) -> PredefinedSplit:
    """
    Tao 1 PredefinedSplit dung dung Train/Val co san:
    -1 = luon nam trong tap TRAIN (khong bao gio dung de danh gia)
     0 = luon nam trong tap VALIDATION (dung de danh gia, chi 1 "fold" duy nhat)
    Day la cach ep RandomizedSearchCV dung DUNG train/val goc, khong tu xao tron.
    """
    test_fold = np.concatenate([
        np.full(n_train, -1),  # train -> khong bao gio danh gia
        np.full(n_val, 0),      # val   -> fold duy nhat dung de danh gia
    ])
    return PredefinedSplit(test_fold)


def apply_smote(X_train, y_train, label_name: str):
    if not USE_SMOTE:
        return X_train, y_train

    # Dem so sample it nhat trong cac lop -- SMOTE can k_neighbors < so sample lop hiem nhat
    unique, counts = np.unique(y_train, return_counts=True)
    min_class_count = counts.min()

    if min_class_count <= 1:
        print(f"    [SMOTE] Bo qua {label_name}: lop hiem nhat chi co {min_class_count} sample")
        return X_train, y_train

    # k_neighbors phai nho hon so sample cua lop hiem nhat
    k = min(SMOTE_K_NEIGHBORS, min_class_count - 1)

    try:
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"    [SMOTE] {label_name}: {len(X_train)} -> {len(X_res)} sample (k_neighbors={k})")
        return X_res, y_res
    except Exception as e:
        print(f"    [SMOTE] Loi voi {label_name}, giu nguyen data goc: {e}")
        return X_train, y_train


def tune_one_label(algo_name: str, algo_config: dict,
                    X_train, y_train, X_val, y_val, label_name: str):
    """Chay RandomizedSearchCV cho 1 thuat toan + 1 label, dung dung Train/Val co san."""

    # AP DUNG SMOTE chi tren Train (Val giu nguyen de danh gia trung thuc)
    X_train_res, y_train_res = apply_smote(X_train, y_train, label_name)

    # Gop Train(da SMOTE)+Val lai (RandomizedSearchCV can toan bo data + chi dinh fold qua cv=)
    X_combined = np.vstack([X_train_res, X_val])
    y_combined = np.concatenate([y_train_res, y_val])

    predefined_split = make_predefined_split(len(X_train_res), len(X_val))

    model = algo_config["class"](**algo_config["fixed_params"])

    f1_macro_scorer = make_scorer(f1_score, average="macro", zero_division=0)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=algo_config["search_space"],
        n_iter=N_ITER_SEARCH,
        scoring=f1_macro_scorer,    
        cv=predefined_split,          
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_combined, y_combined)

    print(f"  [{algo_name} - {label_name}] best_val_f1_macro={search.best_score_:.3f}")
    print(f"    best_params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


def main():
    print("Loading data...")
    train_df, val_df, test_df, feature_cols = load_data()

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values

    all_results = []

    for algo_name, algo_config in SEARCH_SPACES.items():
        print(f"\n{'='*60}\nTuning {algo_name}\n{'='*60}")

        for label_name in LABEL_COLUMNS:
            y_train = train_df[label_name].values
            y_val = val_df[label_name].values
            y_test = test_df[label_name].values

            best_model, best_params, best_val_score = tune_one_label(
                algo_name, algo_config, X_train, y_train, X_val, y_val, label_name
            )

            # Danh gia model tot nhat tren TEST (chi 1 lan duy nhat, sau khi da chon xong params)
            test_pred = best_model.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)

            print(f"    test_accuracy={test_acc:.3f}  test_f1_macro={test_f1:.3f}")

            model_path = MODELS_DIR / f"{algo_name}_{label_name}_tuned.pkl"
            joblib.dump(best_model, model_path)

            all_results.append({
                "model": algo_name,
                "label": label_name,
                "best_params": best_params,
                "val_f1_macro": best_val_score,
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
            })

    results_path = RESULTS_DIR / "tuning_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n\nDa luu model toi uu vao: {MODELS_DIR}")
    print(f"Da luu ket qua tuning vao: {results_path}")

    print(f"\n{'='*60}\nTONG KET (test_f1_macro sau khi tune)\n{'='*60}")
    summary_df = pd.DataFrame(all_results)[["model", "label", "test_f1_macro", "test_accuracy"]]
    pivot = summary_df.pivot(index="label", columns="model", values="test_f1_macro")
    print(pivot.round(3))


if __name__ == "__main__":
    main()