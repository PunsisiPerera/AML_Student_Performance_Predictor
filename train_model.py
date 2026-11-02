from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
)
from interpret.glassbox import ExplainableBoostingClassifier
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
for folder in (DATA_DIR, OUT_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def get_data():
    """Download the official UCI archive once and load the mathematics subset."""
    local_csv = DATA_DIR / "student-mat.csv"
    if local_csv.exists():
        try:
            df = pd.read_csv(local_csv, sep=";")
            if "G3" in df.columns:
                return df
        except Exception:
            pass
    print("Downloading official UCI Student Performance archive (internet required on first run)...")
    import urllib.request
    import zipfile
    import io
    url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = next((n for n in archive.namelist() if n.lower().endswith("student-mat.csv")), None)
            if member is None:
                raise FileNotFoundError("student-mat.csv was not found inside the UCI archive.")
            with archive.open(member) as f:
                df = pd.read_csv(f, sep=";")
        df.to_csv(local_csv, sep=";", index=False)
        return df
    except Exception as exc:
        raise RuntimeError(
            "Could not download the official UCI dataset. Check your internet connection, "
            "or manually download student-mat.csv from the UCI Student Performance dataset "
            "and place it in the project's data/ folder."
        ) from exc

def prepare_data(df):
    """Create a leakage-aware binary target and predictors available before final grade."""
    if "G3" not in df.columns:
        raise ValueError("Dataset must include G3, the final grade.")
    y = (pd.to_numeric(df["G3"], errors="coerce") >= 10).astype(int)
    # Exclude grades and identifying/direct demographic fields from predictors.
    exclude = {"G1", "G2", "G3", "sex", "age", "school", "address"}
    X = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore").copy()
    # Exclude any row with missing target; impute predictor missing values.
    valid = pd.to_numeric(df["G3"], errors="coerce").notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    # Keep strings categorical and cast booleans consistently.
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("Unknown").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())
    return X, y

def main():
    df = get_data()
    print(f"Dataset rows: {len(df)} | columns: {len(df.columns)}")
    X, y = prepare_data(df)
    print(f"Target distribution (0=at risk, 1=pass): {y.value_counts().sort_index().to_dict()}")
    # Stratified 70/15/15 split. Split the temporary 30% equally into validation and test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    # EBM supports categorical features directly; no one-hot encoding or scaling required.
    model = ExplainableBoostingClassifier(
        max_bins=64, max_interaction_bins=16, interactions=5,
        learning_rate=0.03, outer_bags=8, inner_bags=0,
        validation_size=0.15, max_rounds=3000, early_stopping_rounds=100,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    # Validation threshold selection: choose threshold maximizing F1 on validation data.
    val_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.20, 0.81, 0.01)
    val_f1 = [f1_score(y_val, (val_prob >= t).astype(int), zero_division=0) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(val_f1))])
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    metrics = {
        "algorithm": "Explainable Boosting Machine (EBM)",
        "target_definition": "pass if G3 >= 10/20; at_risk otherwise",
        "random_state": RANDOM_STATE,
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "selected_threshold": threshold,
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision_pass": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall_pass": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1_pass": float(f1_score(y_test, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "classification_report": classification_report(
            y_test, test_pred, target_names=["At risk", "Pass"], output_dict=True, zero_division=0
        ),
        "dataset_rows": int(len(X)),
        "feature_count": int(X.shape[1])
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(list(X.columns), indent=2), encoding="utf-8")
    joblib.dump({"model": model, "threshold": threshold, "feature_columns": list(X.columns)}, MODEL_DIR / "ebm_model.joblib")
    (OUT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, test_pred, target_names=["At risk", "Pass"], zero_division=0),
        encoding="utf-8"
    )
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["At risk", "Pass"], yticklabels=["At risk", "Pass"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("EBM Confusion Matrix — Test Set")
    plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=160); plt.close()
    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, test_prob, ax=ax)
    ax.set_title("EBM ROC Curve — Test Set")
    fig.tight_layout(); fig.savefig(OUT_DIR / "roc_curve.png", dpi=160); plt.close(fig)
    # Global feature importance (EBM global explanation)
    global_exp = model.explain_global()
    global_data = global_exp.data()

    names = global_data.get("names", [])
    scores = global_data.get("scores", [])

    imp = []

    for name, score in zip(names, scores):
        importance = float(np.mean(np.abs(np.asarray(score, dtype=float))))

        imp.append({
            "feature": name,
            "importance": importance
        })

    imp_df = pd.DataFrame(
        imp,
        columns=["feature", "importance"]
    )

    imp_df = imp_df.sort_values(
        "importance",
        ascending=False
    )

    imp_df.to_csv(
        OUT_DIR / "feature_importance.csv",
        index=False
    )

    if not imp_df.empty:
        plt.figure(figsize=(8, max(4, 0.35 * min(15, len(imp_df)))))

        sns.barplot(
            data=imp_df.head(15),
            x="importance",
            y="feature"
        )

        plt.title("Top EBM Global Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")

        plt.tight_layout()
        plt.savefig(OUT_DIR / "feature_importance.png", dpi=160)
        plt.close()
    print("\nTEST RESULTS")
    for key in ["accuracy", "precision_pass", "recall_pass", "f1_pass", "roc_auc"]:
        print(f"{key}: {metrics[key]:.3f}")
    print(f"Validation-selected threshold: {threshold:.2f}")
    print(f"Saved model: {MODEL_DIR / 'ebm_model.joblib'}")
    print(f"Saved results in: {OUT_DIR}")
    print("\nImportant: report the actual output metrics; do not copy example scores.")

if __name__ == "__main__":
    main()
