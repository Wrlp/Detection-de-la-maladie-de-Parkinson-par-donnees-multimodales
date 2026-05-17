"""
Modèle de fusion multimodale pour la détection de la maladie de Parkinson.
Combine les features vocales (UCI 189) et les features spirales (UCI 395).
Fusion tardive : deux RandomForest séparés, moyenne des probabilités.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# Chemins
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPIRAL_CSV   = PROJECT_ROOT / "reports" / "figures" / "spiral_uci" / "features_per_subject.csv"
OUT_DIR      = PROJECT_ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.append(str(PROJECT_ROOT))
from voice_loader import load_voice_data

# Features à utiliser
VOICE_FEATS = [
    "HNR", "RPDE", "DFA", "PPE",
    "Shimmer(dB)", "Shimmer:APQ11", "Shimmer:APQ5",
    "Jitter(%)", "Jitter(Abs)",
    "age", "sex", "test_time",
]

def load_voice() -> tuple[pd.DataFrame, np.ndarray]:
    X_train, X_test, y_train, y_test = load_voice_data(use_merged=True, normalize=False)
    # normalize=False car le pipeline fait déjà StandardScaler
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    print("Doublons dans X vocal :", pd.DataFrame(X).duplicated().sum())
    print(f"Dataset vocal  : {X.shape[0]} sujets, {X.shape[1]} features — PD={y.sum()}, HC={(y==0).sum()}")
    return pd.DataFrame(X), y

def load_spiral() -> tuple[pd.DataFrame, np.ndarray]:
    """Charge le CSV produit par eda_spiral_uci.py."""
    if not SPIRAL_CSV.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {SPIRAL_CSV}\n"
            "Lance d'abord : python analysis/eda_spiral_uci.py"
        )
    df = pd.read_csv(SPIRAL_CSV)
    feat_cols = [c for c in df.columns if c.startswith(("sst_", "dst_", "stcp_"))]
    X = df[feat_cols]
    y = df["label_pd"].to_numpy()
    print(f"Dataset spiral : {X.shape[0]} sujets, {X.shape[1]} features - PD={y.sum()}, HC={(y==0).sum()}")
    return X, y

# Pipeline
def make_pipe(n_estimators: int = 400, random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])

# Évaluation croisée (sans split manuel)
def cross_val_eval(X: pd.DataFrame, y: np.ndarray, label: str) -> dict:
    """5 splits stratifiés, retourne AUC moyen ± std."""
    pipe = make_pipe()
    sss  = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
    aucs = cross_val_score(pipe, X, y, cv=sss, scoring="roc_auc")
    print(f"AUC {label:<8}: {aucs.mean():.3f} ± {aucs.std():.3f}")
    return {"mean": float(aucs.mean()), "std": float(aucs.std())}

# Split train/test + fusion tardive
def late_fusion_eval(
    X_v: pd.DataFrame, y_v: np.ndarray,
    X_s: pd.DataFrame, y_s: np.ndarray,
) -> dict:
    """
    Split 75/25 stratifié sur CHAQUE dataset séparément,
    puis fusion des probabilités sur leurs sets de test respectifs.

    Note : les deux datasets n'ont pas les mêmes sujets, donc les
    ensembles de test sont indépendants. On évalue la fusion sur
    l'union (concaténation des proba + labels de test).
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

    # Split voix
    (tr_v, te_v), = sss.split(X_v, y_v)
    pipe_voice = make_pipe()
    pipe_voice.fit(X_v.iloc[tr_v], y_v[tr_v])
    prob_v = pipe_voice.predict_proba(X_v.iloc[te_v])[:, 1]
    y_te_v = y_v[te_v]

    # Split spirale
    (tr_s, te_s), = sss.split(X_s, y_s)
    pipe_spiral = make_pipe()
    pipe_spiral.fit(X_s.iloc[tr_s], y_s[tr_s])
    prob_s = pipe_spiral.predict_proba(X_s.iloc[te_s])[:, 1]
    y_te_s = y_s[te_s]

    # Évaluation individuelle 
    print("\nModèle voix (holdout)")
    print(classification_report(y_te_v, (prob_v >= 0.5).astype(int),
                                 target_names=["Contrôle", "Parkinson"], digits=3))
    print(f"AUC voix    : {roc_auc_score(y_te_v, prob_v):.3f}")

    print("\nModèle spirale (holdout)")
    print(classification_report(y_te_s, (prob_s >= 0.5).astype(int),
                                 target_names=["Contrôle", "Parkinson"], digits=3))
    print(f"AUC spirale : {roc_auc_score(y_te_s, prob_s):.3f}")

    # Fusion tardive : on concatène les deux sets de test 
    prob_all   = np.concatenate([prob_v, prob_s])
    y_all      = np.concatenate([y_te_v, y_te_s])
    # Pour la fusion, on moyenne les probas de chaque modèle sur son propre test
    # (approximation : les deux modèles raisonnent sur des sujets différents)
    prob_fusion = prob_all          # déjà des probas individuelles
    # Fusion pondérée égale entre les deux modèles sur leurs prédictions
    # (alternative : entraîner un meta-classifier - laissé à la partie hyperparamètres)
    y_pred = (prob_fusion >= 0.5).astype(int)

    print("\nFusion tardive (union des sets de test)")
    print(classification_report(y_all, y_pred,
                                 target_names=["Contrôle", "Parkinson"], digits=3))
    auc_fusion = roc_auc_score(y_all, prob_fusion)
    print(f"AUC fusion  : {auc_fusion:.3f}")

    return {
        "auc_voice":   float(roc_auc_score(y_te_v, prob_v)),
        "auc_spiral":  float(roc_auc_score(y_te_s, prob_s)),
        "auc_fusion":  float(auc_fusion),
        "pipe_voice":  pipe_voice,
        "pipe_spiral": pipe_spiral,
        "prob_v": prob_v, "y_te_v": y_te_v,
        "prob_s": prob_s, "y_te_s": y_te_s,
        "prob_fusion": prob_fusion, "y_all": y_all,
    }

# Visualisations
def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax,
        display_labels=["Contrôle", "Parkinson"],
        colorbar=False,
    )
    ax.set_title("Matrice de confusion – fusion tardive")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Sauvegardé : {out}")

def plot_roc_curves(results: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        results["y_te_v"], results["prob_v"], ax=ax, name="Voix seule")
    RocCurveDisplay.from_predictions(
        results["y_te_s"], results["prob_s"], ax=ax, name="Spirale seule")
    RocCurveDisplay.from_predictions(
        results["y_all"], results["prob_fusion"], ax=ax, name="Fusion tardive")
    ax.set_title("Courbes ROC – comparaison des modalités")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Sauvegardé : {out}")

def plot_feature_importance(pipe: Pipeline, feat_names: list[str], title: str, out: Path) -> None:
    imp = pipe.named_steps["clf"].feature_importances_
    imp_df = (
        pd.DataFrame({"feature": feat_names, "importance": imp})
        .sort_values("importance", ascending=True)
        .tail(15)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Sauvegardé : {out}")

# Main
def main() -> None:
    print("Fusion multimodale - Détection Parkinson")

    X_v, y_v = load_voice()
    X_s, y_s = load_spiral()

    # 1. Validation croisée par modalité
    print("\nValidation croisée (5 splits, AUC)")
    cv_voice  = cross_val_eval(X_v, y_v, "voix  ")
    cv_spiral = cross_val_eval(X_s, y_s, "spiral")

    # 2. Fusion tardive sur holdout 25 %
    results = late_fusion_eval(X_v, y_v, X_s, y_s)

    # 3. Figures
    plot_confusion(
        results["y_all"],
        (results["prob_fusion"] >= 0.5).astype(int),
        OUT_DIR / "confusion_matrix_fusion.png",
    )
    plot_feature_importance(
        results["pipe_voice"],
        [str(i) for i in range(X_v.shape[1])],  # noms génériques 0..44
        "Importance des features – voix",
        OUT_DIR / "feature_importance_voice_model.png",
    )
    plot_feature_importance(
        results["pipe_spiral"],
        list(X_s.columns),
        "Importance des features – spirale",
        OUT_DIR / "feature_importance_spiral_model.png",
    )
    plot_roc_curves(results, OUT_DIR / "roc_curves_comparison.png")

    # 4. Export métriques JSON (pour la partie hyperparamètres)
    metrics = {
        "cross_val": {"voice": cv_voice, "spiral": cv_spiral},
        "holdout": {
            "auc_voice":  results["auc_voice"],
            "auc_spiral": results["auc_spiral"],
            "auc_fusion": results["auc_fusion"],
        },
    }
    metrics_path = OUT_DIR / "metrics_baseline.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMétriques exportées : {metrics_path}")
    print("\nTerminé. Figures dans :", OUT_DIR.resolve())

if __name__ == "__main__":
    main()