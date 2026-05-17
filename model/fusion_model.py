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
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    PrecisionRecallDisplay,
    matthews_corrcoef, 
    average_precision_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from sklearn.calibration import CalibrationDisplay


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

    print(f"Dataset vocal  : {X_train.shape[0]} train, {X_test.shape[0]} test (total {X_train.shape[0] + X_test.shape[0]})")
    print(f"Features : {X_train.shape[1]}")
    print(f"Train - PD={y_train.sum()}, HC={(y_train==0).sum()}")
    print(f"Test  - PD={y_test.sum()}, HC={(y_test==0).sum()}")

    return (X_train, X_test, y_train, y_test)

    #X = np.vstack([X_train, X_test])
    #y = np.concatenate([y_train, y_test])
    #print(f"Dataset vocal  : {X.shape[0]} sujets, {X.shape[1]} features — PD={y.sum()}, HC={(y==0).sum()}")
    #return pd.DataFrame(X), y

def load_spiral() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Charge le CSV produit par eda_spiral_uci.py et effectue un split train/test.
    
    Note: Contrairement à load_voice() qui reçoit déjà les splits du fichier merged,
    ici on doit splitter nous-même avec stratification.
    """
    if not SPIRAL_CSV.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {SPIRAL_CSV}\n"
            "Lance d'abord : python analysis/eda_spiral_uci.py"
        )
    
    df = pd.read_csv(SPIRAL_CSV)
    feat_cols = [c for c in df.columns if c.startswith(("sst_", "dst_", "stcp_"))]
    X = df[feat_cols]
    y = df["label_pd"].to_numpy()
    
    # Split train/test avec stratification (même ratio que voix 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset spiral : {X_train.shape[0]} train, {X_test.shape[0]} test (total {X_train.shape[0] + X_test.shape[0]})")
    print(f"Features : {X_train.shape[1]}")
    print(f"Train - PD={y_train.sum()}, HC={(y_train==0).sum()}")
    print(f"Test  - PD={y_test.sum()}, HC={(y_test==0).sum()}")
    
    return (X_train, X_test, y_train, y_test)

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

    # Chaque fold conserve le ration PD/HC
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        'roc_auc': 'roc_auc',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'accuracy': 'accuracy',
    }

    cv_results = cross_validate(
        pipe, X, y, 
        cv=cv, 
        scoring=scoring,
        return_train_score=False
    )

    print(f"CROSS-VALIDATION 5-FOLD : {label}")
    for metric in scoring.keys():
        scores = cv_results[f'test_{metric}']
        print(f"{metric:<12}: {scores.mean():.3f} ± {scores.std():.3f}  "
              f"(min={scores.min():.3f}, max={scores.max():.3f})")
    
    return {
        'roc_auc_mean': float(cv_results['test_roc_auc'].mean()),
        'roc_auc_std': float(cv_results['test_roc_auc'].std()),
        'precision_mean': float(cv_results['test_precision'].mean()),
        'recall_mean': float(cv_results['test_recall'].mean()),
        'f1_mean': float(cv_results['test_f1'].mean()),
        'accuracy_mean': float(cv_results['test_accuracy'].mean()),
    }

# Grid search 
def grid_search_eval(X_train: pd.DataFrame, y_train: np.ndarray, label: str) -> dict:
    """
    Grid Search avec 5-Fold CV pour optimiser hyperparams.
    
    Objectif: Tester combinaisons de paramètres et retourner meilleur modèle.
    
    Combinaisons testées: 3 × 3 × 3 = 27 modèles × 5-fold = 135 entraînements
    """
    param_grid = {
        'clf__n_estimators': [100, 300, 500],
        'clf__max_depth': [10, 20, None],
        'clf__min_samples_split': [2, 5, 10],
    }

    pipe = make_pipe()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,           # Paralléliser
        verbose=1
    )

    print(f"GRID SEARCH : {label}")
    n_combos = len(param_grid['clf__n_estimators']) * len(param_grid['clf__max_depth']) * len(param_grid['clf__min_samples_split'])
    print(f"Testant {n_combos} combinaisons × 5-fold CV...")
    
    gs.fit(X_train, y_train)
    
    # Afficher résultats
    print(f"\nMeilleurs paramètres : {gs.best_params_}")
    print(f"Score CV moyen : {gs.best_score_:.3f}")

    # Top 5 combinaisons
    cv_df = pd.DataFrame(gs.cv_results_).sort_values('rank_test_score')
    print(f"\nTop 5 combinaisons :")
    cols_to_show = ['param_clf__n_estimators', 'param_clf__max_depth', 
                    'param_clf__min_samples_split', 'mean_test_score', 'std_test_score']
    print(cv_df[cols_to_show].head(5).to_string(index=False))

    return {
        'best_params': gs.best_params_,
        'best_score': float(gs.best_score_),
        'best_model': gs.best_estimator_,
    }

# Evalusation complete

def complete_eval(
    X_train: pd.DataFrame, X_test: pd.DataFrame, 
    y_train: np.ndarray, y_test: np.ndarray, 
    label: str
) -> dict:
    """
    Workflow complet :
    1. Grid Search sur TRAIN → trouve meilleurs params
    2. Évalue sur TEST → résultats finaux honnêtes
    3. CV stats → fiabilité du modèle

    Aucun leakage : Grid Search cherche que sur TRAIN
    Test holdout séparé : jamais vu par Grid Search
    CV stats : variance du modèle
    """
    print(f"ÉVALUATION COMPLÈTE : {label}")

     #Grid Search sur train uniquement
    gs_results = grid_search_eval(X_train, y_train, label)
    best_pipe = gs_results['best_model']

    #Évaluer sur test holdout (jamais vu par Grid Search)
    prob_test = best_pipe.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, prob_test)
    pr_auc_test = average_precision_score(y_test, prob_test)

    print(f"RÉSULTATS TEST (holdout final)")
    print(f"AUC test     : {auc_test:.3f}")
    print(f"PR-AUC test  : {pr_auc_test:.3f}")
    print("\n" + classification_report(y_test, (prob_test >= 0.5).astype(int),
                                target_names=["Contrôle", "Parkinson"], digits=3))
    
    #CV stats sur train (pour savoir la variance)
    cv_stats = cross_val_eval(X_train, y_train, f"{label} (CV sur TRAIN)")

    # Resume 
    print(f"RÉSUMÉ : {label}")
    print(f"CV AUC moyen  : {cv_stats['roc_auc_mean']:.3f} ± {cv_stats['roc_auc_std']:.3f}")
    print(f"Test AUC      : {auc_test:.3f}")
    gap = auc_test - cv_stats['roc_auc_mean']
    print(f"Gap CV→Test   : {gap:+.3f} {'(faible = bon)' if abs(gap) < 0.1 else '(grand = possible overfitting)'}")

    return {
        # Grid Search results
        'best_params': gs_results['best_params'],
        'best_cv_score': gs_results['best_score'],
        
        # CV stats
        'cv_roc_auc_mean': cv_stats['roc_auc_mean'],
        'cv_roc_auc_std': cv_stats['roc_auc_std'],
        'cv_precision_mean': cv_stats['precision_mean'],
        'cv_recall_mean': cv_stats['recall_mean'],
        'cv_f1_mean': cv_stats['f1_mean'],
        'cv_accuracy_mean': cv_stats['accuracy_mean'],
        
        # Test holdout
        'auc_test': float(auc_test),
        'pr_auc_test': float(pr_auc_test),
        
        # Model & predictions
        'pipe': best_pipe,
        'prob_test': prob_test,
        'y_test': y_test,
    }
    


# Late fusion 
def late_fusion_eval(
    data_v: tuple,
    data_s: tuple,
) -> dict:
    """
    Évalue la fusion tardive sur les splits train/test pré-existants.

    IMPORTANT: Voix et Spirale ont des datasets de tailles différentes.
    On fusionne uniquement sur les sujets présents dans BOTH modalities.

    Args:
        data_v: (X_train, X_test, y_train, y_test) pour voix
        data_s: (X_train, X_test, y_train, y_test) pour spirale
    
    Note : Utilise directement les splits pré-existants (aucun re-split)
    pour éviter le data leakage.
    """
    X_train_v, X_test_v, y_train_v, y_test_v = data_v
    X_train_s, X_test_s, y_train_s, y_test_s = data_s

    #Resultat complet sur Voix et Spirale
    results_v = complete_eval(X_train_v, X_test_v, y_train_v, y_test_v, "VOIX")
    results_s = complete_eval(X_train_s, X_test_s, y_train_s, y_test_s, "SPIRALE")

    print(f"COMPARAISON FINALE : VOIX vs SPIRALE")
  
    print(f"{'Métrique':<20} {'VOIX':<20} {'SPIRALE':<20}")
 
    print(f"{'CV AUC':<20} {results_v['cv_roc_auc_mean']:.3f} ± {results_v['cv_roc_auc_std']:.3f}{'':<3} "
          f"{results_s['cv_roc_auc_mean']:.3f} ± {results_s['cv_roc_auc_std']:.3f}")
    print(f"{'Test AUC':<20} {results_v['auc_test']:.3f}{'':<15} {results_s['auc_test']:.3f}")
    print(f"{'Test PR-AUC':<20} {results_v['pr_auc_test']:.3f}{'':<15} {results_s['pr_auc_test']:.3f}")
    print(f"{'CV Precision':<20} {results_v['cv_precision_mean']:.3f}{'':<15} {results_s['cv_precision_mean']:.3f}")
    print(f"{'CV Recall':<20} {results_v['cv_recall_mean']:.3f}{'':<15} {results_s['cv_recall_mean']:.3f}")


    return {
        'voix': results_v,
        'spirale': results_s,
        # Raw predictions pour plots
        'prob_v': results_v['prob_test'],
        'y_test_v': results_v['y_test'],
        'prob_s': results_s['prob_test'],
        'y_test_s': results_s['y_test'],
    }

# Visualisations
def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out: Path, title: str = "Matrice de confusion") -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax,
        display_labels=["Contrôle", "Parkinson"],
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Sauvegardé : {out}")

def plot_roc_curves(results: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_predictions(
        results["y_test_v"], results["prob_v"], ax=ax, 
        name=f"Voix (AUC={roc_auc_score(results['y_test_v'], results['prob_v']):.3f})")
    
    RocCurveDisplay.from_predictions(
        results["y_test_s"], results["prob_s"], ax=ax, 
        name=f"Spirale (AUC={roc_auc_score(results['y_test_s'], results['prob_s']):.3f})")
    
    ax.set_title("Courbes ROC – Voix vs Spirale")
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

def plot_precision_recall(results, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        results["y_test_v"], results["prob_v"], ax=ax, 
        name=f"Voix (PR-AUC={average_precision_score(results['y_test_v'], results['prob_v']):.3f})")
    
    PrecisionRecallDisplay.from_predictions(
        results["y_test_s"], results["prob_s"], ax=ax, 
        name=f"Spirale (PR-AUC={average_precision_score(results['y_test_s'], results['prob_s']):.3f})")
    
    ax.set_title("Courbe Precision-Recall")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Sauvegardé : {out}")
    
def plot_calibration(results, out):
    fig, ax = plt.subplots(figsize=(6,5))
    CalibrationDisplay.from_predictions(
        results["y_test_v"], results["prob_v"],
        n_bins=10, ax=ax, name="Voix")
    
    CalibrationDisplay.from_predictions(
        results["y_test_s"], results["prob_s"],
        n_bins=5, ax=ax, name="Spirale (n_bins=5 car peu d'échantillons)")

    ax.set_title("Courves de calibration")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Sauvegardé : {out}")

# Main
def main() -> None:
    print("Fusion multimodale - Détection Parkinson")

    # Charger les données 
    data_v = load_voice()  
    data_s = load_spiral() 

    # 1. Fusion tardive sur holdout 25%
    results = late_fusion_eval(data_v, data_s)

    # 2. Figures
    plot_confusion(
        results["y_test_v"],
        (results["prob_v"] >= 0.5).astype(int),
        OUT_DIR / "confusion_matrix_voice.png",
        title="Matrice de confusion – Voix"
    )

    plot_confusion(
        results["y_test_s"],
        (results["prob_s"] >= 0.5).astype(int),
        OUT_DIR / "confusion_matrix_spiral.png",
        title="Matrice de confusion – Spirale"
    )


    plot_feature_importance(
        results['voix']['pipe'],
        [str(i) for i in range(data_v[0].shape[1])],
        "Importance des features – Voix",
        OUT_DIR / "feature_importance_voice_model.png",
    )

    plot_feature_importance(
        results["spirale"]["pipe"],
        list(data_s[0].columns),
        "Importance des features – Spirale",
        OUT_DIR / "feature_importance_spiral_model.png",
    )


    plot_roc_curves(results, OUT_DIR / "roc_curves_comparison.png")
    plot_precision_recall(results, OUT_DIR / "precision_recall_curves.png")
    plot_calibration(results, OUT_DIR / "calibration_curves.png")

    # 3. Export métriques JSON
    metrics = {
        "voix": {
            "best_params": str(results['voix']['best_params']),
            "best_cv_score": results['voix']['best_cv_score'],
            "cv_roc_auc": f"{results['voix']['cv_roc_auc_mean']:.3f} ± {results['voix']['cv_roc_auc_std']:.3f}",
            "test_auc": results['voix']['auc_test'],
            "test_pr_auc": results['voix']['pr_auc_test'],
            "cv_precision": results['voix']['cv_precision_mean'],
            "cv_recall": results['voix']['cv_recall_mean'],
        },

        "spirale": {
            "best_params": str(results['spirale']['best_params']),
            "best_cv_score": results['spirale']['best_cv_score'],
            "cv_roc_auc": f"{results['spirale']['cv_roc_auc_mean']:.3f} ± {results['spirale']['cv_roc_auc_std']:.3f}",
            "test_auc": results['spirale']['auc_test'],
            "test_pr_auc": results['spirale']['pr_auc_test'],
            "cv_precision": results['spirale']['cv_precision_mean'],
            "cv_recall": results['spirale']['cv_recall_mean'],
        },
    }

    metrics_path = OUT_DIR / "metrics_baseline.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMétriques exportées : {metrics_path}")

    # 4. Sauvegarde du modèle
    joblib.dump({
        "pipe_voice":  results['voix']['pipe'],
        "pipe_spiral": results['spirale']['pipe'],
    }, OUT_DIR / "model_fusion.joblib")

    print("Modèle sauvegardé : model_fusion.joblib")

    with open(OUT_DIR / "model_fusion.pkl", "wb") as f:
        pickle.dump({
            "pipe_voice":  results['voix']['pipe'],
            "pipe_spiral": results['spirale']['pipe'],
        }, f)

    print("Modèle sauvegardé : model_fusion.pkl")
    print("\nTerminé. Figures dans :", OUT_DIR.resolve())

if __name__ == "__main__":
    main()