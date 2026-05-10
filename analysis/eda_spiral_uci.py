#!/usr/bin/env python3
"""
EDA pour le jeu UCI « Parkinson spiral / tablette » (id 395).

- Découverte des fichiers (hw_dataset + new_dataset pour les sujets PD).
- Vérification des colonnes et des blocs d’épreuve (SST / DST / STCP ou équivalent).
- Agrégation par sujet, visualisations, indication de variables utiles (corrélation / RF).
- Démos d’augmentation (tracé 2D et image spirale statique).

Sorties par défaut : ``reports/figures/spiral_uci/`` (voix : ``reports/figures/voice/``).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.ndimage import rotate as ndi_rotate
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/395/"
    "parkinson%2Bdisease%2Bspiral%2Bdrawings%2Busing%2Bdigitized%2Bgraphics%2Btablet.zip"
)

TEST_META = {
    0: ("sst", "Spirale statique (SST)"),
    1: ("dst", "Spirale dynamique (DST)"),
    2: ("stcp", "Stabilité / point fixe (STCP)"),
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Même racine que eda_voice.py (`reports/figures/voice/`) ; ce script écrit dans `spiral_uci/`.
FIGURES_BASE = PROJECT_ROOT / "reports" / "figures"
DEFAULT_SPIRAL_OUTPUT_DIR = FIGURES_BASE / "spiral_uci"


@dataclass(frozen=True)
class SubjectFile:
    label: int  # 1 = Parkinson, 0 = contrôle
    path: Path
    cohort: str


def download_and_extract(target_dir: Path, zip_path: Path | None = None) -> None:
    zip_path = zip_path or PROJECT_ROOT / "data" / "raw" / "spiral_uci.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"Téléchargement : {UCI_ZIP_URL}", file=sys.stderr)
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
    print(f"Extraction vers {target_dir}", file=sys.stderr)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def discover_subjects(data_root: Path) -> list[SubjectFile]:
    out: list[SubjectFile] = []
    ctrl = data_root / "hw_dataset" / "control"
    pd_hw = data_root / "hw_dataset" / "parkinson"
    pd_new = data_root / "new_dataset" / "parkinson"

    if not ctrl.is_dir() or not pd_hw.is_dir():
        raise FileNotFoundError(
            f"Dossiers attendus absents sous {data_root}. "
            "Lancez avec --download ou extrayez manuellement l’archive UCI."
        )

    for p in sorted(ctrl.glob("*.txt")):
        if p.name.lower() == "readme.txt":
            continue
        out.append(SubjectFile(0, p, "control"))

    for p in sorted(pd_hw.glob("*.txt")):
        out.append(SubjectFile(1, p, "pd_hw"))

    if pd_new.is_dir():
        for p in sorted(pd_new.glob("*.txt")):
            out.append(SubjectFile(1, p, "pd_new"))
    return out


def load_trajectory(path: Path) -> tuple[pd.DataFrame, list[str]]:
    cols = ["x", "y", "z", "pressure", "grip_angle", "timestamp", "test_id"]
    issues: list[str] = []
    raw = pd.read_csv(path, sep=";", header=None, names=cols, engine="python")
    for c in cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    n_bad = int(raw.isna().any(axis=1).sum())
    if n_bad:
        issues.append(f"{n_bad} lignes avec valeurs non numériques (supprimées)")
    df = raw.dropna(axis=0, how="any").copy()
    df["test_id"] = df["test_id"].astype(int)
    invalid_tid = ~df["test_id"].isin([0, 1, 2])
    if invalid_tid.any():
        issues.append(f"{int(invalid_tid.sum())} lignes avec test_id ∉ {{0,1,2}}")
        df = df.loc[~invalid_tid].copy()
    return df, issues


def contiguous_segments(df: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    tid = df["test_id"].to_numpy(dtype=int)
    if len(tid) == 0:
        return []
    change = np.r_[True, tid[1:] != tid[:-1]]
    starts = np.nonzero(change)[0]
    ends = np.r_[starts[1:], len(df)]
    merged: dict[int, pd.DataFrame] = {}
    order: list[int] = []
    for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
        sub = df.iloc[s:e].reset_index(drop=True)
        t = int(sub["test_id"].iloc[0])
        if t not in merged:
            merged[t] = sub
            order.append(t)
        else:
            merged[t] = pd.concat([merged[t], sub], ignore_index=True)
    return [(t, merged[t]) for t in order]


def segment_features(seg: pd.DataFrame) -> dict[str, float]:
    x = seg["x"].to_numpy(dtype=float)
    y = seg["y"].to_numpy(dtype=float)
    ts = seg["timestamp"].to_numpy(dtype=float)
    pr = seg["pressure"].to_numpy(dtype=float)
    gr = seg["grip_angle"].to_numpy(dtype=float)
    if len(seg) < 2:
        return {
            "n_points": float(len(seg)),
            "duration": float(np.nan),
            "path_length": float(0.0),
            "mean_speed": float(np.nan),
            "std_speed": float(np.nan),
            "pressure_mean": float(np.mean(pr)) if pr.size else np.nan,
            "pressure_std": float(np.std(pr)) if pr.size else np.nan,
            "grip_mean": float(np.mean(gr)) if gr.size else np.nan,
            "grip_std": float(np.std(gr)) if gr.size else np.nan,
            "bbox_w": float(x.max() - x.min()) if x.size else np.nan,
            "bbox_h": float(y.max() - y.min()) if y.size else np.nan,
        }

    dx = np.diff(x)
    dy = np.diff(y)
    step = np.sqrt(dx * dx + dy * dy)
    path_len = float(step.sum())
    dt = np.diff(ts).astype(float)
    dt = np.where(dt > 0, dt, np.nan)
    speed = step / dt
    speed = speed[np.isfinite(speed)]
    dur = float(ts.max() - ts.min())
    return {
        "n_points": float(len(seg)),
        "duration": dur,
        "path_length": path_len,
        "mean_speed": float(np.nanmean(speed)) if speed.size else np.nan,
        "std_speed": float(np.nanstd(speed)) if speed.size else np.nan,
        "pressure_mean": float(np.mean(pr)),
        "pressure_std": float(np.std(pr)),
        "grip_mean": float(np.mean(gr)),
        "grip_std": float(np.std(gr)),
        "bbox_w": float(x.max() - x.min()),
        "bbox_h": float(y.max() - y.min()),
    }


def trajectory_feature_keys() -> list[str]:
    dummy = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 1.2],
            "z": [0.0, 0.0],
            "pressure": [100.0, 105.0],
            "grip_angle": [900.0, 901.0],
            "timestamp": [0.0, 50.0],
            "test_id": [0, 0],
        }
    )
    return list(segment_features(dummy).keys())


def build_wide_table(subjects: list[SubjectFile]) -> tuple[pd.DataFrame, dict]:
    rows = []
    qc: dict = {"files": {}, "warnings": []}
    feat_keys = trajectory_feature_keys()

    for sf in subjects:
        df, issues = load_trajectory(sf.path)
        segs = {t: seg for t, seg in contiguous_segments(df)}
        present_tests = sorted(segs.keys())
        row = {
            "subject_id": sf.path.stem,
            "cohort": sf.cohort,
            "label_pd": sf.label,
            "path": str(sf.path.relative_to(PROJECT_ROOT)),
            "tests_present": ",".join(str(t) for t in present_tests),
        }
        for tid in (0, 1, 2):
            prefix = TEST_META[tid][0]
            feats = segment_features(segs[tid]) if tid in segs else {k: np.nan for k in feat_keys}
            for k, v in feats.items():
                row[f"{prefix}_{k}"] = v

        rows.append(row)
        qc_entry = {"issues": issues, "present_tests": present_tests, "label": sf.label}
        qc["files"][str(sf.path.relative_to(PROJECT_ROOT))] = qc_entry

    tbl = pd.DataFrame(rows)

    for tid, short, long in [(k, TEST_META[k][0], TEST_META[k][1]) for k in sorted(TEST_META)]:
        key = f"has_{short}"
        tbl[key] = tbl["tests_present"].apply(lambda s, t=tid: str(t) in s.split(",") if s else False)

    qc["missing_tests_counts"] = {
        short: int((~tbl[f"has_{short}"]).sum()) for short, _ in [TEST_META[k][:2] for k in TEST_META]
    }
    qc["n_subjects"] = len(tbl)
    qc["n_pd"] = int(tbl["label_pd"].sum())
    qc["n_hc"] = int((tbl["label_pd"] == 0).sum())
    qc["class_ratio_pd"] = float(tbl["label_pd"].mean())
    return tbl, qc


def plot_class_balance(tbl: pd.DataFrame, out: Path) -> None:
    counts = tbl["label_pd"].value_counts().rename({0: "Contrôle", 1: "Parkinson"})
    fig, ax = plt.subplots(figsize=(5, 3.8))
    counts.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518"])
    ax.set_title("Effectifs par groupe")
    ax.set_ylabel("Nombre de sujets")
    ax.set_xticklabels(list(counts.index), rotation=0)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_missing_tests(tbl: pd.DataFrame, out: Path) -> None:
    tests = [(TEST_META[t][1], f"has_{TEST_META[t][0]}") for t in sorted(TEST_META)]
    data = pd.DataFrame(
        {
            "Épreuve": [name for name, _ in tests],
            "Disponible": [int(tbl[col].sum()) for _, col in tests],
            "Manquant": [int((~tbl[col]).sum()) for _, col in tests],
        }
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bottom = np.zeros(len(data))
    colors = ["#54a24b", "#e45756"]
    ax.bar(data["Épreuve"], data["Disponible"], label="Disponible", color=colors[0])
    ax.bar(data["Épreuve"], data["Manquant"], bottom=data["Disponible"], label="Manquant", color=colors[1])
    ax.set_title("Complétude des épreuves par sujet (tous fichiers)")
    ax.set_ylabel("Nombre de sujets")
    ax.legend(handles=[Patch(color=colors[0], label="Au moins un segment"), Patch(color=colors[1], label="Absent")])
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_feature_box(tbl: pd.DataFrame, feat: str, title: str, out: Path) -> None:
    plot_df = tbl[["label_pd", feat]].dropna()
    plot_df["Groupe"] = plot_df["label_pd"].map({0: "Contrôle", 1: "Parkinson"})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=plot_df, x="Groupe", y=feat, ax=ax, hue="Groupe", dodge=False, legend=False)
    ax.set_title(title)
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def correlation_heatmap(X: pd.DataFrame, out: Path) -> None:
    corr = X.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="vlag",
        center=0,
        ax=ax,
        square=False,
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title("Corrélations (features numériques, valeurs imputées pour la figure)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def train_rf_importance(
    tbl: pd.DataFrame, out_csv: Path, out_bar: Path, random_state: int = 42
) -> dict:
    feature_cols = [c for c in tbl.columns if c.startswith(("sst_", "dst_", "stcp_"))]
    X = tbl[feature_cols]
    y = tbl["label_pd"].to_numpy()

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=random_state)),
        ]
    )

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    (tr, te), = sss.split(X, y)
    pipe.fit(X.iloc[tr], y[tr])
    proba = pipe.predict_proba(X.iloc[te])[:, 1]
    metrics = {
        "roc_auc_holdout": float(roc_auc_score(y[te], proba)),
        "n_train": int(tr.size),
        "n_test": int(te.size),
    }
    report = classification_report(y[te], pipe.predict(X.iloc[te]), digits=3, output_dict=True)
    metrics["classification_report"] = report

    full_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=random_state)),
        ]
    )
    full_pipe.fit(X, y)
    imp = full_pipe.named_steps["clf"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values("importance", ascending=False)
    imp_df.to_csv(out_csv, index=False)

    top = imp_df.head(18)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    sns.barplot(data=top, y="feature", x="importance", ax=ax, color="#4c78a8")
    ax.set_title("Importance des variables (forêt aléatoire, données complètes)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_bar, dpi=160)
    plt.close(fig)
    return metrics


def augment_trajectory_demo(tbl: pd.DataFrame, subjects: list[SubjectFile], out: Path) -> None:
    pd_files = [s for s in subjects if s.label == 1]
    if not pd_files:
        return
    path = pd_files[0].path
    df, _ = load_trajectory(path)
    segs = contiguous_segments(df)
    sst = next((seg for t, seg in segs if t == 0), None)
    if sst is None or len(sst) < 5:
        sst = df.iloc[:500].copy()
    x = sst["x"].to_numpy(dtype=float)
    y = sst["y"].to_numpy(dtype=float)
    xc, yc = x.mean(), y.mean()
    x0, y0 = x - xc, y - yc

    variants = []
    variants.append(("Original", x, y))

    rng = np.random.default_rng(0)

    def rotate(xv, yv, deg):
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        return c * xv - s * yv + xc, s * xv + c * yv + yc

    for deg in (-8.0, 12.0):
        xr, yr = rotate(x0, y0, deg)
        variants.append((f"Rotation {deg:+.0f}°", xr, yr))
    xr = x + rng.normal(0, 0.35, size=x.shape)
    yr = y + rng.normal(0, 0.35, size=y.shape)
    variants.append(("Bruit gaussien (xy)", xr, yr))

    fig, axes = plt.subplots(1, len(variants), figsize=(13, 3.2))
    if len(variants) == 1:
        axes = [axes]
    for ax, (name, xv, yv) in zip(axes, variants, strict=True):
        ax.plot(xv, yv, lw=1.2, color="#4c78a8")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Augmentation illustrative (tracé SST, sujet Parkinson)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def augment_image_demo(drawings_root: Path, out: Path) -> None:
    static = drawings_root / "Static Spiral Test"
    imgs = sorted(static.glob("*.png"))
    if not imgs:
        return
    img_path = imgs[0]
    from imageio.v3 import imread

    rgb = imread(img_path)
    gray = rgb.astype(float).mean(axis=2) if rgb.ndim == 3 else rgb.astype(float)

    rotated = ndi_rotate(gray, angle=15, reshape=False, order=1)
    jitter = np.clip(gray + np.random.default_rng(1).normal(0, 4.0, gray.shape), 0, 255)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    titles = ["Image originale", "Rotation 15°", "Bruit + clipping"]
    arrs = [gray, rotated, jitter]
    for ax, tit, arr in zip(axes, titles, arrs, strict=True):
        ax.imshow(arr, cmap="magma")
        ax.set_title(tit)
        ax.axis("off")
    fig.suptitle(f"Démo d’augmentation sur {img_path.name}")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA spirale Parkinson (UCI 395)")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/raw/spiral_uci")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SPIRAL_OUTPUT_DIR,
        help=f"Défaut: {DEFAULT_SPIRAL_OUTPUT_DIR} (eda_voice: {FIGURES_BASE / 'voice'})",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Si les dossiers sont absents, télécharge et extrait l’archive UCI dans data/raw/.",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    if args.download or not (data_root / "hw_dataset").is_dir():
        download_and_extract(data_root)

    subjects = discover_subjects(data_root)
    tbl, qc = build_wide_table(subjects)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl_csv = out_dir / "features_per_subject.csv"
    tbl.to_csv(tbl_csv, index=False)

    summary_path = out_dir / "qc_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2, ensure_ascii=False)

    plot_class_balance(tbl, out_dir / "01_effectifs_par_groupe.png")
    plot_missing_tests(tbl, out_dir / "02_completude_des_epreuves.png")
    plot_feature_box(
        tbl, "dst_path_length", "Longueur du tracé (DST) — sujets avec DST", out_dir / "03_dst_longueur_tracée.png"
    )
    plot_feature_box(
        tbl, "stcp_pressure_std", "Variabilité de la pression (STCP)", out_dir / "04_stcp_pression_ecart_type.png"
    )

    feature_cols = [c for c in tbl.columns if c.startswith(("sst_", "dst_", "stcp_"))]
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(tbl[feature_cols]), columns=feature_cols)
    correlation_heatmap(X_imp, out_dir / "05_corrélations_features.png")

    rf_metrics = train_rf_importance(tbl, out_dir / "feature_importance_rf.csv", out_dir / "06_importance_rf.png")

    augment_trajectory_demo(tbl, subjects, out_dir / "07_augmentation_trajets.png")
    drawings = data_root / "hw_drawings"
    if drawings.is_dir():
        augment_image_demo(drawings, out_dir / "08_augmentation_image_spirale.png")

    rf_path = out_dir / "metrics_rf_holdout.json"
    with rf_path.open("w", encoding="utf-8") as f:
        serializable = {k: v for k, v in rf_metrics.items() if k != "classification_report"}
        serializable["classification_report"] = rf_metrics["classification_report"]
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(json.dumps({"wrote": str(out_dir.resolve()), "n_subjects": qc["n_subjects"]}, indent=2))


if __name__ == "__main__":
    main()
