from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from ucimlrepo import fetch_ucirepo


OUTPUT_DIR = Path("reports") / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def audit_dataframe(df: pd.DataFrame, label: str, check_duplicates: bool = True) -> None:
    print(f"\n=== {label} ===")
    print("shape:", df.shape)
    print("missing values:\n", df.isna().sum().sort_values(ascending=False).head(10))
    if check_duplicates:
        print("duplicates:", df.duplicated().sum())
    else:
        print("duplicates: not checked on this view")
    print("dtypes:\n", df.dtypes.value_counts())
    print("summary:\n", df.describe(include="all").transpose().head(10))


def save_missingness_plot(df: pd.DataFrame, filename: str) -> None:
    missing_rate = df.isna().mean().sort_values(ascending=False)
    if missing_rate.sum() == 0:
        print("No missing values found; skipping missingness plot.")
        return

    plt.figure(figsize=(10, 5))
    missing_rate[missing_rate > 0].plot(kind="bar", color="#c96e6e")
    plt.ylabel("Missing rate")
    plt.title("Missing values by column")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame, filename: str) -> None:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        print("Not enough numeric columns for correlation heatmap.")
        return

    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def save_target_relationships(df: pd.DataFrame, target_columns: list[str]) -> None:
    for target in target_columns:
        if target not in df.columns:
            continue

        correlations = (
            df.select_dtypes(include="number")
            .corr(numeric_only=True)[target]
            .drop(target)
            .abs()
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(10, 6))
        correlations.head(10).sort_values().plot(kind="barh", color="#4c78a8")
        plt.title(f"Top features correlated with {target}")
        plt.xlabel("Absolute correlation")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"top_correlations_{target}.png", dpi=160)
        plt.close()


def save_feature_importance(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        print(f"Target {target} not found; skipping feature importance.")
        return

    numeric_df = df.select_dtypes(include="number").dropna().copy()
    if target not in numeric_df.columns:
        print(f"Target {target} is not numeric; skipping feature importance.")
        return

    features = numeric_df.drop(columns=[target])
    labels = numeric_df[target]

    if features.shape[1] == 0:
        print("No numeric features available for feature importance.")
        return

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    importance = permutation_importance(
        model, x_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    ranked = (
        pd.Series(importance.importances_mean, index=features.columns)
        .sort_values(ascending=False)
        .head(12)
    )

    plt.figure(figsize=(10, 6))
    ranked.sort_values().plot(kind="barh", color="#72b7b2")
    plt.title(f"Permutation importance for {target}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"feature_importance_{target}.png", dpi=160)
    plt.close()


def main() -> None:
    dataset = fetch_ucirepo(id=189)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    df = pd.concat([features, targets], axis=1)

    print("Dataset:", dataset.metadata["name"])
    audit_dataframe(df, "full dataset")
    audit_dataframe(features, "features")
    audit_dataframe(targets, "targets", check_duplicates=False)

    complete_rows = df.dropna().shape[0]
    print(f"\nComplete rows: {complete_rows}/{len(df)}")
    print(f"Missing rows: {len(df) - complete_rows}")

    if df.isna().sum().sum() == 0:
        print("Dataset is complete: no missing values detected.")
    else:
        print("Dataset has missing values; review imputation before training.")

    save_missingness_plot(df, "missingness_voice.png")
    save_correlation_heatmap(df, "correlation_voice.png")
    save_target_relationships(df, ["motor_UPDRS", "total_UPDRS"])
    save_feature_importance(df, "motor_UPDRS")

    print(f"\nFigures saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()