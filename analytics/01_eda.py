from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
OUTPUT_DIR = BASE_DIR / "outputs"
TITANIC_CSV = BASE_DIR / "titanic.csv"
RAW_SNAPSHOT = BASE_DIR / ".titanic_raw_snapshot.csv"
CLEAN_CSV = BASE_DIR / "cleaned_titanic.csv"
EDA_REPORT = BASE_DIR / "README.md"

PLOTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
sns.set_theme()


def load_raw_once() -> pd.DataFrame:
    """Load Titanic from seaborn exactly once for a fresh module run.

    The first run calls sns.load_dataset('titanic') and immediately saves titanic.csv.
    Later runs use the local file if it already exists, so they do not make a second
    network/cache load of the raw dataset.
    """
    if RAW_SNAPSHOT.exists():
        print("Raw Titanic snapshot already captured; using the local one-time snapshot.")
        return pd.read_csv(RAW_SNAPSHOT)

    print("Loading Titanic with sns.load_dataset('titanic')")
    df = sns.load_dataset("titanic")
    df.to_csv(TITANIC_CSV, index=False)
    df.to_csv(RAW_SNAPSHOT, index=False)
    print(f"Raw Titanic snapshot saved immediately to {TITANIC_CSV}")
    return df


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rates = (df.isna().mean() * 100).sort_values(ascending=False)
    report = (
        pd.DataFrame({"missing_count": df.isna().sum(), "missing_pct": rates})
        .query("missing_count > 0")
        .copy()
    )
    report["missing_pct"] = report["missing_pct"].round(2)
    report.to_csv(OUTPUT_DIR / "missing_value_report.csv")
    return report


def clean_by_threshold(df: pd.DataFrame, missing: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cleaned = df.copy()
    decisions = {}

    for col in missing.index:
        pct = float(missing.loc[col, "missing_pct"])
        if pct < 5:
            cleaned = cleaned.dropna(subset=[col])
            decisions[col] = f"{pct:.2f}% < 5%: drop rows with missing {col}."
        elif pct <= 30:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                median = cleaned[col].median()
                cleaned[col] = cleaned[col].fillna(median)
                decisions[col] = f"{pct:.2f}% in [5%, 30%]: median-impute {col} (median={median:.2f})."
            else:
                mode = cleaned[col].mode(dropna=True)
                fill = mode.iloc[0] if not mode.empty else "Missing"
                cleaned[col] = cleaned[col].fillna(fill)
                decisions[col] = f"{pct:.2f}% in [5%, 30%]: mode-impute {col} (mode={fill})."
        else:
            cleaned = cleaned.drop(columns=[col])
            decisions[col] = f"{pct:.2f}% > 30%: drop column {col} because imputation would be unreliable."

    return cleaned, decisions


def iqr_outliers(series: pd.Series) -> tuple[float, float, int]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())
    return lower, upper, count


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(
    raw_df: pd.DataFrame,
    missing: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    decisions: dict,
    outlier_counts: dict,
    fare_stats: dict,
    corr_pairs: list[tuple[str, str, float]],
    survival_by_sex: pd.Series,
    survival_by_pclass: pd.Series,
    survival_by_sex_pclass: pd.Series,
    standardization_summary: pd.DataFrame,
) -> None:
    report = [
        "# Module 2 — Analytics Pipeline",
        "",
        "## Run order",
        "",
        "1. `python analytics/01_eda.py` loads Titanic once with `sns.load_dataset('titanic')`, saves `titanic.csv` immediately, profiles it, applies the missing-value threshold rule, performs EDA, and saves `cleaned_titanic.csv`.",
        "2. `python analytics/02_modeling.py` reads `cleaned_titanic.csv` with `pd.read_csv` and continues the same cleaned-data story into classification, imbalance handling, tuning, regression, and artifact saving.",
        "",
        "## Dataset profile",
        "",
        f"Raw shape: **{raw_df.shape}**.",
        "",
        "### Missing values and threshold decisions",
        "",
        "| Column | Missing count | Missing % | Decision |",
        "|---|---:|---:|---|",
    ]
    for col in missing.index:
        report.append(
            f"| {col} | {int(missing.loc[col, 'missing_count'])} | {missing.loc[col, 'missing_pct']:.2f}% | {decisions[col]} |"
        )

    report += [
        "",
        "High-missing `deck` is dropped because its missingness exceeds 30%, while `age` is median-imputed because its missingness lies between 5% and 30%. Columns below 5% missingness are handled by dropping the affected rows, exactly following the project threshold rule.",
        "",
        "## Univariate analysis",
        "",
        f"Age IQR outlier count: **{outlier_counts['age']}**.",
        f"Fare IQR outlier count: **{outlier_counts['fare']}**.",
        f"Fare mean = **{fare_stats['mean']:.3f}**, median = **{fare_stats['median']:.3f}**, mode = **{fare_stats['mode']:.3f}**. Because mean > median > mode, fare is interpreted as **right-skewed** rather than symmetric.",
        "",
        "![Age histogram](plots/age_histogram.png)",
        "",
        "![Age box plot](plots/age_boxplot.png)",
        "",
        "![Fare histogram](plots/fare_histogram.png)",
        "",
        "![Fare box plot](plots/fare_boxplot.png)",
        "",
        "## Bivariate analysis",
        "",
        "### Survival by sex",
        "",
        survival_by_sex.rename("survival_rate").to_frame().assign(survival_rate=lambda x: x.survival_rate.round(4)).to_markdown(),
        "",
        "The sex breakdown shows a substantial difference in observed survival rate between the two sex groups. The calculation uses boolean masking and then divides survivors by group size, so the reported values are proportions rather than raw counts.",
        "",
        "### Survival by passenger class",
        "",
        survival_by_pclass.rename("survival_rate").to_frame().assign(survival_rate=lambda x: x.survival_rate.round(4)).to_markdown(),
        "",
        "Passenger class is associated with survival in the cleaned data, with the rates differing across first, second, and third class. This is descriptive evidence of an association, not a causal claim.",
        "",
        "### Survival by sex and passenger class",
        "",
        survival_by_sex_pclass.rename("survival_rate").to_frame().assign(survival_rate=lambda x: x.survival_rate.round(4)).to_markdown(),
        "",
        "The joint breakdown shows that sex differences persist within passenger classes, while passenger class also changes the survival rate inside each sex group. The combination is more informative than either single-variable breakdown because it exposes the interaction-like pattern directly.",
        "",
        "### Correlation matrix",
        "",
        "The heatmap uses exactly `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`. Boolean-derived flags such as `adult_male` and `alone` are intentionally excluded because the project specifies that they are redundant flags rather than independent measured features.",
        "",
        "![Correlation heatmap](plots/correlation_heatmap.png)",
        "",
        "The two strongest off-diagonal correlations by absolute magnitude are:",
    ]
    for a, b, value in corr_pairs:
        report.append(f"- `{a}` vs `{b}`: correlation = **{value:.4f}**.")

    report += [
        "",
        "The sign indicates direction while the absolute value determines the requested ranking. These correlations describe pairwise linear association only and do not establish causality.",
        "",
        "## Multivariate data story",
        "",
        "### Chart 1 — Survival rate by sex",
        "",
        "![Survival by sex](plots/survival_by_sex.png)",
        "",
        "The bar chart makes the sex-based survival difference visible at a glance. The group percentages complement the bivariate table and provide a clear first part of the survival story.",
        "",
        "### Chart 2 — Survival rate by passenger class",
        "",
        "![Survival by class](plots/survival_by_pclass.png)",
        "",
        "Survival rates vary across passenger classes, showing that class was an important descriptive feature in this dataset. The visual makes the class gradient easier to compare than the raw table alone.",
        "",
        "### Chart 3 — Survival by sex and class",
        "",
        "![Survival by sex and class](plots/survival_by_sex_pclass.png)",
        "",
        "The grouped bars show how sex and passenger class jointly relate to survival. Within-class comparisons and within-sex comparisons can both be read directly from the same chart.",
        "",
        "### Chart 4 — Fare distribution by survival",
        "",
        "![Fare by survival](plots/fare_by_survival.png)",
        "",
        "The fare box plot compares the fare distributions for survivors and non-survivors. It helps connect monetary/class-related information to survival without treating fare as the sole explanation for the outcome.",
        "",
        "### Chart 5 — Age vs fare by survival",
        "",
        "![Age vs fare](plots/age_fare_survival.png)",
        "",
        "The scatter plot combines age and fare with survival as the visual grouping variable. It shows overlap between groups while also making the class/fare structure of the dataset visible.",
        "",
        "## Exploratory standardization sanity check",
        "",
        "Age and fare are standardized only for this EDA-stage check; these transformed columns are not reused as model inputs. The model section performs its own train-only preprocessing through scikit-learn pipelines.",
        "",
        standardization_summary.to_markdown(index=False),
        "",
        "![Standardized age and fare](plots/standardization_check.png)",
        "",
        "## Interpretation and modeling continuation",
        "",
        "The EDA establishes the data-quality decisions and descriptive patterns before any predictive model is fitted. The next step is to use a stratified train/test split and fit every preprocessing object only on training data through a `Pipeline`/`ColumnTransformer`, preventing test-set leakage as required by the project."
    ]

    EDA_REPORT.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    df = load_raw_once()

    print("\n=== RAW PROFILE ===")
    df.info()
    print(df.describe(include="all"))
    print("shape:", df.shape)

    missing = missing_report(df)
    print("\n=== MISSING VALUE REPORT ===")
    print(missing)

    cleaned, decisions = clean_by_threshold(df, missing)
    CLEAN_CSV.write_text(cleaned.to_csv(index=False), encoding="utf-8")
    TITANIC_CSV.write_text(cleaned.to_csv(index=False), encoding="utf-8")
    print("\n=== CLEANING DECISIONS ===")
    for col, decision in decisions.items():
        print(f"{col}: {decision}")
    print("Cleaned shape:", cleaned.shape)

    cleaned["survived"] = cleaned["survived"].astype(int)

    for col in ["age", "fare"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(cleaned[col], kde=True, ax=ax)
        ax.set_title(f"{col.title()} distribution")
        save_plot(fig, PLOTS_DIR / f"{col}_histogram.png")

        fig, ax = plt.subplots(figsize=(7, 3))
        sns.boxplot(x=cleaned[col], ax=ax)
        ax.set_title(f"{col.title()} box plot")
        save_plot(fig, PLOTS_DIR / f"{col}_boxplot.png")

    age_low, age_high, age_outliers = iqr_outliers(cleaned["age"])
    fare_low, fare_high, fare_outliers = iqr_outliers(cleaned["fare"])
    print(f"Age IQR fences: [{age_low:.3f}, {age_high:.3f}], outliers={age_outliers}")
    print(f"Fare IQR fences: [{fare_low:.3f}, {fare_high:.3f}], outliers={fare_outliers}")

    fare_mode = float(cleaned["fare"].mode().iloc[0])
    fare_stats = {
        "mean": float(cleaned["fare"].mean()),
        "median": float(cleaned["fare"].median()),
        "mode": fare_mode,
    }
    print("Fare mean/median/mode:", fare_stats)

    sexes = cleaned["sex"].dropna().unique()
    survival_by_sex = pd.Series(
        {sex: cleaned.loc[cleaned["sex"] == sex, "survived"].mean() for sex in sexes}
    ).sort_index()

    classes = sorted(cleaned["pclass"].dropna().unique())
    survival_by_pclass = pd.Series(
        {int(p): cleaned.loc[cleaned["pclass"] == p, "survived"].mean() for p in classes}
    ).sort_index()

    pairs = []
    for sex in sorted(cleaned["sex"].dropna().unique()):
        for pclass in classes:
            mask = (cleaned["sex"] == sex) & (cleaned["pclass"] == pclass)
            if mask.any():
                pairs.append(((sex, int(pclass)), cleaned.loc[mask, "survived"].mean()))
    survival_by_sex_pclass = pd.Series(dict(pairs)).sort_index()

    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = cleaned[corr_cols].corr()
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Titanic correlation matrix — required six columns")
    save_plot(fig, PLOTS_DIR / "correlation_heatmap.png")

    pairs_abs = []
    for i, a in enumerate(corr_cols):
        for b in corr_cols[i + 1 :]:
            pairs_abs.append((a, b, float(corr.loc[a, b])))
    pairs_abs.sort(key=lambda x: abs(x[2]), reverse=True)
    strongest = pairs_abs[:2]
    print("Two strongest absolute off-diagonal correlations:")
    for item in strongest:
        print(item)

    sex_plot = survival_by_sex.reset_index()
    sex_plot.columns = ["sex", "survival_rate"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=sex_plot, x="sex", y="survival_rate", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival rate by sex")
    save_plot(fig, PLOTS_DIR / "survival_by_sex.png")

    class_plot = survival_by_pclass.reset_index()
    class_plot.columns = ["pclass", "survival_rate"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=class_plot, x="pclass", y="survival_rate", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Passenger class")
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival rate by passenger class")
    save_plot(fig, PLOTS_DIR / "survival_by_pclass.png")

    joint_plot = survival_by_sex_pclass.rename("survival_rate").reset_index()
    joint_plot.columns = ["sex", "pclass", "survival_rate"]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=joint_plot, x="pclass", y="survival_rate", hue="sex", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Passenger class")
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival rate by sex and passenger class")
    save_plot(fig, PLOTS_DIR / "survival_by_sex_pclass.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=cleaned, x="survived", y="fare", ax=ax)
    ax.set_xticklabels(["Did not survive", "Survived"])
    ax.set_xlabel("Outcome")
    ax.set_title("Fare distribution by survival")
    save_plot(fig, PLOTS_DIR / "fare_by_survival.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=cleaned, x="age", y="fare", hue="survived", alpha=0.65, ax=ax)
    ax.set_title("Age vs fare by survival")
    save_plot(fig, PLOTS_DIR / "age_fare_survival.png")

    scaler = StandardScaler()
    standardized = scaler.fit_transform(cleaned[["age", "fare"]])
    standardized_df = pd.DataFrame(standardized, columns=["age_z", "fare_z"])
    standardization_summary = pd.DataFrame(
        {
            "column": ["age_z", "fare_z"],
            "mean_after": [standardized_df[c].mean() for c in standardized_df.columns],
            "std_after_ddof0": [standardized_df[c].std(ddof=0) for c in standardized_df.columns],
        }
    )
    standardization_summary.to_csv(OUTPUT_DIR / "standardization_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(standardized_df["age_z"], kde=True, ax=axes[0])
    axes[0].set_title("Standardized age")
    sns.histplot(standardized_df["fare_z"], kde=True, ax=axes[1])
    axes[1].set_title("Standardized fare")
    save_plot(fig, PLOTS_DIR / "standardization_check.png")

    write_report(
        raw_df=df,
        missing=missing,
        cleaned_df=cleaned,
        decisions=decisions,
        outlier_counts={"age": age_outliers, "fare": fare_outliers},
        fare_stats=fare_stats,
        corr_pairs=strongest,
        survival_by_sex=survival_by_sex,
        survival_by_pclass=survival_by_pclass,
        survival_by_sex_pclass=survival_by_sex_pclass,
        standardization_summary=standardization_summary,
    )

    print("EDA complete. Continue with: python analytics/02_modeling.py")


if __name__ == "__main__":
    main()
