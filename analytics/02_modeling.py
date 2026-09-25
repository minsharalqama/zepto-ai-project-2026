from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

BASE_DIR = Path(__file__).resolve().parent
CLEAN_CSV = BASE_DIR / "titanic.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = BASE_DIR / "plots"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
README = BASE_DIR / "README.md"

OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_CLASSIFICATION = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_CLASSIFICATION = ["sex", "embarked"]
CLASSIFICATION_FEATURES = NUMERIC_CLASSIFICATION + CATEGORICAL_CLASSIFICATION

REGRESSION_NUMERIC = ["survived", "pclass", "age", "sibsp", "parch"]
REGRESSION_CATEGORICAL = ["sex", "embarked"]
REGRESSION_FEATURES = REGRESSION_NUMERIC + REGRESSION_CATEGORICAL


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def evaluate_classifier(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def save_classifier_plots(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, pipeline) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "classifier_confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipeline in models.items():
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "classifier_roc_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_decision_tree_plot(tree_pipeline: Pipeline) -> None:
    preprocessor = tree_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    tree = tree_pipeline.named_steps["classifier"]
    fig, ax = plt.subplots(figsize=(22, 12))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=["Not survived", "Survived"],
        filled=False,
        max_depth=4,
        fontsize=7,
        ax=ax,
    )
    ax.set_title("Decision Tree (first 4 levels shown for readability)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "decision_tree.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_model_report(
    cleaned: pd.DataFrame,
    train_class_balance: pd.Series,
    classifier_metrics: pd.DataFrame,
    imbalance_df: pd.DataFrame,
    grid: GridSearchCV,
    regression_metrics: dict,
    hetero_flag: bool,
    best_name: str,
    best_metric_row: pd.Series,
    reload_prediction: int,
) -> None:
    recommendation = (
        f"The classifier selected for deployment is **{best_name}** because it has the highest test-set F1 score among the three baseline classifiers in this run ({best_metric_row['f1']:.3f}). "
        f"Its accuracy is {best_metric_row['accuracy']:.3f}, precision is {best_metric_row['precision']:.3f}, recall is {best_metric_row['recall']:.3f}, and AUC is {best_metric_row['auc']:.3f}. "
        "This choice is based on the observed held-out metrics rather than a single training score. "
        "The saved artifact is the complete fitted pipeline, so its preprocessing and estimator can be applied together to raw rows at inference time."
    )

    base = README.read_text(encoding="utf-8") if README.exists() else "# Module 2 — Analytics Pipeline\n"
    comparison_table = classifier_metrics[["model", "accuracy", "precision", "recall", "f1", "auc"]].copy()
    comparison_table["MAE"] = np.nan
    comparison_table["RMSE"] = np.nan
    comparison_table["R2"] = np.nan
    comparison_table["Adjusted_R2"] = np.nan
    regression_row = pd.DataFrame([{
        "model": "Linear Regression (fare)",
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "auc": np.nan,
        "MAE": regression_metrics["mae"],
        "RMSE": regression_metrics["rmse"],
        "R2": regression_metrics["r2"],
        "Adjusted_R2": regression_metrics["adjusted_r2"],
    }])
    combined_comparison = pd.concat([comparison_table, regression_row], ignore_index=True)
    combined_comparison["model_type"] = ["classification"] * len(classifier_metrics) + ["regression"]
    combined_comparison = combined_comparison[["model_type", "model", "accuracy", "precision", "recall", "f1", "auc", "MAE", "RMSE", "R2", "Adjusted_R2"]]
    combined_comparison.to_csv(OUTPUT_DIR / "model_comparison_all_metrics.csv", index=False)

    section = [
        "",
        "# Predictive Modeling Results",
        "",
        "## Train/test split and leakage control",
        "",
        f"The cleaned dataset contains **{len(cleaned)} rows**. A stratified {1-TEST_SIZE:.0%}/{TEST_SIZE:.0%} train/test split with `random_state={RANDOM_STATE}` preserves the observed survived/not-survived class proportions between train and test. Preprocessing uses `Pipeline` + `ColumnTransformer`: numeric features are median-imputed and standardized, while `sex` and `embarked` are most-frequent imputed and one-hot encoded. These transformers are fit only inside the training step of each pipeline, then reused in transform-only mode on held-out data.",
        "",
        "## Classification class balance",
        "",
        train_class_balance.rename("count").to_frame().assign(percentage=lambda x: (x["count"] / x["count"].sum() * 100).round(2)).to_markdown(),
        "",
        "Stratification matters because the target classes are not exactly equal in the Titanic data. Keeping the class proportions similar in train and test makes the held-out metrics more representative of the same target mix used for training.",
        "",
        "## Classifier comparison",
        "",
        classifier_metrics[["model", "accuracy", "precision", "recall", "f1", "auc"]].round(4).to_markdown(index=False),
        "",
        "## Combined model comparison table",
        "",
        "The table below keeps classification and regression metrics as separate column groups. `—` means the metric family does not apply to that model type.",
        "",
        combined_comparison.round(4).fillna("—").to_markdown(index=False),
        "",
        "![Confusion matrices](plots/classifier_confusion_matrices.png)",
        "",
        "![ROC curves](plots/classifier_roc_curves.png)",
        "",
        "The confusion matrices show the error counts for each classifier on the same stratified test split. The ROC plot compares ranking quality across thresholds through AUC, while the table reports fixed-threshold classification metrics.",
        "",
        "![Decision tree](plots/decision_tree.png)",
        "",
        "The decision tree visualization labels the transformed feature names and the two output classes. Only the first four levels are displayed to keep the chart readable while still exposing the learned split structure.",
        "",
        "## Imbalance handling comparison",
        "",
        imbalance_df.round(4).to_markdown(index=False),
        "",
        f"Across the three variants, **{imbalance_df.loc[imbalance_df['f1'].idxmax(), 'strategy']}** has the highest F1 score in this run ({imbalance_df['f1'].max():.3f}). The baseline is a reference point, `class_weight='balanced'` changes the estimator's class weighting, and SMOTE creates synthetic minority examples only inside the training pipeline so test information is not used to create synthetic samples.",
        "",
        "## Random Forest hyperparameter tuning",
        "",
        f"Best parameters: **{grid.best_params_}**.",
        f"Best cross-validation score (F1): **{grid.best_score_:.4f}**.",
        f"Refitted best estimator OOB score: **{grid.best_estimator_.named_steps['classifier'].oob_score_:.4f}**.",
        "",
        "The Random Forest estimator is explicitly created with `oob_score=True`, so the refitted best estimator exposes `oob_score_`. The grid search itself is wrapped around the complete preprocessing + estimator pipeline, preventing preprocessing leakage during cross-validation.",
        "",
        "## Regression side-task — predict fare",
        "",
        f"MAE: **{regression_metrics['mae']:.4f}**  \nRMSE: **{regression_metrics['rmse']:.4f}**  \nR²: **{regression_metrics['r2']:.4f}**  \nAdjusted R²: **{regression_metrics['adjusted_r2']:.4f}**",
        "",
        "![Regression residual plot](plots/regression_residuals.png)",
        "",
        f"The residual-vs-predicted diagnostic is flagged as **{'showing evidence of heteroscedasticity' if hetero_flag else 'not showing strong evidence of heteroscedasticity'}** by the simple absolute-residual/predicted-value correlation check implemented in the script. The plot should be inspected alongside this numeric flag because the visual pattern, not the threshold alone, is the diagnostic of interest.",
        "",
        "## Final model decision",
        "",
        recommendation,
        "",
        "### Classification and regression metrics are separate groups",
        "",
        "Classifier metrics (accuracy, precision, recall, F1, AUC) describe classification performance, while MAE, RMSE, R², and Adjusted R² describe regression performance. They are intentionally reported as separate metric groups and are not treated as directly comparable numbers.",
        "",
        "## Saved complete pipeline",
        "",
        "`artifacts/best_classifier_pipeline.joblib` contains the preprocessing transformer(s) and final classifier together. The script reloads this object with `joblib.load` and predicts directly from a raw, unpreprocessed feature row.",
        "",
        f"Reload check prediction for one held-out raw row: **{reload_prediction}**.",
    ]
    README.write_text(base + "\n".join(section) + "\n", encoding="utf-8")


def main() -> None:
    if not CLEAN_CSV.exists():
        raise FileNotFoundError("Run 01_eda.py first so analytics/titanic.csv exists.")

    df = pd.read_csv(CLEAN_CSV)
    df["survived"] = df["survived"].astype(int)

    X = df[CLASSIFICATION_FEATURES].copy()
    y = df["survived"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train class balance:")
    print(y_train.value_counts())

    classifier_specs = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }

    models = {}
    results = []
    for name, estimator in classifier_specs.items():
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(NUMERIC_CLASSIFICATION, CATEGORICAL_CLASSIFICATION),
                ),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        models[name] = pipeline
        result = evaluate_classifier(name, pipeline, X_test, y_test)
        results.append(result)
        print("\n", name)
        print(classification_report(y_test, result["y_pred"], zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_test, result["y_pred"]))

    metrics_df = pd.DataFrame(results)
    metrics_for_report = metrics_df[["model", "accuracy", "precision", "recall", "f1", "auc"]].copy()
    metrics_for_report.to_csv(OUTPUT_DIR / "classifier_metrics.csv", index=False)
    save_classifier_plots(models, X_test, y_test)
    save_decision_tree_plot(models["Decision Tree"])

    imbalance_specs = {
        "baseline": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "class_weight=balanced": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    imbalance_results = []
    for strategy, estimator in imbalance_specs.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(NUMERIC_CLASSIFICATION, CATEGORICAL_CLASSIFICATION)),
                ("classifier", estimator),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        imbalance_results.append(
            {
                "strategy": strategy,
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
        )

    smote_pipe = ImbPipeline(
        steps=[
            ("preprocessor", make_preprocessor(NUMERIC_CLASSIFICATION, CATEGORICAL_CLASSIFICATION)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
            ),
        ]
    )
    smote_pipe.fit(X_train, y_train)
    smote_pred = smote_pipe.predict(X_test)
    imbalance_results.append(
        {
            "strategy": "SMOTE(training only)",
            "precision": precision_score(y_test, smote_pred, zero_division=0),
            "recall": recall_score(y_test, smote_pred, zero_division=0),
            "f1": f1_score(y_test, smote_pred, zero_division=0),
        }
    )
    imbalance_df = pd.DataFrame(imbalance_results)
    imbalance_df.to_csv(OUTPUT_DIR / "imbalance_comparison.csv", index=False)
    print("\nImbalance comparison:")
    print(imbalance_df.round(4))

    rf_tune_pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(NUMERIC_CLASSIFICATION, CATEGORICAL_CLASSIFICATION)),
            (
                "classifier",
                RandomForestClassifier(oob_score=True, random_state=RANDOM_STATE, n_jobs=-1),
            ),
        ]
    )
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__max_features": ["sqrt", "log2"],
    }
    grid = GridSearchCV(
        rf_tune_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    print("\nBest RF parameters:", grid.best_params_)
    print("Best CV F1:", grid.best_score_)
    print("Best estimator OOB score:", grid.best_estimator_.named_steps["classifier"].oob_score_)

    grid_pred = grid.best_estimator_.predict(X_test)
    grid_auc = roc_auc_score(y_test, grid.best_estimator_.predict_proba(X_test)[:, 1])
    print("Tuned RF test metrics:", {
        "accuracy": accuracy_score(y_test, grid_pred),
        "precision": precision_score(y_test, grid_pred, zero_division=0),
        "recall": recall_score(y_test, grid_pred, zero_division=0),
        "f1": f1_score(y_test, grid_pred, zero_division=0),
        "auc": grid_auc,
    })
    pd.DataFrame(grid.cv_results_).to_csv(OUTPUT_DIR / "rf_grid_search_results.csv", index=False)

    X_reg = df[REGRESSION_FEATURES].copy()
    y_reg = df["fare"].copy()
    X_reg_train = X_reg.loc[X_train.index]
    X_reg_test = X_reg.loc[X_test.index]
    y_reg_train = y_reg.loc[X_train.index]
    y_reg_test = y_reg.loc[X_test.index]

    regression_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(REGRESSION_NUMERIC, REGRESSION_CATEGORICAL),
            ),
            ("regressor", LinearRegression()),
        ]
    )
    regression_pipeline.fit(X_reg_train, y_reg_train)
    reg_pred = regression_pipeline.predict(X_reg_test)

    mae = mean_absolute_error(y_reg_test, reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, reg_pred))
    r2 = r2_score(y_reg_test, reg_pred)
    transformed_p = int(regression_pipeline.named_steps["preprocessor"].transform(X_reg_test).shape[1])
    n = len(y_reg_test)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - transformed_p - 1)
    regression_metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "adjusted_r2": adjusted_r2,
    }

    residuals = y_reg_test.to_numpy() - reg_pred
    abs_residual_corr = float(np.corrcoef(np.abs(residuals), reg_pred)[0, 1])
    hetero_flag = abs(abs_residual_corr) > 0.30
    print("\nRegression metrics:", regression_metrics)
    print("Absolute residual vs prediction correlation:", abs_residual_corr)
    print("Heteroscedasticity heuristic:", hetero_flag)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=reg_pred, y=residuals, ax=ax)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted fare")
    ax.set_ylabel("Residual")
    ax.set_title("Fare regression residual plot")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "regression_residuals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    best_idx = metrics_df["f1"].idxmax()
    best_name = metrics_df.loc[best_idx, "model"]
    best_pipeline = models[best_name]
    pipeline_path = ARTIFACTS_DIR / "best_classifier_pipeline.joblib"
    joblib.dump(best_pipeline, pipeline_path)

    reloaded = joblib.load(pipeline_path)
    raw_sample = X_test.iloc[[0]].copy()
    reload_prediction = int(reloaded.predict(raw_sample)[0])
    print("Reloaded pipeline prediction:", reload_prediction)

    summary = {
        "classifier_accuracy": float(metrics_for_report.loc[metrics_for_report["model"] == best_name, "accuracy"].iloc[0]),
        "classifier_precision": float(metrics_for_report.loc[metrics_for_report["model"] == best_name, "precision"].iloc[0]),
        "classifier_recall": float(metrics_for_report.loc[metrics_for_report["model"] == best_name, "recall"].iloc[0]),
        "classifier_f1": float(metrics_for_report.loc[metrics_for_report["model"] == best_name, "f1"].iloc[0]),
        "classifier_auc": float(metrics_for_report.loc[metrics_for_report["model"] == best_name, "auc"].iloc[0]),
        **regression_metrics,
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "final_metric_groups.csv", index=False)

    train_balance = y_train.value_counts().sort_index()
    write_model_report(
        cleaned=df,
        train_class_balance=train_balance,
        classifier_metrics=metrics_for_report,
        imbalance_df=imbalance_df,
        grid=grid,
        regression_metrics=regression_metrics,
        hetero_flag=hetero_flag,
        best_name=best_name,
        best_metric_row=metrics_for_report.loc[best_idx],
        reload_prediction=reload_prediction,
    )

    print("Modeling complete. Saved:", pipeline_path)


if __name__ == "__main__":
    main()
