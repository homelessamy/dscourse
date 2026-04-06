import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, ParameterGrid, train_test_split, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# ── 1. Imports & Config ──

CONFIG = {
    "train_path": "train.csv",
    "test_path": "test.csv",
    "target": "Fake",
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "n_jobs": -1,
    "models": {
        "LogisticRegression": {
            "estimator_params": {"max_iter": 1000, "random_state": 42},
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__penalty": ["l2"],
                "classifier__class_weight": [None, "balanced"]
            }
        },
        "KNeighborsClassifier": {
            "estimator_params": {},
            "param_grid": {
                "classifier__n_neighbors": [5, 10, 15, 25],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2]
            }
        },
        "RandomForestClassifier": {
            "estimator_params": {"random_state": 42},
            "param_grid": {
                "classifier__n_estimators": [200, 300, 500],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", "log2"],
                "classifier__class_weight": ["balanced"]
            }
        },
        "GradientBoostingClassifier": {
            "estimator_params": {"random_state": 42},
            "param_grid": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__learning_rate": [0.03, 0.1, 0.3],
                "classifier__max_depth": [3, 4, 5],
                "classifier__subsample": [0.6, 0.8, 1.0]
            }
        },
        "SVC": {
            "estimator_params": {"probability": True, "random_state": 42},
            "param_grid": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__gamma": ["scale", 0.01, 0.1],
                "classifier__class_weight": [None, "balanced"]
            }
        }
    },
    "figure_dpi": 150,
    "output_dir": "outputs/"
}

EXPECTED_COLUMNS = [
    "ID", "IssueDateTime", "DeclarationOfficeID", "ProcessType", 
    "TransactionNature", "Type", "PaymentType", "BorderTransportMeans", 
    "DeclarerID", "ImporterID", "SellerID", "ExpressID", "ClassificationID", 
    "ExportationCountry", "OriginCountry", "TaxRate", "DutyRegime", 
    "DisplayIndicator", "TotalGrossMassMeasure(KG)", "AdValoremTaxBaseAmount(Won)"
]

CATEGORICAL_COLS = [
    "DeclarationOfficeID", "PaymentType", "BorderTransportMeans", 
    "ProcessType", "TransactionNature", "Type", "DeclarerID", 
    "ImporterID", "SellerID", "DisplayIndicator", "DutyRegime", 
    "ExportationCountry", "OriginCountry"
]

NUMERIC_COLS = [
    "ClassificationID", "TaxRate", "TotalGrossMassMeasure(KG)", 
    "AdValoremTaxBaseAmount(Won)", "mass_per_taxrate", "log_tax_base", 
    "month", "quarter", "is_year_end"
]

# ── 2. Data Ingestion & Validation ──

def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train and test CSVs, validates columns, and logs dataset information.

    """
    train_df = pd.read_csv(config["train_path"])
    test_df = pd.read_csv(config["test_path"])

    assert all(col in train_df.columns for col in EXPECTED_COLUMNS), "Train data misses expected columns."
    assert all(col in test_df.columns for col in EXPECTED_COLUMNS), "Test data misses expected columns."
    assert config["target"] in train_df.columns, "Target column missing from train data."

    return train_df, test_df

# ── 3. Feature Engineering ──

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    features strictly keepi
    

    """
    df_out = df.copy()

    dt_series = pd.to_datetime(df_out["IssueDateTime"])
    df_out["month"] = dt_series.dt.month
    df_out["quarter"] = dt_series.dt.quarter
    df_out["is_year_end"] = (df_out["month"] >= 11).astype(int)

    class_id = df_out["ClassificationID"].astype(float)
    df_out["ClassificationID"] = np.where(class_id > 999999999, class_id // 100, class_id // 10)

    df_out["mass_per_taxrate"] = np.log1p(df_out["TotalGrossMassMeasure(KG)"]) / (df_out["TaxRate"] + 1e-6)
    df_out["log_tax_base"] = np.log1p(df_out["AdValoremTaxBaseAmount(Won)"])

    df_out = df_out.drop(columns=["ExpressID", "IssueDateTime"])
    
    return df_out

# ── 4. Preprocessing & Encoding ──

def build_preprocessor(categorical_cols: list, numeric_cols: list) -> ColumnTransformer:
    """
columnTransformer to handle imputation and scaling/encoding.
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ], remainder='drop', verbose_feature_names_out=False)

    return preprocessor

# ── 5. Pipeline Construction ──

def build_pipelines(preprocessor: ColumnTransformer, config: dict) -> dict[str, Pipeline]:
    """
 instantiated sklearn Pipelines.
    """
    models_cfg = config["models"]
    
    pipelines = {
        "LogisticRegression": Pipeline([
            ('preprocessor', preprocessor),
            ('selector', SelectKBest(f_classif, k=10)),
            ('classifier', LogisticRegression(**models_cfg["LogisticRegression"]["estimator_params"]))
        ]),
        "KNeighborsClassifier": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(**models_cfg["KNeighborsClassifier"]["estimator_params"]))
        ]),
        "RandomForestClassifier": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(**models_cfg["RandomForestClassifier"]["estimator_params"]))
        ]),
        "GradientBoostingClassifier": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(**models_cfg["GradientBoostingClassifier"]["estimator_params"]))
        ]),
        "SVC": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', SVC(**models_cfg["SVC"]["estimator_params"]))
        ])
    }
    
    return pipelines

# ── 6. Evaluation Engine ──

def evaluate_pipelines(pipelines: dict, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, config: dict) -> tuple[pd.DataFrame, dict]:
    """
    scores models using GridSearchCV / RandomizedSearchCV and a hold-out test set.

    """
    results = []
    fitted_pipelines = {}
    
    cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=config["random_state"])
    scoring = ["accuracy", "f1", "roc_auc"]
    
    for name, pipeline in pipelines.items():
        start_time = time.perf_counter()
        
        param_grid = config["models"][name]["param_grid"]
        n_configs = len(ParameterGrid(param_grid))
        n_iter = min(n_configs, 10)
        
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            refit="roc_auc",
            cv=cv,
            n_jobs=config["n_jobs"],
            random_state=config["random_state"]
        )
        
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        fitted_pipelines[name] = best_pipeline
        
        cv_res = search.cv_results_
        best_index = search.best_index_
        best_cv_roc_auc = cv_res["mean_test_roc_auc"][best_index]
        
        y_pred = best_pipeline.predict(X_test)
        
        if hasattr(best_pipeline, "predict_proba"):
            y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(best_pipeline, "decision_function"):
            y_pred_proba = best_pipeline.decision_function(X_test)
        else:
            y_pred_proba = y_pred

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        duration = time.perf_counter() - start_time

        results.append({
            "Model": name,
            "Best_CV_ROC_AUC": best_cv_roc_auc,
            "Test_Accuracy": accuracy,
            "Test_Precision": precision,
            "Test_Recall": recall,
            "Test_F1": test_f1,
            "Test_ROC_AUC": roc_auc,
            "Best_Params": str(search.best_params_),
            "Train_Time_sec": duration
        })

    results_df = pd.DataFrame(results).sort_values(by="Test_ROC_AUC", ascending=False).reset_index(drop=True)
    results_df.to_csv(Path(config["output_dir"]) / "model_comparison.csv", index=False)
    
    return results_df, fitted_pipelines

# ── 7. Visualization Suite ──

def _plot_leaderboard(results_df: pd.DataFrame, out_path: Path, dpi: int):
    plot_data = results_df.set_index("Model")[["Test_Accuracy", "Test_F1", "Test_ROC_AUC"]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data.plot(kind="barh", ax=ax, color=sns.color_palette("muted", 3))
    
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, size=8)
        
    ax.set_title("Model Performance Comparison — Held-Out Test Set", pad=15, weight="bold")
    ax.set_xlabel("Score")
    ax.legend(title="Metric", loc="upper left", bbox_to_anchor=(1, 1))
    
    sns.despine(left=True, bottom=True)
    ax.xaxis.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path / "leaderboard.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _plot_roc_curves(pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series, out_path: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for name, pipeline in pipelines.items():
        if hasattr(pipeline, "predict_proba"):
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_pred_proba = pipeline.decision_function(X_test)
        else:
            y_pred_proba = pipeline.predict(X_test)
            
        auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
        
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.5, label="Random Baseline")
    
    ax.set_title("ROC Curves", pad=15, weight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    sns.despine(top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(out_path / "roc_curves.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _plot_confusion_matrices(pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series, out_path: Path, dpi: int):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, pipeline) in enumerate(pipelines.items()):
        ax = axes[i]
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        
        labels = []
        for row_raw, row_norm in zip(cm, cm_norm):
            labels.append([f"{raw}\n({norm:.1%})" for raw, norm in zip(row_raw, row_norm)])
            
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"{name}\nF1: {f1:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
    axes[5].axis("off")
    
    plt.tight_layout()
    plt.savefig(out_path / "confusion_matrices.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _plot_feature_importance(pipeline: Pipeline, feature_names: list, out_path: Path, dpi: int):
    rf_classifier = pipeline.named_steps.get("classifier")
    
    if not hasattr(rf_classifier, "feature_importances_"):
        return
        
    importances = rf_classifier.feature_importances_
    
    if len(importances) != len(feature_names):
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    imp_df = imp_df.sort_values(by="Importance", ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=imp_df, x="Importance", y="Feature", palette="YlOrRd_r", ax=ax)
    
    mean_imp = importances.mean()
    ax.axvline(mean_imp, color="black", linestyle="--", alpha=0.5, label="Mean Importance")
    
    ax.set_title("Random Forest — Feature Importances (Gini)", pad=15, weight="bold")
    ax.legend(loc="lower right")
    sns.despine(top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(out_path / "feature_importance.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _plot_learning_curve(pipeline: Pipeline, name: str, X_train: pd.DataFrame, y_train: pd.Series, config: dict, out_path: Path, dpi: int):
    train_sizes, train_scores, cv_scores = learning_curve(
        pipeline,
        X_train, y_train,
        cv=StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=config["random_state"]),
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="roc_auc",
        n_jobs=config["n_jobs"]
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    cv_mean = np.mean(cv_scores, axis=1)
    cv_std = np.std(cv_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(train_sizes, train_mean, 'o-', color="blue", label="Training score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
    
    ax.plot(train_sizes, cv_mean, 'o-', color="red", label="Cross-validation score")
    ax.fill_between(train_sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.2, color="red")
    
    ax.set_title(f"Learning Curve — {name}", pad=15, weight="bold")
    ax.set_xlabel("Training Configuration Examples")
    ax.set_ylabel("ROC AUC Score")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(out_path / "learning_curve.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _plot_correlation_heatmap(df: pd.DataFrame, target: str, out_path: Path, dpi: int):
    num_df = df.select_dtypes(include=[np.number])
    if target not in num_df.columns:
        return
        
    corr = num_df.corr()
    
    corr_target = corr[target].abs().sort_values(ascending=False).head(13)
    top_features = corr_target.index
    
    corr_top = num_df[top_features].corr()
    mask = np.triu(np.ones_like(corr_top, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_top, mask=mask, cmap=cmap, vmax=1, center=0, annot=True,
                fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .7}, ax=ax)
                
    ax.set_title("Correlation Heatmap — Top Features", pad=15, weight="bold")
    
    plt.tight_layout()
    plt.savefig(out_path / "correlation_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def generate_visualizations(pipelines: dict, results_df: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, feature_names: list, config: dict) -> None:
    """
    visualizations

    """
    out_dir = Path(config["output_dir"])
    dpi = config["figure_dpi"]
    
    _plot_leaderboard(results_df, out_dir, dpi)
    _plot_roc_curves(pipelines, X_test, y_test, out_dir, dpi)
    _plot_confusion_matrices(pipelines, X_test, y_test, out_dir, dpi)
    
    best_name = results_df.iloc[0]["Model"]
    best_pipeline = pipelines[best_name]
    _plot_learning_curve(best_pipeline, best_name, X_train, y_train, config, out_dir, dpi)
    
    train_full = pd.concat([X_train, y_train], axis=1)
    _plot_correlation_heatmap(train_full, config["target"], out_dir, dpi)
    
    if "RandomForestClassifier" in pipelines:
        rf_pipe = pipelines["RandomForestClassifier"]
        _plot_feature_importance(rf_pipe, feature_names, out_dir, dpi)

# ── 8. Reporting ──

def print_report(results_df: pd.DataFrame) -> None:
    """
    Termi
    """
    table_columns = ["Model", "Test_ROC_AUC", "Test_Accuracy", "Test_F1", "Best_CV_ROC_AUC"]
    print("\n" + tabulate(results_df[table_columns], headers="keys", showindex=False, tablefmt="rounded_outline", floatfmt=".4f"))
    
    best_model = results_df.iloc[0]
    best_name = best_model["Model"]
    best_auc = best_model["Test_ROC_AUC"]
    print(f"\n★ Best Model: {best_name} with ROC-AUC {best_auc:.4f}\n")

# ── 9. Submission ──

def generate_submission(best_pipeline: Pipeline, test_df: pd.DataFrame, test_ids: pd.Series, config: dict) -> None:
    """
    Saves predictions for test submissions.

    """
    test_features = engineer_features(test_df)
    test_preds = best_pipeline.predict(test_features)
    
    assert len(test_preds) == len(test_ids), "Predictions and IDs row-count mismatch"
    
    submission = pd.DataFrame({"ID": test_ids, "Fake": test_preds})
    out_file = Path(config["output_dir"]) / "submission.csv"
    submission.to_csv(out_file, index=False)


# ── 10. Entrypoint ──

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    print("Step 1/9 — Systems & config checks")
    
    print("Step 2/9 — Data ingestion & validation")
    train_df, test_df = load_data(CONFIG)
    
    print("Step 3/9 — Feature engineering & default split")
    train_df = engineer_features(train_df)
    test_ids = test_df["ID"]
    
    X_full = train_df.drop(columns=[CONFIG["target"]])
    y_full = train_df[CONFIG["target"]]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, 
        test_size=CONFIG["test_size"], 
        random_state=CONFIG["random_state"], 
        stratify=y_full
    )
    
    print("Step 4/9 — Preprocessing & encoding")
    preprocessor = build_preprocessor(CATEGORICAL_COLS, NUMERIC_COLS)
    
    print("Step 5/9 — Pipeline construction")
    pipelines = build_pipelines(preprocessor, CONFIG)
    
    print("Step 6/9 — Pipeline tuning & evaluation")
    results_df, fitted_pipelines = evaluate_pipelines(pipelines, X_train, X_test, y_train, y_test, CONFIG)
    
    print("Step 7/9 — Visualization suite")
    if "RandomForestClassifier" in fitted_pipelines:
        rf_pipe = fitted_pipelines["RandomForestClassifier"]
        try:
            fn_out = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
        except:
            fn_out = NUMERIC_COLS + CATEGORICAL_COLS
    else:
        fn_out = NUMERIC_COLS + CATEGORICAL_COLS
        
    generate_visualizations(fitted_pipelines, results_df, X_train, X_test, y_train, y_test, list(fn_out), CONFIG)
    
    print("Step 8/9 — Final reporting")
    print_report(results_df)
    
    print("Step 9/9 — Finalizing submissions file")
    best_model_name = results_df.iloc[0]["Model"]
    generate_submission(fitted_pipelines[best_model_name], test_df, test_ids, CONFIG)
