import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple, Any


try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False



data = pd.read_csv('data.csv')
col = [
    "Number Matching",
    "Planned Codes",
    "Planned Connections",
    "Nonverbal Matrices",
    "Verbal Spatial Relations",
    "Figure Memory",
    "Expressive Attention",
    "Number Detection",
    "Receptive Attention",
    "Word Series",
    "Sentence Repetition",
    "Speech Rate / Sentence Questions",
    "class"
]
data = data[col]
data["class"] = data["class"].map({"ADHD": 1, "TD": 0}).astype(int)


RANDOM_STATE = 42
TARGET = "class"     
df = data              


y = df['class']
X = df.drop(columns=['class'])


# ====== Two types of preprocessing pipeline components ======
# For models that require standardization (LogReg / SVM)
scale_block = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# For models that do not require standardization (tree-based: DT / RF / XGB / LGBM)
tree_block = "passthrough"   # feed raw continuous features directly into the model


# ====== Define models and parameter grids ======

models_and_grids = {}

# Logistic Regression
models_and_grids["LogisticRegression"] = (
    Pipeline([
        ("prep", scale_block),
        ("clf", LogisticRegression(
            max_iter=15000,              
            random_state=RANDOM_STATE
        ))
    ]),
    [
        # ① L2 + lbfgs 
        {
            "clf__solver": ["lbfgs"],
            "clf__penalty": ["l2"],
            "clf__C": [0.05, 0.1, 0.5, 1, 3, 10],
            "clf__class_weight": [None, "balanced"],
        },
        # ② L1 + saga 
        {
            "clf__solver": ["saga"],
            "clf__penalty": ["l1"],
            "clf__C": [0.05, 0.1, 0.5, 1, 3],
            "clf__class_weight": [None, "balanced"],
            "clf__tol": [1e-3],      
        }
    ]
)
# SVM
models_and_grids["SVM_RBF"] = (
    Pipeline([
        ("prep", scale_block),
        ("clf", SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE,
                    cache_size=1000)) 
    ]),
    {
        "clf__C": [1e-2, 1e-1, 1, 3, 10, 30, 100],
        "clf__gamma": [1e-3, 5e-3, 1e-2, 0.1, 1, "scale"],
        "clf__class_weight": [None, "balanced"],
        "clf__shrinking": [True, False],
        "clf__tol": [1e-4, 1e-3]
    }
)

# DecisionTree
models_and_grids["DecisionTree"] = (
    Pipeline([
        ("prep", tree_block),
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]),
    {
        "clf__max_depth": [None, 3, 5, 8, 12],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__criterion": ["gini", "entropy"]
    }
)

# RandomForest
models_and_grids["RandomForest"] = (
    Pipeline([
        ("prep", tree_block),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ]),
    {
        "clf__n_estimators": [100,200],
        "clf__max_depth": [None, 8, 12],
        "clf__min_samples_leaf": [2, 4],
        "clf__max_features": ["sqrt", 0.5]
    }
)

if HAS_XGB:
    models_and_grids["XGBoost"] = (
        Pipeline([
            ("prep", tree_block),
            ("clf", XGBClassifier(
                random_state=RANDOM_STATE,
                n_estimators=400,
                n_jobs=-1,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="auto"
            ))
        ]),
        {
            "clf__max_depth": [3, 4, 5],              
            "clf__learning_rate": [0.03, 0.05, 0.1],  
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.7, 0.9],
            "clf__min_child_weight": [1, 3],
        }
    )




def _encode_binary_labels(y_series: pd.Series) -> Tuple[np.ndarray, Dict[Any, int], int]:
    """
    Map any binary classification labels to {0,1}.
    Rules:
    1) If 'ADHD' is present, set ADHD as the positive class (1), others as 0;
    2) Otherwise, assign the minority class as the positive class (1) 
        (for more robust sensitivity measurement).

    Returns:
    y01: np.ndarray with 0/1 labels
    label_map: dict mapping original labels -> 0/1
    pos_label: original label corresponding to positive class (always 1, for readability only)
    """
    vals, counts = np.unique(y_series, return_counts=True)
    if len(vals) != 2:
        raise ValueError(f"Expected binary classification, but detected {len(vals)} classes: {vals}")


    pos_raw = vals[np.argmin(counts)]
    if any(str(v).upper() == "ADHD" for v in vals):
        for v in vals:
            if str(v).upper() == "ADHD":
                pos_raw = v
                break

    label_map = {vals[0]: 0, vals[1]: 1}
    if label_map[pos_raw] == 0:
        label_map = {k: (1 - v) for k, v in label_map.items()}

    y01 = y_series.map(label_map).values
    return y01, label_map, 1  

def _spec_sens_from_confmat(y_true01: np.ndarray, y_pred01: np.ndarray) -> Tuple[float, float]:
    """specificity=TN/(TN+FP)，sensitivity=TPR=recall(positive)."""
    tn, fp, fn, tp = confusion_matrix(y_true01, y_pred01, labels=[0,1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return spec, sens

def _ci95(values: np.ndarray) -> Tuple[float, float, float]:
    """Return (mean, lower, upper) for a 95% CI (normal approximation)."""
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return mean, mean - half, mean + half

def run_repeated_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models_and_grids: Dict[str, Tuple[Any, Any]],
    outer_repeats: int = 100,
    test_size: float = 0.2,
    inner_cv: int = 5,
    base_seed: int = 0,
    scoring: str = "accuracy",  
    n_jobs: int = -1,
    verbose: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict repeated nested CV evaluation:
    - Outer loop: StratifiedShuffleSplit repeated `outer_repeats` times, each 80/20 (configurable via `test_size`)
    - Inner loop: GridSearchCV with StratifiedKFold(inner_cv) on the current outer training set
    - Metrics recorded on each outer test set: Accuracy / Sensitivity / Specificity

    Returns:
    summary_df: per-model means and 95% CIs for the three metrics
    details_df: per-repeat records (useful for plotting/export)
    """
    y01, label_map, pos_label = _encode_binary_labels(y)

    outer_sss = StratifiedShuffleSplit(
        n_splits=outer_repeats,
        test_size=test_size,
        random_state=base_seed
    )

    all_records = [] 

    for model_name, (pipe, param_grid) in models_and_grids.items():
        acc_list, spec_list, sens_list = [], [], []

        for split_idx, (tr_idx, te_idx) in enumerate(outer_sss.split(X, y01), start=1):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y01[tr_idx], y01[te_idx]

            inner_kfold = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=base_seed + split_idx)

            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=scoring,
                cv=inner_kfold,
                n_jobs=n_jobs,
                refit=True,          
                verbose=verbose
            )
            gs.fit(X_tr, y_tr)

            y_pred = gs.best_estimator_.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            spec, sens = _spec_sens_from_confmat(y_te, y_pred)

            acc_list.append(acc); spec_list.append(spec); sens_list.append(sens)

            all_records.append({
                "model": model_name,
                "repeat_id": split_idx,
                "accuracy": acc,
                "specificity": spec,
                "sensitivity": sens,
                "best_params": gs.best_params_
            })
            if split_idx % 2 == 0:
                print(f"[{model_name}] Completed {split_idx}/{outer_repeats} outer evaluations")



    details_df = pd.DataFrame(all_records)

    # report summary with means and 95% CIs
    rows = []
    for model_name, grp in details_df.groupby("model"):
        acc_mean, acc_lo, acc_hi = _ci95(grp["accuracy"].values)
        spec_mean, spec_lo, spec_hi = _ci95(grp["specificity"].values)
        sens_mean, sens_lo, sens_hi = _ci95(grp["sensitivity"].values)
        rows.append({
            "model": model_name,
            "acc_mean": acc_mean, "acc_95CI_low": acc_lo, "acc_95CI_high": acc_hi,
            "spec_mean": spec_mean, "spec_95CI_low": spec_lo, "spec_95CI_high": spec_hi,
            "sens_mean": sens_mean, "sens_95CI_low": sens_lo, "sens_95CI_high": sens_hi,
            "n_repeats": len(grp)
        })

    summary_df = pd.DataFrame(rows).sort_values("acc_mean", ascending=False).reset_index(drop=True)
    return summary_df, details_df


y = df[TARGET]
X = df.drop(columns=[TARGET])

summary_df, details_df = run_repeated_nested_cv(
    X=X,
    y=y,
    models_and_grids=models_and_grids,
    outer_repeats=20,   # number of strict outer splits (e.g., 20)
    test_size=0.2,
    inner_cv=3,         # inner K-fold CV (can be 3/5/10)
    base_seed=42,       # base random seed for outer splits
    scoring="accuracy", # can also use 'roc_auc' (only for hyperparam selection;
                        # outer loop still reports acc/spec/sens)
    n_jobs=1,
    verbose=0
)


print(summary_df)