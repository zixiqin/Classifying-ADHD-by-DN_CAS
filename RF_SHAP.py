import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, make_scorer,roc_auc_score, f1_score
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV, train_test_split



data = pd.read_csv('data.csv', encoding='utf-8')

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
    "Speech Rate / Sentence Questions"
]
data = data[col]
col_full = col + ["class"]


data["class"] = data["class"].map({"ADHD": 1, "TD": 0}).astype(int)

X = data.drop(columns=['class'])
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)



# define the model and parameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 8, 12],
    'min_samples_split': [2, 4],
    'max_features': ["sqrt", 0.5],
}


# evaluate accuracy, sensitivity (recall for positive class), and specificity (recall for negative class)
scoring = {
    'accuracy': 'accuracy',
    'sensitivity': make_scorer(recall_score, pos_label=1),
    'specificity': make_scorer(recall_score, pos_label=0)
}


# using grid search with 3-fold cross-validation to find the best parameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring=scoring, refit='accuracy', return_train_score=True)
grid_search.fit(X_train, y_train)

# print the best parameters
print(f"Best parameters found: {grid_search.best_params_}")

# use the best model to make predictions
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred, pos_label=1)
specificity = recall_score(y_test, y_pred, pos_label=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)



# --- computing SHAP values ---
explainer = shap.TreeExplainer(best_rf, model_output="raw")
sv = explainer.shap_values(X_test)

def to_sample_feature_for_pos_class(shap_out, n_features):
    """
    Normalize different SHAP output formats to shape (n_samples, n_features),
    taking the positive class (1).

    Compatible with:
    - list: [class0 (n, f), class1 (n, f)]
    - ndarray: (n, f, c) or (n, c, f)
    - Explanation.values: same as above
    """

    # case 1: list
    if isinstance(shap_out, list):
        arr = shap_out[1]          
        if arr.ndim != 2:
            raise ValueError(f"Expected shape (n, f), but got {arr.shape}")

        return arr

    # case 2: ndarray
    arr = np.asarray(shap_out)
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        n, a, b = arr.shape

        if a == n_features and b == 2:
            # (n_samples, n_features, n_classes)
            return arr[:, :, 1]   
        elif a == 2 and b == n_features:
            # (n_samples, n_classes, n_features)
            return arr[:, 1, :]   
        else:
            raise ValueError(f"Unable to determine feature/class dimension: shape={arr.shape}, n_features={n_features}")

    raise ValueError(f"Unsupported SHAP output shape: {arr.shape if hasattr(arr,'shape') else type(arr)}")


n_features = X_test.shape[1]
shap_values_pos = to_sample_feature_for_pos_class(sv, n_features)

print("X_test shape:", X_test.shape)
print("shap_values_pos shape:", shap_values_pos.shape)  # (50, 17)


# --- compute mean |SHAP| for each feature ---
shap_importance = np.abs(shap_values_pos).mean(axis=0)


if hasattr(X_test, "columns") and X_test.shape[1] == shap_values_pos.shape[1]:
    feature_names = list(X_test.columns)
else:
    feature_names = [f"f{i}" for i in range(shap_values_pos.shape[1])]

imp_df = (pd.DataFrame({"feature": feature_names, "mean_abs_shap": shap_importance})
          .sort_values("mean_abs_shap", ascending=False)
          .reset_index(drop=True))

print(imp_df)



sample_idx = 4
sample = X_test.iloc[sample_idx, :]

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_pos[sample_idx, :],
        base_values=explainer.expected_value[1],
        data=sample
    ),
    show=False
)
plt.gcf().set_size_inches(4, 6)
plt.show()



sample_idx = 22
sample = X_test.iloc[sample_idx, :]

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_pos[sample_idx, :],
        base_values=explainer.expected_value[1],
        data=sample
    ),
    show=False
)
plt.gcf().set_size_inches(4, 6)
plt.show()




# ===== input: your feature importance ranking & data =====
FEATURES_ALL = imp_df["feature"].tolist()     
X = data[FEATURES_ALL].copy()
y = data['class'].astype(int).copy()

# ===== your grid =====
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 8, 12],
    'min_samples_split': [2, 4],
    'max_features': ["sqrt", 0.5],
}

def _spec_sens(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return spec, sens

def run_nested_cv_topk(
    X, y, FEATURES_ALL, param_grid,
    outer_repeats: int = 5,
    test_size: float = 0.2,
    inner_cv: int = 3,
    base_seed: int = 42,
    scoring: str = "accuracy",
    n_jobs: int = -1,
    verbose: int = 0
):
    """
    Strict nested CV:
    Outer loop: StratifiedShuffleSplit -> 80/20
    Inner loop: GridSearchCV with StratifiedKFold(inner_cv) on the outer training set
    Top-k: evaluate the top k features one by one (full run for each outer split)

    Returns:
    summary_df: aggregated mean/std grouped by num_features
    details_df: per outer split × k details (including best_params)
    """

    outer = StratifiedShuffleSplit(
        n_splits=outer_repeats,
        test_size=test_size,
        random_state=base_seed
    )

    all_records = []

    for split_id, (tr_idx, te_idx) in enumerate(outer.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values

        # Define inner CV
        inner = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=base_seed + split_id)

        # Top-k: k from 1 to all
        for k in range(1, len(FEATURES_ALL) + 1):
            feats = FEATURES_ALL[:k]

            # Base model
            base = RandomForestClassifier(
                random_state=base_seed + split_id,
                n_jobs=-1,
                class_weight="balanced_subsample"
            )
            # Inner GridSearch (only run on outer training set)
            gs = GridSearchCV(
                estimator=base,
                param_grid=param_grid,
                scoring=scoring,
                cv=inner,
                refit=True,   
                n_jobs=n_jobs,
                verbose=verbose
            )
            gs.fit(X_tr[feats], y_tr)

            # Outer test set evaluation (strictly not involved in hyperparameter tuning)
            proba = gs.predict_proba(X_te[feats])[:, 1]
            y_pred = (proba >= 0.5).astype(int)

            auc = roc_auc_score(y_te, proba)
            acc = accuracy_score(y_te, y_pred)
            f1  = f1_score(y_te, y_pred)
            spec, sens = _spec_sens(y_te, y_pred)

            all_records.append({
                "outer_split": split_id,
                "num_features": k,
                "auc": auc, "acc": acc, "f1": f1,
                "sens": sens, "spec": spec,
                "best_params": gs.best_params_,
            })

        if verbose:
            print(f"[NestedCV] Completed outer split {split_id}/{outer_repeats}")

    details_df = pd.DataFrame(all_records)

    # summary_df: aggregated mean/std grouped by num_features 
    summary_df = details_df.groupby("num_features").agg(
        auc_mean=("auc", "mean"),  auc_std=("auc", "std"),
        acc_mean=("acc", "mean"),  acc_std=("acc", "std"),
        f1_mean=("f1", "mean"),    f1_std=("f1", "std"),
        sens_mean=("sens", "mean"), sens_std=("sens", "std"),
        spec_mean=("spec", "mean"), spec_std=("spec", "std"),
        n_splits=("outer_split", "nunique")
    ).reset_index().sort_values("num_features")

    return summary_df, details_df


summary_df, details_df = run_nested_cv_topk(
    X, y, FEATURES_ALL, param_grid,
    outer_repeats=20,     
    test_size=0.2,
    inner_cv=3,           
    base_seed=43,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

print("Summary (mean ± std, aggregated by num_features) — first few rows:")
print(summary_df.head())

print("\nDetails (per outer split × k) — first few rows:")
print(details_df.head())



print(summary_df)