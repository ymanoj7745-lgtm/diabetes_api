"""
Model training pipeline for NFHS-5 diabetes prediction.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.config import FEATURES, RANDOM_STATE, TEST_SIZE, MODELS


def prepare_data(df, target_col="diabetes"):
    """
    Prepare train/test splits with imputation, scaling, and SMOTE.

    Returns
    -------
    X_train_res, y_train_res : SMOTE-resampled training data
    X_test_sc, y_test : scaled test data
    imputer, scaler : fitted transformers (save for inference)
    """
    X = df[FEATURES].copy()
    y = df[target_col].copy()

    # Drop rows with missing target
    mask = y.notna()
    X, y = X[mask], y[mask].astype(int)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Impute missing features
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test), columns=X.columns, index=X_test.index
    )

    # Scale
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    # SMOTE for class imbalance
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)

    return X_train_res, y_train_res, X_test_sc, y_test, imputer, scaler


def train_all(X_train, y_train):
    """
    Train Logistic Regression, Random Forest, and XGBoost.
    Saves each model to models/ directory.

    Returns
    -------
    models : dict of fitted models
    results : dict of mean CV AUC scores
    """
    models = {
        "logistic": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12,
            class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        aucs = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        results[name] = aucs.mean()
        print(f"{name:20s}: AUC = {aucs.mean():.4f} ± {aucs.std():.4f}")

        # Fit on full training set
        model.fit(X_train, y_train)
        model_path = MODELS / f"{name}.pkl"
        joblib.dump(model, model_path)
        print(f"  → Saved to {model_path}")

    return models, results


def evaluate_best(models, results, X_test, y_test):
    """Evaluate the best model on the held-out test set."""
    best_name = max(results, key=results.get)
    best_model = models[best_name]

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"Best Model: {best_name}")
    print(f"Test AUC:   {auc:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    return best_name, auc, report, cm
