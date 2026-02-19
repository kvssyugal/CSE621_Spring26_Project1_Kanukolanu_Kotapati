#!/usr/bin/env python3
"""
CSE621 Project 1 – Experiment Runner
Runs: 3 preprocessing variants x 3 feature selection settings x 3 models + 1 stacking ensemble.
Saves results under project1_outputs/.

EDIT THESE 3 LINES ONLY:
  DATA_PATH, TEXT_COL, LABEL_COL
"""
import os, re, json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# -------- EDIT THESE --------
DATA_PATH = "data/imdb_reviews.csv"     # <-- change to your dataset filename
TEXT_COL  = "review"       # <-- change to your text column name
LABEL_COL = "sentiment"    # <-- change to your label column name
# ----------------------------

LABEL_MAP = {
    "positive": 1, "pos": 1, 1: 1, "1": 1,
    "negative": 0, "neg": 0, 0: 0, "0": 0,
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
OUT_DIR = "project1_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"CSV must contain '{TEXT_COL}' and '{LABEL_COL}'. Found: {list(df.columns)}")
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[TEXT_COL] = df[TEXT_COL].astype(str).map(basic_clean)

    def map_label(x):
        if x in LABEL_MAP: return LABEL_MAP[x]
        xs = str(x).strip().lower()
        if xs in LABEL_MAP: return LABEL_MAP[xs]
        raise ValueError(f"Unknown label: {x}")

    df["y"] = df[LABEL_COL].map(map_label)
    return df[[TEXT_COL, "y"]]

def score(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return acc, p, r, f1

def main():
    df = load_data(DATA_PATH)
    X = df[TEXT_COL].values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("=== DATASET SUMMARY ===")
    print("File:", os.path.basename(DATA_PATH))
    print("Total:", len(df))
    print("Train:", len(X_train), "Test:", len(X_test))

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # 3 preprocessing variants (simple + realistic)
    preprocess = {
        "P1_basic_unigram": TfidfVectorizer(lowercase=False, stop_words=None, ngram_range=(1,1), min_df=2),
        "P2_stopwords_uni+bi": TfidfVectorizer(lowercase=False, stop_words="english", ngram_range=(1,2), min_df=2),
        "P3_stopwords_uni+bi_strict": TfidfVectorizer(lowercase=False, stop_words="english", ngram_range=(1,2), min_df=5),
    }

    # 2 feature selection + baseline
    feature_select = {
        "FS_none": "passthrough",
        "FS_chi2_k10k": SelectKBest(chi2, k=10000),
        "FS_mutual_info_k10k": SelectKBest(mutual_info_classif, k=10000),
    }

    # 3 models + parameter exploration
    model_grids = {
        "MultinomialNB": (MultinomialNB(), {"clf__alpha": [0.1, 0.5, 1.0]}),
        "LinearSVC": (LinearSVC(), {"clf__C": [0.5, 1.0, 2.0]}),
        "SGDClassifier": (
            SGDClassifier(random_state=RANDOM_STATE, max_iter=2000, tol=1e-3),
            {"clf__loss": ["hinge", "log_loss"], "clf__alpha": [1e-4, 1e-3]}
        ),
    }

    rows = []
    for p_name, vec in preprocess.items():
        for fs_name, fs in feature_select.items():
            for m_name, (m, grid) in model_grids.items():
                pipe = Pipeline([("tfidf", vec), ("fs", fs), ("clf", m)])
                gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
                print(f"Training: {p_name} + {fs_name} + {m_name}")
                gs.fit(X_train, y_train)

                best = gs.best_estimator_
                y_pred = best.predict(X_test)
                acc, p, r, f1 = score(y_test, y_pred)

                rows.append({
                    "preprocess": p_name,
                    "feature_select": fs_name,
                    "model": m_name,
                    "best_params": gs.best_params_,
                    "cv_best_f1_weighted": float(gs.best_score_),
                    "test_accuracy": float(acc),
                    "test_precision_weighted": float(p),
                    "test_recall_weighted": float(r),
                    "test_f1_weighted": float(f1),
                })

    df_res = pd.DataFrame(rows).sort_values("test_f1_weighted", ascending=False)
    df_res.to_csv(os.path.join(OUT_DIR, "all_results.csv"), index=False)

    best_single = df_res.iloc[0].to_dict()
    with open(os.path.join(OUT_DIR, "best_single.json"), "w") as f:
        json.dump(best_single, f, indent=2)

    # Ensemble: stacking (required)
    best_p = best_single["preprocess"]
    best_fs = best_single["feature_select"]
    vec_best = preprocess[best_p]
    fs_best = feature_select[best_fs]

    stacking = StackingClassifier(
        estimators=[
            ("nb", MultinomialNB(alpha=0.5)),
            ("svm", LinearSVC(C=1.0)),
            ("sgd", SGDClassifier(random_state=RANDOM_STATE, max_iter=2000, tol=1e-3, loss="hinge", alpha=1e-4)),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        n_jobs=-1
    )

    stack_pipe = Pipeline([("tfidf", vec_best), ("fs", fs_best), ("clf", stacking)])
    stack_pipe.fit(X_train, y_train)
    y_stack = stack_pipe.predict(X_test)
    acc_s, p_s, r_s, f1_s = score(y_test, y_stack)

    ensemble = {
        "preprocess": best_p,
        "feature_select": best_fs,
        "model": "StackingClassifier",
        "test_accuracy": float(acc_s),
        "test_precision_weighted": float(p_s),
        "test_recall_weighted": float(r_s),
        "test_f1_weighted": float(f1_s),
    }
    with open(os.path.join(OUT_DIR, "ensemble.json"), "w") as f:
        json.dump(ensemble, f, indent=2)

    with open(os.path.join(OUT_DIR, "classification_report_ensemble.txt"), "w") as f:
        f.write(classification_report(y_test, y_stack, digits=4))

    np.savetxt(os.path.join(OUT_DIR, "confusion_ensemble.csv"), confusion_matrix(y_test, y_stack), delimiter=",", fmt="%d")

    print("\nSaved results to:", OUT_DIR)
    print("Best single:", best_single["model"], "F1:", best_single["test_f1_weighted"])
    print("Ensemble F1:", f1_s)

if __name__ == "__main__":
    main()
