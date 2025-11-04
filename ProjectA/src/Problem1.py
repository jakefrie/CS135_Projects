#current version - sort CV groups by author
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from scipy.stats import loguniform

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.base import clone


RANDOM_STATE = 0
custom_stopwords = list(
    text.ENGLISH_STOP_WORDS.union({
    'im', 'ive', 'dont', 'doesnt', 'wasnt', 'didnt',
    'cant', 'couldnt', 'wouldnt', 'youre', 'hes', 'shes',
    'theyre', 'thats', 'also', 'said', 'one'
    })
    )

def make_pipeline():
    return Pipeline([
        # create our Bag of Words
        ("bow", CountVectorizer(
            lowercase=True,          # normalize case before tokenizing
            stop_words=custom_stopwords,    # drop common English stop words
            min_df=1,                # drop terms appearing in < 3 docs (can be overridden by search)
            max_df=0.85,              # drop terms appearing in > 90% of docs (too common)
            ngram_range=(1, 1),      # unigrams only (search may try (1,2) too)
            binary=False             # use integer counts, not just 0/1 presence
        )),
        ("clf", LogisticRegression(
            penalty="l2",            # default regularization (search may flip to "l1")
            # solver="liblinear",      # supports l1/l2; good for smaller, sparse problems
            solver="lbfgs", #AUROC: 0.8307
            # # solver="sag", #AUROC: 0.8247
            # # solver="newton-cg", #AUROC: 0.8309
            # solver="newton-cholesky",
            max_iter=5000,           # ensure convergence
            random_state=RANDOM_STATE
        ))
    ])

def make_RandomizedSearch(cv):
    param_distributions = {
        # ----- Vectorizer knobs -----
        # vocabulary size can matter a lot for AUROC
        "bow__max_features": [None, 20000, 50000, 100000],
        "bow__stop_words": [custom_stopwords, None],
        # prune rare/common terms
        "bow__min_df": [1, 3, 5, 10],
        "bow__max_df": [0.6, 0.65, 0.7, 0.75, 0.775, 0.8, 0.825, 0.85],
        # try adding bigrams
        "bow__ngram_range": [(1,1), (1,2)],
        # binary vs counts
        "bow__binary": [False],

        # ----- Classifier knobs -----
        "clf__C": loguniform(1e-4, 1e2),      # regularization strength
        # "clf__penalty": ["l1", "l2"]
        "clf__penalty": ["l2"],
        "clf__class_weight": [None, "balanced"]
    }

    # K-fold Cross Validation with class-balance preserved in every fold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    #   - draws n_iter random combos from param_distributions
    #   - for each combo: run 5-fold CV, score with AUROC, keep best
    search = RandomizedSearchCV(
        make_pipeline(),
        param_distributions=param_distributions,
        n_iter=100,                 # number of random combos to evaluate
        scoring="roc_auc",         # rank-based metric robust to class imbalance
        cv=cv,                     # 5-fold stratified CV
        n_jobs=-1,                 # use all cores
        random_state=RANDOM_STATE, # make the random draws reproducible
        verbose=1                  # print progress
    )

    return search


def _pick_author_col(df):
    # Try a few common possibilities; adjust if your column is named differently
    for c in ["author", "Author", "user", "user_id", "writer"]:
        if c in df.columns:
            return c
    raise ValueError(
        "No author column found. Please add an 'author' column to x_train/x_test (or rename here)."
    )

def create_model(x_df, y_df, x_te_df):  
    # --- 1) Extract features/labels from input DataFrames ---
    # Raw texts: fill NaN with empty strings so vectorizer doesn't choke
    X_all   = x_df["text"].fillna("")
    # Binary labels - positive class is level 4-5
    Y_all = (y_df["Coarse Label"] == "Key Stage 4-5").astype(int)
    
    author_col = _pick_author_col(x_df)                            # <-- NEW
    groups_all = x_df[author_col].astype(str).fillna("UNK")     # <-- NEW


    # --- 2) Hold-out validation split BEFORE any CV/search ---
    # Keep 20% aside for a final, untouched evaluation of the chosen model
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(outer_cv.split(X_all, Y_all, groups_all))
    X_tr = X_all.iloc[train_idx]
    Y_tr = Y_all.iloc[train_idx]
    g_tr = groups_all.iloc[train_idx]

    X_va = X_all.iloc[val_idx]
    Y_va = Y_all.iloc[val_idx]
    g_va = groups_all.iloc[val_idx]


    # --- 4) Build and run randomized hyperparameter search with 5-fold CV on TRAIN only ---
    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = make_RandomizedSearch(inner_cv)
    search.fit(X_tr, Y_tr, groups=g_tr)

    best_model = search.best_estimator_


    # final_model = make_pipeline()
    # final_model.set_params(**search.best_params_)
    # final_model.fit(text_series, labels_series)

    # --- 5) Predict proba on the full test set and save exactly one float per line ---
    y_va_proba = best_model.predict_proba(X_va)[:, 1]
    val_auc = roc_auc_score(Y_va, y_va_proba)
    print(f"[Grouped] Validation AUROC: {val_auc:.4f}")
    print("Best params:", search.best_params_)
    
    # 6) FINAL REFIT on **all** available training data (train+val), then predict test
    final_model = clone(search.best_estimator_)
    final_model.fit(pd.concat([X_tr, X_va]), pd.concat([Y_tr, Y_va]))

    X_test = x_te_df["text"].fillna("")
    yproba_test = final_model.predict_proba(X_test)[:, 1]
    np.savetxt("yproba1_test.txt", yproba_test, fmt="%.6f")

if __name__ == '__main__':

    data_dir = 'data'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    create_model(x_train_df, y_train_df, x_test_df)