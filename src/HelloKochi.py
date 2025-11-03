# ======================================================
# Problem 2: Logistic Regression Experiment
# ======================================================
import os
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------
# 0) Setup
# ------------------------------------------------------
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
AUTHOR_COL = "author"

# ------------------------------------------------------
# 1) Load data
# ------------------------------------------------------
x_train = pd.read_csv("data/x_train.csv")
y_train = pd.read_csv("data/y_train.csv")
x_test  = pd.read_csv("data/x_test.csv")

# Binary target: Key Stage 2-3 → 0, Key Stage 4-5 → 1
y = y_train["Coarse Label"].map({"Key Stage 2-3": 0, "Key Stage 4-5": 1}).values
groups = x_train[AUTHOR_COL].astype(str).values

# Identify feature groups
KEYS = ["author", "title", "passage_id"]
TEXT = "text"
base_num_cols = [c for c in x_train.columns if c not in KEYS + [TEXT]]

# ------------------------------------------------------
# 2) Load provided BERT embeddings
# ------------------------------------------------------
def load_arr_from_npz(npz_path):
    npz_file_obj = np.load(npz_path)
    arr = npz_file_obj.f.arr_0.copy()
    npz_file_obj.close()
    return arr

train_bert_path = "data/x_train_BERT_embeddings.npz"
test_bert_path  = "data/x_test_BERT_embeddings.npz"

bert_train = load_arr_from_npz(train_bert_path)
bert_test  = load_arr_from_npz(test_bert_path)

bert_train_df = pd.DataFrame(bert_train, index=x_train.index).add_prefix("bert_")
bert_test_df  = pd.DataFrame(bert_test,  index=x_test.index ).add_prefix("bert_")

x_train = pd.concat([x_train, bert_train_df], axis=1)
x_test  = pd.concat([x_test,  bert_test_df ], axis=1)

bert_cols = [c for c in x_train.columns if c.startswith("bert_")]
num_cols = base_num_cols

print(f"# Numeric features: {len(num_cols)}, # BERT dims: {len(bert_cols)}")

# ------------------------------------------------------
# 3) Preprocessing pipeline: TF-IDF + SVD + numeric + BERT
# ------------------------------------------------------
tfidf = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    token_pattern=r"\b[a-zA-Z0-9]+\b",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    max_features=50000,
)

text_branch = Pipeline([
    ("tfidf", tfidf),
    ("svd", TruncatedSVD(n_components=300, random_state=RANDOM_STATE)),
    ("scale", StandardScaler(with_mean=True))
])

num_branch = Pipeline([
    ("scale", StandardScaler(with_mean=True))
])

bert_branch = Pipeline([
    ("scale", StandardScaler(with_mean=True))
])

preprocess = ColumnTransformer(
    transformers=[
        ("text", text_branch, TEXT),
        ("num",  num_branch,  num_cols),
        ("bert", bert_branch, bert_cols),
    ],
    remainder="drop",
    sparse_threshold=0.0,
)

# ------------------------------------------------------
# 4) Logistic Regression pipeline
# ------------------------------------------------------
pipe_lr = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=3000,
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])

# ------------------------------------------------------
# 5) Hyperparameter search
# ------------------------------------------------------
param_dist_lr = {
    "prep__text__tfidf__min_df": [1, 2, 3, 5],                  # drop rare terms
    "prep__text__tfidf__max_df": [0.7, 0.8, 0.9, 0.95],         # drop overly common terms
    "prep__text__tfidf__ngram_range": [(1,1), (1,2), (1,3)],     # uni/bi/tri-grams
    "prep__text__tfidf__max_features": [30000, 50000, 80000, 120000],

    "prep__text__svd__n_components": [200, 300, 500, 800],
    
    "clf__C": loguniform(1e-4, 1e2),
    "clf__class_weight": [None, "balanced"]
}

gkf = GroupKFold(n_splits=5)
scorer = make_scorer(roc_auc_score, needs_proba=True)

search_lr = RandomizedSearchCV(
    estimator=pipe_lr,
    param_distributions=param_dist_lr,
    n_iter=20,
    scoring=scorer,
    cv=gkf.split(x_train, y, groups=groups),
    n_jobs=-1,
    verbose=2,
    refit=True,
    random_state=RANDOM_STATE,
    pre_dispatch="2*n_jobs"
)

# ------------------------------------------------------
# 6) Fit and report
# ------------------------------------------------------
print("\n=== Fitting Logistic Regression (Problem 2) ===")
search_lr.fit(x_train, y)
print("\nBest params:", search_lr.best_params_)
print("Best mean CV AUROC:", round(search_lr.best_score_, 4))

# ------------------------------------------------------
# 7) Train final model and save test predictions
# ------------------------------------------------------
print("\nTraining final model on FULL data and predicting test set...")
best_model = search_lr.best_estimator_
best_model.fit(x_train, y)

test_proba = best_model.predict_proba(x_test)[:, 1]
np.savetxt("yproba2_lr_test.txt", test_proba, fmt="%.7f")
print("Saved yproba2_lr_test.txt")
