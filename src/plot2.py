# ======================================================
# Problem 2: Logistic Regression — Full Version (2C + 2D)
# ======================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, make_scorer, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_recall_fscore_support, brier_score_loss, log_loss
)
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

# Binary target
y = y_train["Coarse Label"].map({"Key Stage 2-3": 0, "Key Stage 4-5": 1}).values
groups = x_train[AUTHOR_COL].astype(str).values

# Identify columns
KEYS = ["author", "title", "passage_id"]
TEXT = "text"
base_num_cols = [c for c in x_train.columns if c not in KEYS + [TEXT]]

# ------------------------------------------------------
# 2) Load BERT embeddings
# ------------------------------------------------------
def load_arr_from_npz(npz_path):
    npz = np.load(npz_path)
    arr = npz.f.arr_0.copy()
    npz.close()
    return arr

train_bert_path = "data/x_train_BERT_embeddings.npz"
test_bert_path  = "data/x_test_BERT_embeddings.npz"
bert_train = load_arr_from_npz(train_bert_path)
bert_test  = load_arr_from_npz(test_bert_path)

bert_train_df = pd.DataFrame(bert_train, index=x_train.index).add_prefix("bert_")
bert_test_df  = pd.DataFrame(bert_test,  index=x_test.index).add_prefix("bert_")

x_train = pd.concat([x_train, bert_train_df], axis=1)
x_test  = pd.concat([x_test,  bert_test_df ], axis=1)

bert_cols = [c for c in x_train.columns if c.startswith("bert_")]
num_cols = base_num_cols
print(f"# Numeric features: {len(num_cols)}, # BERT dims: {len(bert_cols)}")

# ------------------------------------------------------
# 3) Preprocessing pipeline
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
num_branch  = Pipeline([("scale", StandardScaler(with_mean=True))])
bert_branch = Pipeline([("scale", StandardScaler(with_mean=True))])

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
# 5) Hyperparameter search (2C)
# ------------------------------------------------------
param_dist_lr = {
    "prep__text__tfidf__min_df": [1, 2, 3, 5],
    "prep__text__tfidf__max_df": [0.7, 0.8, 0.9, 0.95],
    "prep__text__tfidf__ngram_range": [(1,1), (1,2), (1,3)],
    "prep__text__tfidf__max_features": [30000, 50000, 80000, 120000],
    "prep__text__svd__n_components": [200, 300, 500, 800],
    "clf__C": loguniform(1e-6, 1e2),
    "clf__class_weight": [None, "balanced"],
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
    return_train_score=True,
)

# ------------------------------------------------------
# 6) Fit & plot (2C)
# ------------------------------------------------------
print("\n=== Fitting Logistic Regression (Problem 2) ===")
search_lr.fit(x_train, y)
print("\nBest params:", search_lr.best_params_)
print("Best mean CV AUROC:", round(search_lr.best_score_, 4))

# --- Plot C vs mean AUROC ---
cv_results = pd.DataFrame(search_lr.cv_results_)
cv_results["C"] = cv_results["param_clf__C"].astype(float)
agg = cv_results.groupby("C").agg(
    mean_val_auc=("mean_test_score", "mean"),
    mean_train_auc=("mean_train_score", "mean")
).reset_index().sort_values("C")

plt.figure(figsize=(8, 5))
plt.plot(agg["C"], agg["mean_train_auc"], marker="o", label="Train AUROC")
plt.plot(agg["C"], agg["mean_val_auc"], marker="o", label="Validation AUROC")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Mean AUROC")
plt.title("Problem 2: Logistic Regression — C vs AUROC")
plt.legend(loc="lower right")
plt.grid(True, linewidth=0.3, alpha=0.5)
plt.tight_layout()
os.makedirs("outputs_p2", exist_ok=True)
plt.savefig("outputs_p2/p2_logreg_C_vs_AUROC.png", dpi=300, bbox_inches="tight")
plt.close()

best_row = agg.loc[agg["mean_val_auc"].idxmax()]
print(
    f"[2C] Best C by mean validation AUROC: {best_row['C']:.4g} "
    f"(val={best_row['mean_val_auc']:.4f}, train={best_row['mean_train_auc']:.4f})"
)

# ------------------------------------------------------
# 7) Error analysis (2D)
# ------------------------------------------------------
print("\n=== Computing OOF predictions for 2D analysis ===")
y_proba_cv = cross_val_predict(
    search_lr.best_estimator_,
    x_train,
    y,
    cv=gkf.split(x_train, y, groups=groups),
    method="predict_proba",
    n_jobs=-1
)[:, 1]
y_pred_cv = (y_proba_cv >= 0.5).astype(int)

cv_auc = roc_auc_score(y, y_proba_cv)
cv_brier = brier_score_loss(y, y_proba_cv)
cv_logloss = log_loss(y, np.column_stack([1 - y_proba_cv, y_proba_cv]))
print(f"[Grouped OOF] AUROC={cv_auc:.4f} | Brier={cv_brier:.4f} | LogLoss={cv_logloss:.4f}")

# --- Confusion matrix ---
cm = confusion_matrix(y, y_pred_cv)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["KS2-3", "KS4-5"])
disp.plot(values_format="d")
plt.title("Problem 2 — Logistic Regression Confusion Matrix (GroupKFold OOF)")
plt.savefig("outputs_p2/p2_logreg_confusion_matrix.png", bbox_inches="tight")
plt.close()

TN, FP, FN, TP = cm.ravel()
tpr = TP / (TP + FN)
tnr = TN / (TN + FP)
acc = accuracy_score(y, y_pred_cv)
prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred_cv, average="binary", zero_division=0)
print(f"[Grouped OOF] Acc={acc:.4f} | TPR={tpr:.4f} | TNR={tnr:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")

# --- Helper: document length ---
def _doc_lengths(text_series):
    return text_series.fillna("").map(lambda t: len(str(t).split())).astype(int)

doc_len = _doc_lengths(x_train["text"])
bins = pd.qcut(doc_len, q=5, duplicates="drop")
by_len = pd.DataFrame({
    "len_bin": bins.astype(str),
    "correct": (y == y_pred_cv).astype(int)
}).groupby("len_bin")["correct"].mean().reset_index()

print("\n[2D] Accuracy by document length quintile:")
for _, row in by_len.iterrows():
    print(f"  {row['len_bin']:>24s} : acc={row['correct']:.3f}")

# --- Author-level accuracy ---
tmp = pd.DataFrame({
    "author": x_train[AUTHOR_COL].astype(str).fillna("UNK"),
    "correct": (y == y_pred_cv).astype(int)
})
by_author = tmp.groupby("author").agg(n=("correct", "size"), acc=("correct", "mean")).reset_index()
by_author = by_author[by_author["n"] >= 5].sort_values("acc", ascending=False)

print("\n[2D] Top/Bottom authors (min 5 docs):")
if not by_author.empty:
    head, tail = by_author.head(5), by_author.tail(5)
    print("  Top:")
    for _, r in head.iterrows():
        print(f"    {r['author'][:32]:<32} n={int(r['n']):<3} acc={r['acc']:.3f}")
    print("  Bottom:")
    for _, r in tail.iterrows():
        print(f"    {r['author'][:32]:<32} n={int(r['n']):<3} acc={r['acc']:.3f}")

# ------------------------------------------------------
# 8) Final training and test prediction
# ------------------------------------------------------
print("\nTraining final model on FULL data and predicting test set...")
best_model = search_lr.best_estimator_
best_model.fit(x_train, y)
test_proba = best_model.predict_proba(x_test)[:, 1]
np.savetxt("yproba2_lr_test.txt", test_proba, fmt="%.7f")
print("Saved yproba2_lr_test.txt")
