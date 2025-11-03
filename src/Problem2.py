import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from scipy.stats import loguniform


RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

AUTHOR_COL = "author"

# ---------------------------------------------------------
# 1) Load core CSVs
# ---------------------------------------------------------
x_train = pd.read_csv("data/x_train.csv")
y_train = pd.read_csv("data/y_train.csv")
x_test  = pd.read_csv("data/x_test.csv")

# Pasted from github
def load_arr_from_npz(npz_path):
    ''' Load array from npz compressed file given path

    Returns
    -------
    arr : numpy ndarray
    '''
    npz_file_obj = np.load(npz_path)
    arr = npz_file_obj.f.arr_0.copy() # Rely on default name from np.savez
    npz_file_obj.close()
    return arr

# Binary target from Coarse Label
y = y_train["Coarse Label"].map({"Key Stage 2-3": 0, "Key Stage 4-5": 1}).values
groups = x_train["author"].values  # GroupKFold by author to mimic unseen authors

KEYS = ["author", "title", "passage_id"]
TEXT = "text"
base_num_cols = [c for c in x_train.columns if c not in KEYS + [TEXT]]

# ---------------------------------------------------------
# 2) Load PROVIDED BERT embeddings and merge by keys
#    Assumes files: x_train_bert.csv, x_test_bert.csv with the same KEYS + many bert_* columns
# ---------------------------------------------------------
train_bert_path = "data/x_train_BERT_embeddings.npz"
test_bert_path  = "data/x_test_BERT_embeddings.npz"

if not (os.path.exists(train_bert_path) and os.path.exists(test_bert_path)):
    raise FileNotFoundError(
        "Expected provided BERT embeddings as x_train_bert.csv and x_test_bert.csv. "
        "Please place them next to x_train.csv/x_test.csv."
    )

bert_train = load_arr_from_npz(train_bert_path)
bert_test = load_arr_from_npz(test_bert_path)


# Merge
# x_train = x_train.merge(bert_train, on=KEYS, how="inner", validate="one_to_one")
# x_test  = x_test.merge(bert_test,  on=KEYS, how="inner", validate="one_to_one")
# Turn arrays into DataFrames aligned with x_* and add a clear prefix
bert_train_df = pd.DataFrame(bert_train, index=x_train.index).add_prefix("bert_")
bert_test_df  = pd.DataFrame(bert_test,  index=x_test.index ).add_prefix("bert_")

# Attach (no merge needed—assumes same row order as x_* CSVs)
x_train = pd.concat([x_train, bert_train_df], axis=1)
x_test  = pd.concat([x_test,  bert_test_df ], axis=1)




# Identify BERT columns (assume they start with 'bert_' or are named numerically)
# bert_col_prefixes = ("bert_", "emb_", "vec_")
# bert_cols = [c for c in x_train.columns if c not in KEYS + [TEXT] + base_num_cols]
# # If your BERT columns are mixed with numeric feature names, filter by a prefix:
# if not any(c.startswith(bert_col_prefixes) for c in bert_cols) and len(bert_cols) > 0:
#     # Keep all newly-added cols as BERT
#     pass
# else:
#     # Prefer explicit prefix filter
#     bert_cols = [c for c in x_train.columns if c.startswith(bert_col_prefixes)]

# if len(bert_cols) == 0:
#     # Fallback: assume all non-key, non-text, non-base numeric columns that appeared after merge are BERT
#     merged_num_cols = [c for c in x_train.columns if c not in KEYS + [TEXT]]
#     bert_cols = [c for c in merged_num_cols if c not in base_num_cols]
bert_cols = [c for c in x_train.columns if c.startswith("bert_")]

# Numeric columns = original numeric (not BERT)
num_cols = base_num_cols

print(f"#Numeric features: {len(num_cols)}, #BERT dims: {len(bert_cols)}")

# ---------------------------------------------------------
# 3) Feature representation
#    - TF-IDF -> SVD -> scale  (dense, compact)
#    - Numeric -> scale
#    - BERT -> scale
# ---------------------------------------------------------
tfidf = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    token_pattern=r"\b[a-zA-Z0-9]+\b",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    max_features=50000,     # you may lower to 30k if you want more speed
    # sublinear_tf=True, # Was worse - maybe remove
)

text_branch = Pipeline([
    ("tfidf", tfidf),
    ("svd", TruncatedSVD(n_components=300, random_state=RANDOM_STATE)),  # try 200–400
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
    sparse_threshold=0.0,  # force dense output (MLP needs dense anyway)
)

# ---------------------------------------------------------
# 4) MLP + caching (to reuse TF-IDF/SVD work across candidates)
# ---------------------------------------------------------
os.makedirs("cache_p2", exist_ok=True)
cache = joblib.Memory(location="cache_p2", verbose=0)

pipe_mlp = Pipeline([
    ("prep", preprocess),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="relu",
        learning_rate_init=1e-4,
        alpha=1e-4,
        batch_size=512,
        max_iter=150,
        early_stopping=True,
        n_iter_no_change=8,
        tol=1e-3,
        random_state=RANDOM_STATE
    ))
], memory=cache)

# Small, informed search around your known best settings
param_dist = {
    # Text branch
    "prep__text__tfidf__ngram_range": [(1,1), (1,2), (1,3)],
    "prep__text__tfidf__min_df": [2, 3, 5],
    "prep__text__tfidf__max_df": [0.85, 0.90, 0.95],
    "prep__text__tfidf__max_features": [30000, 50000, 80000],
    "prep__text__svd__n_components": [200, 300, 500, 800],

    # MLP branch
    "clf__hidden_layer_sizes": [(512,), (384,128), (256,), (256,128)],
    "clf__alpha": loguniform(1e-6, 1e-2),          # much wider regularization sweep
    "clf__learning_rate": ["constant", "adaptive"],
    "clf__learning_rate_init": [1e-4, 3e-4, 1e-3],
    "clf__batch_size": [128, 256, 512],
    "clf__beta_1": [0.8, 0.9],
    "clf__beta_2": [0.99, 0.999],
}

# ---------------------------------------------------------
# 5) CV and search
# ---------------------------------------------------------
gkf = GroupKFold(n_splits=5)
scorer = make_scorer(roc_auc_score, needs_proba=True)

search = RandomizedSearchCV(
    estimator=pipe_mlp,
    param_distributions=param_dist,
    n_iter=20,                       # compact search = fast
    scoring=scorer,
    cv=gkf.split(x_train, y, groups=groups),
    n_jobs=-1,
    verbose=2,
    refit=True,
    random_state=RANDOM_STATE,
    pre_dispatch="2*n_jobs"
)
search.fit(x_train, y)

print("\nBest params:", search.best_params_)
print("Best mean CV AUROC:", round(search.best_score_, 4))
# ---------- NEW: Plot hyperparameter effects ----------
import os
import matplotlib.pyplot as plt

os.makedirs("outputs_p2/plots", exist_ok=True)

cv = pd.DataFrame(search.cv_results_).copy()
# Make a clean AUROC column
cv["auroc"] = cv["mean_test_score"]

# Utility: stringify non-scalar params (e.g., tuples) for plotting/grouping
def _stringify(v):
    try:
        # tuples, ranges, etc.
        return str(v)
    except Exception:
        return v

# 1) One-way plots: for each parameter, show AUROC aggregated by value
param_cols = [c for c in cv.columns if c.startswith("param_")]

for p in param_cols:
    series = cv[p]
    # Coerce to friendly plotting keys
    if series.dtype == "object":
        series = series.map(_stringify)

    tmp = cv.assign(param_val=series)
    # For noisy randomized search, best-of-per-value is often more revealing than mean
    agg = tmp.groupby("param_val", dropna=False)["auroc"].agg(["count", "mean", "max"]).reset_index()

    # Try to sort numerically if possible
    def _try_num(x):
        try:
            return float(x)
        except Exception:
            return x

    # Sort: numeric ascending else lexical
    try:
        agg_sorted = agg.assign(_sort_key=agg["param_val"].map(_try_num)).sort_values("_sort_key").drop(columns=["_sort_key"])
    except Exception:
        agg_sorted = agg

    # Plot: line for mean, markers for best (max)
    plt.figure(figsize=(8, 4.2))
    x = agg_sorted["param_val"].astype(str)
    plt.plot(x, agg_sorted["mean"], marker="o", label="Mean AUROC")
    plt.scatter(x, agg_sorted["max"], label="Best AUROC")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("CV AUROC")
    plt.title(f"Hyperparameter effect: {p}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs_p2/plots/oneway_{p}.png", dpi=160)
    plt.close()

# 2) Two-way views (scatter) for a few high-impact pairs
pairs = [
    ("param_prep__text__svd__n_components", "param_prep__text__tfidf__ngram_range"),
    ("param_clf__alpha", "param_clf__hidden_layer_sizes"),
    ("param_clf__learning_rate_init", "param_clf__batch_size"),
    ("param_prep__text__tfidf__max_features", "param_prep__text__tfidf__min_df"),
]
for xcol, ycol in pairs:
    if xcol in cv.columns and ycol in cv.columns:
        xvals = cv[xcol]
        yvals = cv[ycol]
        # Stringify categorical/tuple params for plotting
        if xvals.dtype == "object":
            xvals = xvals.map(_stringify)
        if yvals.dtype == "object":
            yvals = yvals.map(_stringify)

        plt.figure(figsize=(7.5, 4.8))
        # Use index on x if categorical; matplotlib handles both fine as strings
        plt.scatter(xvals.astype(str), yvals.astype(str), s=36, alpha=0.7)
        for xi, yi, s in zip(xvals.astype(str), yvals.astype(str), cv["auroc"]):
            # annotate lightly with AUROC rounded; skip dense clutter
            if s >= cv["auroc"].quantile(0.85):
                plt.text(xi, yi, f"{s:.3f}", fontsize=7, ha="center", va="center")

        plt.title(f"Two-way grid (size ≈ trials): {xcol} vs {ycol}")
        plt.xlabel(xcol)
        plt.ylabel(ycol)
        plt.tight_layout()
        plt.savefig(f"outputs_p2/plots/twoway_{xcol}_vs_{ycol}.png", dpi=160)
        plt.close()

# 3) AUROC vs numeric scale (log-scale where helpful)
if "param_clf__alpha" in cv.columns:
    plt.figure(figsize=(6.4, 4.2))
    plt.scatter(cv["param_clf__alpha"], cv["auroc"])
    plt.xscale("log")
    plt.xlabel("clf__alpha (log scale)")
    plt.ylabel("CV AUROC")
    plt.title("Regularization sweep")
    plt.tight_layout()
    plt.savefig("outputs_p2/plots/scatter_alpha_auroc.png", dpi=160)
    plt.close()

# 4) Save full results with best-first order for easy inspection
cv.sort_values("auroc", ascending=False).to_csv("outputs_p2/plots/cv_results_sorted.csv", index=False)

os.makedirs("outputs_p2", exist_ok=True)
pd.DataFrame(search.cv_results_).to_csv("outputs_p2/mlp_tfidf_bert_cv_results.csv", index=False)

# ---------------------------------------------------------
# 6) Confusion matrix on held-out predictions (grouped)
# ---------------------------------------------------------
print("\nComputing held-out predictions for confusion matrix...")
y_proba_cv = cross_val_predict(
    search.best_estimator_,
    x_train,
    y,
    cv=gkf.split(x_train, y, groups=groups),
    method="predict_proba",
    n_jobs=-1
)[:, 1]

y_pred_cv = (y_proba_cv >= 0.5).astype(int)


# After computing y_proba_cv and before plotting CM:
cv_auroc = roc_auc_score(y, y_proba_cv)
cv_brier = brier_score_loss(y, y_proba_cv)
cv_logloss = log_loss(y, np.column_stack([1 - y_proba_cv, y_proba_cv]))

print(f"[Grouped OOF] AUROC: {cv_auroc:.4f}  |  Brier: {cv_brier:.4f}  |  LogLoss: {cv_logloss:.4f}")


cm = confusion_matrix(y, y_pred_cv)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["KS2-3", "KS4-5"])
disp.plot(values_format="d")
plt.title("Confusion Matrix — TF-IDF+SVD + Numeric + BERT → MLP (GroupKFold)")
plt.savefig("outputs_p2/mlp_tfidf_bert_confusion_matrix.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 7) Final fit + test predictions
# ---------------------------------------------------------
print("\nTraining final model on FULL train and writing yproba2_test.txt ...")
best_model = search.best_estimator_
best_model.fit(x_train, y)
test_proba = best_model.predict_proba(x_test)[:, 1]
np.savetxt("yproba2_test.txt", test_proba, fmt="%.7f")
print("Saved yproba2_test.txt")
