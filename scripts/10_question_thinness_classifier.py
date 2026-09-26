"""Step 10: Predict whether a question will produce thin or thick thinking.

Aggregates thinking length across C1/C2/C3 per question (mean), classifies
each question as thin (mean <= THRESHOLD) or thick (mean > THRESHOLD), then
trains a classifier on question-level features to predict which bucket.

Question features extracted:
  - question length (chars, words)
  - question word (who/what/which/when/where/how/other)
  - correct answer length (chars, words)
  - correct answer type (person, year, number, place, other) — heuristic
  - wrong answer length (chars, words)
  - wrong/correct answer similarity (char-level)
  - domain (one-hot)

Target: binary — thin (1) vs thick (0), based on mean thinking length across
        all 3 conditions for that question.

Reads:
  data/main_pipeline_results.json

Args:
  --threshold N   Thin/thick boundary in chars. Default: 200.
  --model         logistic | gbm | rf. Default: logistic.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_PATH = os.path.join(DATA_DIR, "main_pipeline_results.json")

QUESTION_WORDS = ("who", "what", "which", "when", "where", "how many", "how", "why")

DOMAINS = ("entertainment", "music", "literature", "science",
           "geography", "history", "sports", "general")

# ── answer type heuristics ──────────────────────────────────────────────────

_YEAR_RE = re.compile(r"^\d{4}$")
_NUM_RE  = re.compile(r"^\d+$")

def answer_type(text: str) -> str:
    t = text.strip()
    if _YEAR_RE.match(t):
        return "year"
    if _NUM_RE.match(t):
        return "number"
    # Capitalized single/double word → likely a proper noun (person or place)
    words = t.split()
    if all(w[0].isupper() for w in words if w):
        return "proper_noun"
    return "other"


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalized edit distance similarity [0, 1]."""
    a, b = a.lower().strip(), b.lower().strip()
    if not a and not b:
        return 1.0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i, ca in enumerate(a):
        ndp = [i + 1] + [0] * lb
        for j, cb in enumerate(b):
            ndp[j + 1] = min(ndp[j] + 1, dp[j + 1] + 1,
                             dp[j] + (0 if ca == cb else 1))
        dp = ndp
    return 1 - dp[lb] / max(la, lb)


# ── feature extraction ───────────────────────────────────────────────────────

def extract_question_features(row: dict) -> dict:
    q  = row["question"]
    ca = row["correct_answer"]
    wa = row["wrong_answer"]
    domain = row.get("domain", "general")

    q_lower = q.lower()
    qword = "other"
    for w in QUESTION_WORDS:
        if q_lower.startswith(w):
            qword = w
            break

    atype = answer_type(ca)

    feats = {
        "q_len_chars":    len(q),
        "q_len_words":    len(q.split()),
        "ca_len_chars":   len(ca),
        "ca_len_words":   len(ca.split()),
        "wa_len_chars":   len(wa),
        "wa_len_words":   len(wa.split()),
        "answer_similarity": levenshtein_ratio(ca, wa),
        # question word one-hot
        **{f"qword_{w.replace(' ', '_')}": int(qword == w) for w in QUESTION_WORDS},
        # answer type one-hot
        **{f"atype_{t}": int(atype == t) for t in ("year", "number", "proper_noun", "other")},
        # domain one-hot
        **{f"domain_{d}": int(domain == d) for d in DOMAINS},
    }
    return feats


FEATURE_NAMES: list[str] = []  # populated after first extraction


def build_dataset(rows: list[dict], threshold: int):
    # Aggregate thinking length per question (mean across conditions)
    by_q: dict[str, dict] = {}
    lengths: dict[str, list] = defaultdict(list)
    for r in rows:
        q = r["question"]
        lengths[q].append(r["thinking_length"])
        if q not in by_q:
            by_q[q] = r  # keep one row per question for feature extraction

    questions = list(by_q.keys())
    mean_lengths = {q: np.mean(lengths[q]) for q in questions}

    X_dicts = [extract_question_features(by_q[q]) for q in questions]
    global FEATURE_NAMES
    FEATURE_NAMES = list(X_dicts[0].keys())
    X = np.array([[d[k] for k in FEATURE_NAMES] for d in X_dicts], dtype=np.float32)
    y = np.array([int(mean_lengths[q] <= threshold) for q in questions])  # 1=thin, 0=thick

    return X, y, questions, mean_lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=200)
    parser.add_argument("--model", default="logistic", choices=["logistic", "gbm", "rf"])
    args = parser.parse_args()

    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    rows = [r for r in rows if r.get("judge_label")]

    X, y, questions, mean_lengths = build_dataset(rows, args.threshold)

    thin_n  = y.sum()
    thick_n = len(y) - thin_n
    print(f"Questions: {len(y)}  |  threshold: {args.threshold} chars")
    print(f"Thin  (mean <= {args.threshold}): {thin_n} ({100*thin_n/len(y):.1f}%)")
    print(f"Thick (mean  > {args.threshold}): {thick_n} ({100*thick_n/len(y):.1f}%)")
    print(f"Features: {len(FEATURE_NAMES)}  |  model: {args.model}")
    print()

    if args.model == "logistic":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
        model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif args.model == "gbm":
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
        model = Pipeline([("clf", clf)])
    else:
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
        model = Pipeline([("clf", clf)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    f1s    = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"5-fold CV F1:       {f1s.mean():.3f} ± {f1s.std():.3f}")
    print()

    y_pred = cross_val_predict(model, X, y, cv=cv)
    print("Classification report:")
    print(classification_report(y, y_pred, target_names=["thick", "thin"], digits=3))
    print("Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y, y_pred)
    for i, label in enumerate(["thick", "thin"]):
        print(f"  {label:<6}: {cm[i]}")
    print()

    # Feature importances
    model.fit(X, y)
    fitted_clf = model.named_steps["clf"]
    print("Top 15 features:")
    if hasattr(fitted_clf, "feature_importances_"):
        imps = fitted_clf.feature_importances_
        ranked = sorted(enumerate(imps), key=lambda x: x[1], reverse=True)
        for idx, imp in ranked[:15]:
            print(f"  {FEATURE_NAMES[idx]:<35} {imp:.4f}")
    elif hasattr(fitted_clf, "coef_"):
        coef = fitted_clf.coef_[0]
        ranked = sorted(enumerate(coef), key=lambda x: abs(x[1]), reverse=True)
        for idx, w in ranked[:15]:
            direction = "-> THIN" if w > 0 else "-> THICK"
            print(f"  {FEATURE_NAMES[idx]:<35} {w:+.3f}  {direction}")

    # Spot-check: most confidently thin and thick predictions
    print()
    probs = model.predict_proba(X)[:, 1]  # P(thin)
    thin_idx  = sorted(range(len(questions)), key=lambda i: probs[i], reverse=True)[:5]
    thick_idx = sorted(range(len(questions)), key=lambda i: probs[i])[:5]

    print("Most confidently predicted THIN questions:")
    for i in thin_idx:
        print(f"  p={probs[i]:.2f} | mean_think={mean_lengths[questions[i]]:.0f}ch | "
              f"actual={'thin' if y[i] else 'thick'} | {questions[i][:70]}")
    print()
    print("Most confidently predicted THICK questions:")
    for i in thick_idx:
        print(f"  p={probs[i]:.2f} | mean_think={mean_lengths[questions[i]]:.0f}ch | "
              f"actual={'thin' if y[i] else 'thick'} | {questions[i][:70]}")


if __name__ == "__main__":
    main()
