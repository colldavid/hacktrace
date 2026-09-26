"""Step 15: Deployment-realism analyses — prefix truncation + truthful-doc mix.

Two questions a production RAG warning system must answer, both computable
from existing data (no API calls):

A. PREFIX CURVE — how many characters of thinking trace are needed before
   capitulation is predictable? Trains the classifier on traces truncated to
   the first N chars. In streaming, min(len, N) is what you'd know at char N
   (a trace that ended before N is informative). Reported for two feature
   sets:
     oracle     — all 18 features (includes correct/wrong_in_thinking, which
                  need the answer key; research setting only)
     deployable — ground-truth-free features only (length, doubt/correction/
                  scrutiny markers, etc.); what production actually has.

B. TRUTHFUL-MIX — the thin-trace flag and the classifier are evaluated on a
   realistic population where most documents are truthful. Truthful-doc rows
   come from the correct-document control (95% thin, 99.8% correct); false-
   doc rows from bare C1. Since truthful docs also produce thin traces, this
   measures how precision degrades as the truthful fraction rises.

Reads:
  data/main_pipeline_results_bare.json   (false-doc traces, C1)
  data/correct_control_results.json      (truthful-doc traces)

Args:
  --model  logistic | gbm  (default gbm)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import extract_with_meta, FEATURE_NAMES

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BARE_PATH = os.path.join(DATA_DIR, "main_pipeline_results_bare.json")
CONTROL_PATH = os.path.join(DATA_DIR, "correct_control_results.json")

ORACLE_FEATURES = FEATURE_NAMES + ["condition", "n_documents", "correct_in_thinking", "wrong_in_thinking"]
DEPLOYABLE_FEATURES = [f for f in ORACLE_FEATURES if f not in ("correct_in_thinking", "wrong_in_thinking")]

PREFIX_LENGTHS = (25, 50, 75, 100, 150, 200, 300, 500, 800, None)  # None = full trace


def truncated_features(row: dict, n: int | None) -> dict:
    r = dict(row)
    if n is not None:
        r["thinking_trace"] = row["thinking_trace"][:n]
    return extract_with_meta(r)


def make_model(kind: str):
    if kind == "logistic":
        return Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))])
    return Pipeline([("c", GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gbm", choices=["logistic", "gbm"])
    args = parser.parse_args()

    with open(BARE_PATH, encoding="utf-8") as f:
        bare = [r for r in json.load(f) if r.get("judge_label")]
    with open(CONTROL_PATH, encoding="utf-8") as f:
        control = json.load(f)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ------------------------------------------------------------------
    # A. Prefix curve (all bare rows, capitulated-vs-not)
    # ------------------------------------------------------------------
    y = np.array([1 if r["judge_label"] == "capitulated" else 0 for r in bare])
    print(f"A. PREFIX CURVE — {len(bare)} bare rows, target=capitulated ({100*y.mean():.1f}%)")
    print(f"   model: {args.model}")
    print()
    print(f"   {'prefix':>7} | {'oracle acc':>10} {'oracle AUC':>10} | {'deploy acc':>10} {'deploy AUC':>10}")

    for n in PREFIX_LENGTHS:
        dicts = [truncated_features(r, n) for r in bare]
        Xo = np.array([[d[k] for k in ORACLE_FEATURES] for d in dicts])
        Xd = np.array([[d[k] for k in DEPLOYABLE_FEATURES] for d in dicts])
        acc_o = cross_val_score(make_model(args.model), Xo, y, cv=cv, scoring="accuracy").mean()
        auc_o = cross_val_score(make_model(args.model), Xo, y, cv=cv, scoring="roc_auc").mean()
        acc_d = cross_val_score(make_model(args.model), Xd, y, cv=cv, scoring="accuracy").mean()
        auc_d = cross_val_score(make_model(args.model), Xd, y, cv=cv, scoring="roc_auc").mean()
        label = "full" if n is None else str(n)
        print(f"   {label:>7} | {acc_o:>10.3f} {auc_o:>10.3f} | {acc_d:>10.3f} {auc_d:>10.3f}")

    # ------------------------------------------------------------------
    # B. Truthful-doc mix (C1 only, apples-to-apples: 1 document each)
    # ------------------------------------------------------------------
    false_c1 = [r for r in bare if r["condition"] == 1]
    bad = [r for r in false_c1 if r["judge_label"] == "capitulated"]      # positives
    ok_false = [r for r in false_c1 if r["judge_label"] != "capitulated"] # negatives (false docs, resisted/hedged)
    ok_true = control                                                     # negatives (truthful docs)

    thin = lambda r: r["thinking_length"] <= 200
    p_flag_bad = sum(map(thin, bad)) / len(bad)
    p_flag_okf = sum(map(thin, ok_false)) / len(ok_false)
    p_flag_okt = sum(map(thin, ok_true)) / len(ok_true)

    print()
    print(f"B. TRUTHFUL-MIX — thin-trace flag (<=200ch) as the detector")
    print(f"   flag rate | capitulated (false docs): {100*p_flag_bad:.1f}%")
    print(f"   flag rate | non-cap (false docs):     {100*p_flag_okf:.1f}%")
    print(f"   flag rate | truthful docs:            {100*p_flag_okt:.1f}%")
    print()
    print(f"   Assume per-query P(doc false)=q and P(cap | false doc)={len(bad)/len(false_c1):.2f}:")
    print(f"   {'truthful%':>9} | {'precision':>9} | {'recall':>7} | {'flagged%':>8}")
    p_cap_given_false = len(bad) / len(false_c1)
    for truthful_frac in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
        q = 1 - truthful_frac
        p_bad = q * p_cap_given_false
        p_okf = q * (1 - p_cap_given_false)
        p_okt = truthful_frac
        tp = p_bad * p_flag_bad
        fp = p_okf * p_flag_okf + p_okt * p_flag_okt
        flagged = tp + fp
        precision = tp / flagged if flagged else 0
        recall = p_flag_bad
        print(f"   {100*truthful_frac:>8.0f}% | {100*precision:>8.1f}% | {100*recall:>6.1f}% | {100*flagged:>7.1f}%")

    # Can the deployable classifier separate thin-cap from thin-truthful at all?
    print()
    print("   Deployable classifier within THIN traces only (thin-cap vs thin-truthful):")
    thin_bad = [r for r in bad if thin(r)]
    thin_true = [r for r in ok_true if thin(r)]
    rows_mix = thin_bad + thin_true
    y_mix = np.array([1] * len(thin_bad) + [0] * len(thin_true))
    # Force identical condition/n_documents — control rows lack these keys and
    # would otherwise leak the class through the 0-vs-1 default.
    dicts = [truncated_features({**r, "condition": 1, "n_documents": 1}, None) for r in rows_mix]
    Xd = np.array([[d[k] for k in DEPLOYABLE_FEATURES] for d in dicts])
    auc = cross_val_score(make_model(args.model), Xd, y_mix, cv=cv, scoring="roc_auc").mean()
    acc = cross_val_score(make_model(args.model), Xd, y_mix, cv=cv, scoring="accuracy").mean()
    base = max(y_mix.mean(), 1 - y_mix.mean())
    print(f"   n={len(rows_mix)} (thin-cap {len(thin_bad)}, thin-truthful {len(thin_true)})")
    print(f"   accuracy={acc:.3f} (majority baseline {base:.3f})  AUC={auc:.3f}")


if __name__ == "__main__":
    main()
