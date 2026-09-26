"""Step 9: UMAP visualization of thinking traces.

Two modes:
  --source embeddings  (default) Embed raw trace text with a sentence-transformer.
                       Captures semantic/topical similarity — tends to cluster by
                       question topic rather than label.
  --source features    Use the 18 hand-crafted features from src/features.py.
                       These are directly label-predictive (doubt, correction,
                       thinking length, etc.) so clusters separate by
                       resisted / capitulated / hedged.

Points are colored by judge label (resisted / capitulated / hedged) and shaped
by condition (1 / 2 / 3 documents). Hovering shows the question, thinking
trace excerpt, and model answer excerpt.

Reads:
- data/main_pipeline_results.json

Produces:
- data/umap_embeddings.npy   (embedding cache — skipped in features mode)
- outputs/embedding_viz.html (interactive Plotly figure)

Args:
  --source    embeddings | features  (default: features)
  --model     Sentence-transformer model name (embeddings mode only).
  --sample N  Only use N randomly sampled rows (for fast previews).
  --no-cache  Force re-embedding even if umap_embeddings.npy exists.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.features import extract_with_meta, FEATURE_NAMES
ALL_FEATURE_NAMES = FEATURE_NAMES + ["condition", "n_documents", "correct_in_thinking", "wrong_in_thinking"]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
INPUT_PATH = os.path.join(DATA_DIR, "main_pipeline_results.json")
EMBED_CACHE = os.path.join(DATA_DIR, "umap_embeddings.npy")
HTML_OUT = os.path.join(OUT_DIR, "embedding_viz.html")

LABEL_COLORS = {
    "resisted":    "#2ecc71",   # green
    "capitulated": "#e74c3c",   # red
    "hedged":      "#f39c12",   # amber
}
CONDITION_SYMBOLS = {1: "circle", 2: "square", 3: "diamond"}


def load_rows(path: str, sample: int | None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    rows = [r for r in rows if r.get("judge_label") and r.get("thinking_trace")]
    if sample:
        import random
        random.seed(42)
        rows = random.sample(rows, min(sample, len(rows)))
    return rows


def embed_traces(traces: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Embedding {len(traces)} traces...")
    embeddings = model.encode(traces, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


def reduce_umap(embeddings: np.ndarray) -> np.ndarray:
    import umap
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, metric="cosine")
    return reducer.fit_transform(embeddings)


def build_figure(rows: list[dict], coords: np.ndarray):
    import plotly.graph_objects as go

    traces_plot = []
    for label in ("resisted", "capitulated", "hedged"):
        for cond in (1, 2, 3):
            mask = [
                i for i, r in enumerate(rows)
                if r["judge_label"] == label and r["condition"] == cond
            ]
            if not mask:
                continue
            x = coords[mask, 0]
            y = coords[mask, 1]
            hover = [
                (
                    f"<b>Q:</b> {rows[i]['question'][:80]}<br>"
                    f"<b>correct:</b> {rows[i]['correct_answer']} &nbsp;"
                    f"<b>wrong:</b> {rows[i]['wrong_answer']}<br>"
                    f"<b>label:</b> {rows[i]['judge_label']} &nbsp;"
                    f"<b>C{rows[i]['condition']}</b><br>"
                    f"<b>thinking:</b> {rows[i]['thinking_trace'][:120]}..."
                )
                for i in mask
            ]
            traces_plot.append(go.Scatter(
                x=x, y=y,
                mode="markers",
                name=f"{label} / C{cond}",
                marker=dict(
                    color=LABEL_COLORS[label],
                    symbol=CONDITION_SYMBOLS[cond],
                    size=6,
                    opacity=0.75,
                    line=dict(width=0),
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
            ))

    fig = go.Figure(traces_plot)
    fig.update_layout(
        title=dict(
            text="Thinking Trace Embeddings — Colored by Capitulation Label",
            font=dict(size=18),
        ),
        xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False),
        legend=dict(title="label / condition", itemsizing="constant"),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font=dict(color="#eaeaea"),
        width=1100,
        height=750,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="features", choices=["embeddings", "features"])
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    rows = load_rows(INPUT_PATH, args.sample)
    print(f"Loaded {len(rows)} rows  |  source: {args.source}")

    if args.source == "features":
        print("Extracting features...")
        embeddings = np.array([list(extract_with_meta(r).values()) for r in rows], dtype=np.float32)
        print(f"Feature matrix: {embeddings.shape}")
    else:
        traces = [r["thinking_trace"] for r in rows]
        cache_path = EMBED_CACHE if not args.sample else None
        if cache_path and os.path.exists(cache_path) and not args.no_cache:
            print(f"Loading cached embeddings from {cache_path}")
            embeddings = np.load(cache_path)
            if embeddings.shape[0] != len(rows):
                print("Cache size mismatch — re-embedding")
                embeddings = embed_traces(traces, args.model)
                np.save(cache_path, embeddings)
        else:
            embeddings = embed_traces(traces, args.model)
            if cache_path:
                np.save(cache_path, embeddings)
                print(f"Embeddings cached to {cache_path}")

    coords = reduce_umap(embeddings)

    fig = build_figure(rows, coords)
    fig.write_html(HTML_OUT, include_plotlyjs="cdn")
    print(f"\nSaved to {HTML_OUT}")
    print("Open in a browser to explore.")


if __name__ == "__main__":
    main()
