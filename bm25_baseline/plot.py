#!/usr/bin/env python3
"""
bm25_translation_plot.py

Draw a bar plot comparing BM25 performance using translated vs untranslated queries
on MAP@10 and NDCG@10 metrics.

Output: bm25_translation_comparison.png
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # --- Data setup ---
    data = [
        {"Query Type": "Translated",   "Metric": "MAP@10",  "Score": 0.1659},
        {"Query Type": "Translated",   "Metric": "NDCG@10", "Score": 0.1600},
        {"Query Type": "Untranslated", "Metric": "MAP@10",  "Score": 0.3916},
        {"Query Type": "Untranslated", "Metric": "NDCG@10", "Score": 0.3745},
    ]
    df = pd.DataFrame(data)

    # --- Plot style ---
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    # --- Plot ---
    sns.barplot(
        data=df,
        x="Metric",
        y="Score",
        hue="Query Type",
        ax=ax,
        width=0.6,
        edgecolor="black"
    )

    # --- Customize ---
    ax.set_ylim(0, 0.45)
    ax.set_title("BM25 Performance Comparison\n(Translated vs Untranslated Queries)", fontsize=14, pad=12)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(title="Query Type", loc="upper right", frameon=True)
    sns.despine()

    # --- Annotate bars ---
    for p in ax.patches:
        height = p.get_height()

    # --- Save & show ---
    plt.tight_layout()
    plt.savefig("bm25_translation_comparison.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
