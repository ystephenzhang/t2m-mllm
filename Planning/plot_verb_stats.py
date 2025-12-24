import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Plot stats from verb_representations.json.")
    parser.add_argument(
        "--json-path",
        type=str,
        default="verb_representations.json",
        help="Path to verb_representations.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="visualization",
        help="Output directory for plots.",
    )
    return parser


def load_entries(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def plot_top_verbs(entries, out_path, top_k=10):
    items = []
    for entry in entries:
        verb = entry.get("verb_text", "")
        motion_ids = entry.get("motion_ids", [])
        items.append((verb, len(motion_ids)))
    items.sort(key=lambda x: x[1], reverse=True)
    top = items[:top_k]

    verbs = [v for v, _ in top]
    counts = [c for _, c in top]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(verbs, counts, color="#2F6690")
    plt.title(f"Top-{top_k} Most Frequent Verbs")
    plt.xlabel("Verb")
    plt.ylabel("Motion Count")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2.0, count, str(count), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_length_distribution(entries, out_path):
    lengths = [len(entry.get("verb_representation", [])) for entry in entries]
    length_counts = Counter(lengths)
    lengths_sorted = sorted(length_counts.items(), key=lambda x: x[0])

    x = [k for k, _ in lengths_sorted]
    y = [v for _, v in lengths_sorted]

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color="#81B29A")
    plt.title("Verb Representation Length Distribution")
    plt.xlabel("Token Sequence Length")
    plt.ylabel("Verb Count")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    entries = load_entries(args.json_path)

    top_path = os.path.join(args.out_dir, "top_10_verbs.png")
    dist_path = os.path.join(args.out_dir, "verb_rep_length_dist.png")

    plot_top_verbs(entries, top_path, top_k=10)
    plot_length_distribution(entries, dist_path)

    print(f"Saved: {top_path}")
    print(f"Saved: {dist_path}")


if __name__ == "__main__":
    main()
