import argparse
import os
import codecs as cs
from os.path import join as pjoin

import json
import numpy as np
import torch
from tqdm import tqdm

import models.vqvae as vqvae


def get_dataset_paths(dataname: str):
    if dataname == "t2m":
        root = "../HumanML3D"
        meta_dir = "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
        texts_dir = pjoin(root, "texts")
    elif dataname == "kit":
        root = "./dataset/KIT-ML"
        meta_dir = "checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
        texts_dir = pjoin(root, "texts")
    else:
        raise ValueError(f"Unknown dataname {dataname}")

    motion_dir = pjoin(root, "new_joint_vecs")
    split_file = pjoin(root, "train.txt")
    mean_path = pjoin(meta_dir, "mean.npy")
    std_path = pjoin(meta_dir, "std.npy")

    return motion_dir, split_file, mean_path, std_path, texts_dir


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate motion tokens for all training motions using a pretrained VQ-VAE."
    )
    parser.add_argument("--dataname", type=str, default="t2m", choices=["t2m", "kit"])
    parser.add_argument(
        "--resume-pth",
        type=str,
        required=False,
        help="VQ-VAE checkpoint (.pth) with key 'net' (required unless --tokens-dir is set).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to store <motion>.npy token files. Defaults to dataset/<name>/<exp_name>.",
    )
    parser.add_argument("--exp-name", type=str, default="VQVAE_tokens", help="Subfolder name under dataset root.")
    # Model hyper-parameters (must match the checkpoint)
    parser.add_argument("--nb-code", type=int, default=512)
    parser.add_argument("--code-dim", type=int, default=512)
    parser.add_argument("--output-emb-width", type=int, default=512)
    parser.add_argument("--down-t", type=int, default=2)
    parser.add_argument("--stride-t", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation-growth-rate", type=int, default=3)
    parser.add_argument("--vq-act", type=str, default="relu", choices=["relu", "silu", "gelu"])
    parser.add_argument("--vq-norm", type=str, default=None)
    parser.add_argument(
        "--quantizer",
        type=str,
        default="ema_reset",
        choices=["ema", "orig", "ema_reset", "reset"],
        help="Must match training.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run encoding (cuda/cpu)."
    )
    parser.add_argument(
        "--compute-verbs",
        action="store_true",
        help="Aggregate token representations for verbs using HumanML3D/texts.",
    )
    parser.add_argument(
        "--cluster-verbs",
        action="store_true",
        help="Cluster verb token sequences and store medoid token sequences.",
    )
    parser.add_argument("--cluster-k", type=int, default=3, help="Clusters per verb.")
    parser.add_argument("--cluster-seed", type=int, default=123, help="Random seed for k-means.")
    parser.add_argument("--cluster-max-iter", type=int, default=50, help="Max k-means iterations.")
    parser.add_argument(
        "--texts-dir",
        type=str,
        default=None,
        help="Directory containing text annotations (e.g., HumanML3D/texts).",
    )
    parser.add_argument(
        "--verb-json",
        type=str,
        default=None,
        help="Output JSON path for verb representations. Defaults to <out-dir>/verb_representations.json.",
    )
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default=None,
        help="Directory containing precomputed <motion>.npy token files. Skips VQ-VAE encoding when set.",
    )
    return parser


def load_ids(split_file):
    ids = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            ids.append(line.strip())
    return ids


def parse_verbs_from_tags(tagged_sentence: str):
    verbs = []
    for token in tagged_sentence.strip().split():
        if "/" not in token:
            continue
        word, pos = token.rsplit("/", 1)
        if pos == "VERB":
            verbs.append(word.lower())
    return verbs


def upsample_sequence(seq: np.ndarray, target_len: int):
    if len(seq) == target_len:
        return seq.astype(np.float32)
    if len(seq) == 1:
        return np.repeat(seq.astype(np.float32), target_len)
    x_old = np.linspace(0.0, 1.0, num=len(seq), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(x_new, x_old, seq.astype(np.float32))


def kmeans_features(X: np.ndarray, k: int, max_iter: int, seed: int):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    if k >= n:
        labels = np.arange(n)
        centers = X.copy()
        return centers, labels
    indices = rng.choice(n, size=k, replace=False)
    centers = X[indices].copy()
    for _ in range(max_iter):
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_centers = centers.copy()
        for c in range(k):
            mask = labels == c
            if not np.any(mask):
                new_centers[c] = X[rng.randint(0, n)]
            else:
                new_centers[c] = X[mask].mean(axis=0)
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < 1e-4:
            break
    return centers, labels


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    motion_dir, split_file, mean_path, std_path, default_texts_dir = get_dataset_paths(args.dataname)
    mean = None
    std = None
    if args.tokens_dir is None:
        mean = np.load(mean_path)
        std = np.load(std_path)

    # Output dir
    if args.out_dir is None:
        root = "./dataset/HumanML3D" if args.dataname == "t2m" else "./dataset/KIT-ML"
        args.out_dir = pjoin(root, args.exp_name)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.compute_verbs:
        if args.texts_dir is None:
            args.texts_dir = default_texts_dir
        if args.verb_json is None:
            args.verb_json = pjoin(args.out_dir, "verb_representations.json")
        if not os.path.isdir(args.texts_dir):
            raise FileNotFoundError(f"Texts directory not found: {args.texts_dir}")

    net = None
    device = None
    if args.tokens_dir is None:
        if not args.resume_pth:
            raise ValueError("--resume-pth is required unless --tokens-dir is set.")
        # Build model
        vq_args = argparse.Namespace(
            dataname=args.dataname,
            quantizer=args.quantizer,
            mu=0.99,
            beta=1.0,
            vq_act=args.vq_act,
            vq_norm=args.vq_norm,
            down_t=args.down_t,
            stride_t=args.stride_t,
            width=args.width,
            depth=args.depth,
            dilation_growth_rate=args.dilation_growth_rate,
            output_emb_width=args.output_emb_width,
            code_dim=args.code_dim,
        )

        net = vqvae.HumanVQVAE(
            vq_args,
            nb_code=args.nb_code,
            code_dim=args.code_dim,
            output_emb_width=args.output_emb_width,
            down_t=args.down_t,
            stride_t=args.stride_t,
            width=args.width,
            depth=args.depth,
            dilation_growth_rate=args.dilation_growth_rate,
            activation=args.vq_act,
            norm=args.vq_norm,
        )

        ckpt = torch.load(args.resume_pth, map_location="cpu")
        net.load_state_dict(ckpt["net"], strict=True)
        net.eval()
        device = torch.device(args.device)
        net.to(device)

    unit_length = 2 ** args.down_t  # keep lengths aligned with the VQ encoder
    ids = load_ids(split_file)
    if args.tokens_dir is None:
        print(f"Found {len(ids)} motions. Saving tokens to {args.out_dir}")
    else:
        print(f"Found {len(ids)} motions. Loading tokens from {args.tokens_dir}")

    verb_to_segments = {}
    verb_to_motion_ids = {}
    verb_to_segment_ids = {}

    with torch.no_grad():
        for name in tqdm(ids):
            if args.tokens_dir is None:
                motion_path = pjoin(motion_dir, name + ".npy")
                if not os.path.exists(motion_path):
                    continue
                motion = np.load(motion_path)
                if len(motion) < unit_length:
                    continue

                # trim to a multiple of the unit_length to match the temporal stride
                T = (len(motion) // unit_length) * unit_length
                motion = motion[:T]
                motion_norm = (motion - mean) / std

                motion_tensor = torch.from_numpy(motion_norm).unsqueeze(0).to(device=device, dtype=torch.float32)
                token_seq = net.encode(motion_tensor)  # (1, L)
                token_seq = token_seq.cpu().numpy()
                np.save(pjoin(args.out_dir, name + ".npy"), token_seq)
            else:
                token_path = pjoin(args.tokens_dir, name + ".npy")
                if not os.path.exists(token_path):
                    continue
                token_seq = np.load(token_path)

            if not args.compute_verbs:
                continue

            text_path = pjoin(args.texts_dir, name + ".txt")
            if not os.path.exists(text_path):
                continue

            token_seq_1d = token_seq.reshape(-1)
            with cs.open(text_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split("#")
                    if len(parts) < 2:
                        continue
                    verbs = parse_verbs_from_tags(parts[1])
                    if not verbs:
                        continue
                    if len(verbs) == 1:
                        segments = [token_seq_1d]
                    else:
                        segments = np.array_split(token_seq_1d, len(verbs))

                    for verb, segment in zip(verbs, segments):
                        verb_to_segments.setdefault(verb, []).append(segment)
                        verb_to_motion_ids.setdefault(verb, set()).add(name)
                        verb_to_segment_ids.setdefault(verb, []).append(name)

    if args.compute_verbs:
        verb_entries = []
        for verb, segments in verb_to_segments.items():
            target_len = max(len(s) for s in segments)
            upsampled = [upsample_sequence(s, target_len) for s in segments]
            entry = {
                "motion_ids": sorted(verb_to_motion_ids.get(verb, set())),
                "verb_text": verb,
                "verb_representation": np.stack(upsampled, axis=0).tolist(),
            }

            if args.cluster_verbs:
                X = np.stack([seq.reshape(-1) for seq in upsampled], axis=0)
                k = min(args.cluster_k, X.shape[0])
                centers, labels = kmeans_features(X, k, args.cluster_max_iter, args.cluster_seed)
                clustered = []
                clustered_motion_ids = []
                segment_ids = verb_to_segment_ids.get(verb, [])
                for c_idx in range(centers.shape[0]):
                    mask = labels == c_idx
                    if not np.any(mask):
                        continue
                    cluster_indices = np.where(mask)[0]
                    dists = ((X[cluster_indices] - centers[c_idx]) ** 2).sum(axis=1)
                    medoid_idx = int(cluster_indices[int(dists.argmin())])
                    clustered.append(segments[medoid_idx].astype(int).tolist())
                    clustered_motion_ids.append(
                        segment_ids[medoid_idx] if medoid_idx < len(segment_ids) else None
                    )
                entry["clustered_representation"] = clustered
                entry["clustered_motion_ids"] = clustered_motion_ids

            verb_entries.append(entry)

        verb_entries.sort(key=lambda x: x["verb_text"])
        with cs.open(args.verb_json, "w") as f:
            json.dump(verb_entries, f, indent=2)


if __name__ == "__main__":
    main()
