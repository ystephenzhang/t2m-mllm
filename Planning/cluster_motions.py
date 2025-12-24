import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
T2M_DIR = ROOT_DIR / "T2M-GPT"
sys.path.insert(0, str(T2M_DIR))

import models.vqvae as vqvae
from utils.motion_process import recover_from_ric


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Cluster verb motion token sequences and export medoid motions for rendering."
    )
    parser.add_argument(
        "--verb-json",
        type=str,
        default="verb_representations.json",
        help="Path to verb_representations.json containing token sequences.",
    )
    parser.add_argument(
        "--resume-pth",
        type=str,
        required=True,
        help="VQ-VAE checkpoint (.pth) with key 'net'.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/verb_clusters",
        help="Output directory for medoid motion .npy files.",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of clusters per verb.")
    parser.add_argument("--length", type=int, default=60, help="Resampled length for clustering.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for k-means.")
    parser.add_argument("--max-iter", type=int, default=50, help="Max k-means iterations.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu).")

    parser.add_argument("--dataname", type=str, default="t2m", choices=["t2m", "kit"])
    parser.add_argument(
        "--mean",
        type=str,
        default="HumanML3D/Mean.npy",
        help="Path to dataset mean.npy for denormalization.",
    )
    parser.add_argument(
        "--std",
        type=str,
        default="HumanML3D/Std.npy",
        help="Path to dataset std.npy for denormalization.",
    )

    # VQ-VAE hyper-parameters (must match checkpoint)
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
    return parser


def sanitize_name(text):
    cleaned = []
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch.lower())
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_")


def resample_motion(motion, target_len):
    if motion.shape[0] == target_len:
        return motion.astype(np.float32)
    if motion.shape[0] == 1:
        return np.repeat(motion.astype(np.float32), target_len, axis=0)
    old_idx = np.linspace(0.0, 1.0, num=motion.shape[0], dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    flat = motion.reshape(motion.shape[0], -1)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for d in range(flat.shape[1]):
        out[:, d] = np.interp(new_idx, old_idx, flat[:, d])
    return out.reshape(target_len, motion.shape[1], motion.shape[2])


def motion_to_feature(motion, target_len):
    motion = resample_motion(motion, target_len)
    root = motion[:, :1, :]
    motion = motion - root
    return motion.reshape(-1).astype(np.float32)


def kmeans(X, k, max_iter=50, seed=123):
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


def load_verb_sequences(verb_json):
    data = json.loads(Path(verb_json).read_text())
    entries = []
    for entry in data:
        verb = entry.get("verb_text")
        rep = entry.get("verb_representation", [])
        if not verb or not rep:
            continue
        if isinstance(rep[0], list):
            sequences = rep
        else:
            sequences = [rep]
        cleaned = []
        for seq in sequences:
            if not seq:
                continue
            cleaned.append([int(x) for x in seq])
        if cleaned:
            entries.append((verb, cleaned))
    return entries


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean = torch.from_numpy(np.load(args.mean)).float()
    std = torch.from_numpy(np.load(args.std)).float()

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
    mean = mean.to(device)
    std = std.to(device)

    num_joints = 21 if args.dataname == "kit" else 22

    verb_entries = load_verb_sequences(args.verb_json)
    manifest = []

    with torch.no_grad():
        for verb, sequences in verb_entries:
            verb_name = sanitize_name(verb)
            verb_dir = out_dir / verb_name
            verb_dir.mkdir(parents=True, exist_ok=True)

            motions = []
            features = []
            for seq in sequences:
                token_ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
                pred = net.forward_decoder(token_ids)
                pred = pred * std + mean
                motion_xyz = recover_from_ric(pred, num_joints)[0].detach().cpu().numpy()
                motions.append(motion_xyz)
                features.append(motion_to_feature(motion_xyz, args.length))

            if not motions:
                continue

            X = np.stack(features, axis=0)
            centers, labels = kmeans(X, k=min(args.k, len(motions)), max_iter=args.max_iter, seed=args.seed)

            verb_manifest = {
                "verb": verb,
                "num_sequences": len(motions),
                "clusters": [],
            }

            for c_idx in range(centers.shape[0]):
                mask = labels == c_idx
                if not np.any(mask):
                    continue
                cluster_indices = np.where(mask)[0]
                dists = ((X[cluster_indices] - centers[c_idx]) ** 2).sum(axis=1)
                medoid_local = cluster_indices[int(dists.argmin())]

                out_name = f"cluster_{c_idx:03d}"
                out_path = verb_dir / f"{out_name}.npy"
                np.save(out_path, motions[medoid_local][None, ...])

                verb_manifest["clusters"].append(
                    {
                        "cluster_index": int(c_idx),
                        "medoid_index": int(medoid_local),
                        "output_npy": str(out_path),
                    }
                )

            with open(verb_dir / "medoids.json", "w") as f:
                json.dump(verb_manifest, f, indent=2)

            manifest.append(verb_manifest)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
