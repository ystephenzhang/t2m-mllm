import argparse
import os
import codecs as cs
from os.path import join as pjoin

import numpy as np
import torch
from tqdm import tqdm

import models.vqvae as vqvae


def get_dataset_paths(dataname: str):
    if dataname == "t2m":
        root = "../HumanML3D"
        meta_dir = "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
    elif dataname == "kit":
        root = "./dataset/KIT-ML"
        meta_dir = "checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
    else:
        raise ValueError(f"Unknown dataname {dataname}")

    motion_dir = pjoin(root, "new_joint_vecs")
    split_file = pjoin(root, "train.txt")
    mean_path = pjoin(meta_dir, "mean.npy")
    std_path = pjoin(meta_dir, "std.npy")

    return motion_dir, split_file, mean_path, std_path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate motion tokens for all training motions using a pretrained VQ-VAE."
    )
    parser.add_argument("--dataname", type=str, default="t2m", choices=["t2m", "kit"])
    parser.add_argument("--resume-pth", type=str, required=True, help="VQ-VAE checkpoint (.pth) with key 'net'.")
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
    return parser


def load_ids(split_file):
    ids = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            ids.append(line.strip())
    return ids


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    motion_dir, split_file, mean_path, std_path = get_dataset_paths(args.dataname)
    mean = np.load(mean_path)
    std = np.load(std_path)

    # Output dir
    if args.out_dir is None:
        root = "./dataset/HumanML3D" if args.dataname == "t2m" else "./dataset/KIT-ML"
        args.out_dir = pjoin(root, args.exp_name)
    os.makedirs(args.out_dir, exist_ok=True)

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
    print(f"Found {len(ids)} motions. Saving tokens to {args.out_dir}")

    with torch.no_grad():
        for name in tqdm(ids):
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


if __name__ == "__main__":
    main()
