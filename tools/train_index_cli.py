#!/usr/bin/env python3
"""
Command-line tool to train feature index for RVC models
Usage: python train_index_cli.py --model_name "kamel's voice" --version v2
"""
import os
import sys
import argparse
import platform

# Set environment variables before imports
os.environ["FORCE_CPU_MODE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

now_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(now_dir)

from dotenv import load_dotenv
import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans
from configs.config import Config

load_dotenv()


def train_index(exp_dir1, version19="v2"):
    """Train feature index for a model"""
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)

    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )

    if not os.path.exists(feature_dir):
        return f"Error: Feature directory does not exist: {feature_dir}\nPlease run feature extraction first!"

    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return f"Error: Feature directory is empty: {feature_dir}\nPlease run feature extraction first!"

    print(f"Loading features from: {feature_dir}")
    infos = []
    npys = []
    for name in sorted(listdir_res):
        if name.endswith(".npy"):
            phone = np.load("%s/%s" % (feature_dir, name))
            npys.append(phone)
            print(f"  Loaded {name}: shape {phone.shape}")

    if len(npys) == 0:
        return f"Error: No .npy files found in {feature_dir}"

    print(f"\nConcatenating {len(npys)} feature files...")
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    print(f"Total features shape: {big_npy.shape}")

    if big_npy.shape[0] > 2e5:
        print(
            f"Large dataset ({big_npy.shape[0]} features), applying k-means clustering to 10k centers..."
        )
        try:
            config = Config()
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
            print(f"K-means completed. New shape: {big_npy.shape}")
        except Exception as e:
            print(f"Warning: K-means failed: {e}")
            print("Continuing without k-means...")

    # Save total features
    total_fea_path = "%s/total_fea.npy" % exp_dir
    print(f"\nSaving total features to: {total_fea_path}")
    np.save(total_fea_path, big_npy)

    # Calculate IVF parameters
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    print(f"IVF clusters: {n_ivf}")

    # Create index
    feature_dim = 256 if version19 == "v1" else 768
    index = faiss.index_factory(feature_dim, "IVF%s,Flat" % n_ivf)
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1

    print("\nTraining index...")
    index.train(big_npy)

    trained_index_path = "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
        exp_dir,
        n_ivf,
        index_ivf.nprobe,
        exp_dir1,
        version19,
    )
    print(f"Saving trained index to: {trained_index_path}")
    faiss.write_index(index, trained_index_path)

    print("\nAdding features to index...")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
        if (i // batch_size_add) % 10 == 0:
            print(
                f"  Added {min(i + batch_size_add, big_npy.shape[0])}/{big_npy.shape[0]} features"
            )

    added_index_path = "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
        exp_dir,
        n_ivf,
        index_ivf.nprobe,
        exp_dir1,
        version19,
    )
    print(f"\nSaving final index to: {added_index_path}")
    faiss.write_index(index, added_index_path)

    # Create symlink with model name
    index_root = os.getenv("index_root", "logs")
    outside_index_root = "%s/%s" % (now_dir, index_root)
    os.makedirs(outside_index_root, exist_ok=True)

    symlink_path = "%s/%s_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
        outside_index_root,
        exp_dir1,
        n_ivf,
        index_ivf.nprobe,
        exp_dir1,
        version19,
    )

    try:
        link = os.link if platform.system() == "Windows" else os.symlink
        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        link(added_index_path, symlink_path)
        print(f"Created symlink: {symlink_path}")
    except Exception as e:
        print(f"Warning: Could not create symlink: {e}")

    # Also create a simple named index file for easier access
    simple_index_path = "%s/%s.index" % (exp_dir, exp_dir1)
    try:
        if os.path.exists(simple_index_path):
            os.remove(simple_index_path)
        link = os.link if platform.system() == "Windows" else os.symlink
        link(added_index_path, simple_index_path)
        print(f"Created simple index link: {simple_index_path}")
    except Exception as e:
        print(f"Warning: Could not create simple index link: {e}")

    return f"\n✓ Successfully created index!\n  Main index: {added_index_path}\n  Simple link: {simple_index_path}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train feature index for RVC model")
    parser.add_argument(
        "--model_name",
        required=True,
        help="Model/experiment name (e.g., 'kamel's voice')",
    )
    parser.add_argument(
        "--version", default="v2", choices=["v1", "v2"], help="Model version (v1 or v2)"
    )

    args = parser.parse_args()

    result = train_index(args.model_name, args.version)
    print(result)
