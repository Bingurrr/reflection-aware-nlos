#!/usr/bin/env python3
"""Upload data_folder/ to a Hugging Face dataset repository.

    pip install -U "huggingface_hub[cli]"
    huggingface-cli login                 # or export HF_TOKEN=...
    python upload_dataset_hf.py --repo-id <user-or-org>/<dataset-name>

Uploading ~4 GB takes a while; the transfer resumes if it is interrupted, so
re-running the same command is safe. Pass --private to keep the dataset hidden
until you are ready to publish it.
"""
import argparse
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_folder")
CARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DATASET_CARD.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True,
                    help="target dataset repo, e.g. myuser/nlos-radar-release")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--path-in-repo", default="data_folder")
    ap.add_argument("--commit-message", default="Add release data")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit('huggingface_hub is missing. Run: pip install -U "huggingface_hub[cli]"')

    if not os.path.isdir(DATA_DIR):
        sys.exit(f"{DATA_DIR} not found")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)
    print(f"[hf] repo ready: {args.repo_id} (private={args.private})")

    if os.path.isfile(CARD):
        api.upload_file(path_or_fileobj=CARD, path_in_repo="README.md",
                        repo_id=args.repo_id, repo_type="dataset",
                        commit_message="Add dataset card")
        print("[hf] dataset card uploaded")

    api.upload_large_folder(
        folder_path=DATA_DIR,
        repo_id=args.repo_id,
        repo_type="dataset",
        print_report=True,
    )
    print(f"[hf] done -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
