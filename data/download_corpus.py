"""Optional: fetch a real technical-documentation corpus.

The repo ships with a small, self-contained authored corpus (`data/corpus/*.md`)
so everything runs offline. If you want to run against real docs, point this at a
docs site or a GitHub repo's markdown and drop the files under
`data/corpus/downloaded/`, then rebuild the gold set for that corpus.

Usage:
    python data/download_corpus.py --repo tiangolo/fastapi --path docs/en/docs --out data/corpus/downloaded

This uses the GitHub API (no auth needed for public repos, subject to rate
limits) and only downloads .md files. Swapping corpora means the bundled gold set
no longer applies — build a new gold_qa.jsonl for the new domain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

API = "https://api.github.com/repos/{repo}/contents/{path}"
RAW = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def list_markdown(repo: str, path: str, branch: str) -> list[str]:
    files: list[str] = []
    stack = [path]
    with httpx.Client(timeout=30) as client:
        while stack:
            cur = stack.pop()
            r = client.get(API.format(repo=repo, path=cur), params={"ref": branch})
            r.raise_for_status()
            for item in r.json():
                if item["type"] == "dir":
                    stack.append(item["path"])
                elif item["name"].endswith(".md"):
                    files.append(item["path"])
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/name, e.g. tiangolo/fastapi")
    ap.add_argument("--path", default="docs", help="path within the repo to crawl")
    ap.add_argument("--branch", default="master")
    ap.add_argument("--out", default="data/corpus/downloaded")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    md_files = list_markdown(args.repo, args.path, args.branch)[: args.limit]
    print(f"Found {len(md_files)} markdown files; downloading up to {args.limit}...")

    with httpx.Client(timeout=30) as client:
        for i, fp in enumerate(md_files):
            url = RAW.format(repo=args.repo, branch=args.branch, path=fp)
            r = client.get(url)
            if r.status_code != 200:
                continue
            name = fp.replace("/", "__")
            (out / name).write_text(r.text, encoding="utf-8")
            print(f"  [{i+1}/{len(md_files)}] {name}")

    print(
        f"\nDone. Set `corpus_dir: {args.out}` in config.yaml (or ARAG_CORPUS_DIR), "
        "then rebuild the gold set for this corpus before running eval."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
