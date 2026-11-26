"""Download pinned public base files on an internet-connected computer for offline evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from extended_table1 import MODEL_FILES, fetch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODEL_FILES)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--files", nargs="+")
    args = parser.parse_args()
    base = MODEL_FILES[args.model]["base"]
    mirror = base.get("public_modelscope_repo")
    if base["gated"] and not mirror:
        parser.error("Use your authorized installation for a gated base")

    def download(name):
        url = f"https://huggingface.co/{base['repo']}/resolve/{base['revision']}/{name}?download=true"
        if mirror:
            revision = base["files"][name]["modelscope_revision"]
            url = f"https://modelscope.cn/models/{mirror}/resolve/{revision}/{name}"
        fetch(url, args.directory / name, base["files"][name])
        print("Verified", name, flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, args.files or base["files"]))


if __name__ == "__main__":
    main()
