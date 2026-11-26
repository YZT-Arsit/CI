"""Retrieve a slow ModelScope shard by validated byte ranges, without touching an active SDK part."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key")
    parser.add_argument("filename")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    spec = json.loads(Path("extended_files.json").read_bytes())[args.model_key]["base"]
    if spec["gated"]:
        raise ValueError("An authorized local base is required")
    info = spec["files"][args.filename]
    dest = args.directory / args.filename
    if dest.exists():
        with dest.open("rb") as file:
            assert hashlib.file_digest(file, "sha256").hexdigest() == info["sha256"]
        return
    url = f"https://modelscope.cn/models/{spec['repo']}/resolve/{info['modelscope_revision']}/{args.filename}"
    # Resolve once so concurrent range requests go directly to the storage endpoint.
    redirect = requests.head(url, allow_redirects=True, timeout=(30, 60))
    redirect.raise_for_status()
    url = redirect.url
    parts = dest.with_suffix(".segments")
    parts.mkdir(exist_ok=True)
    total = info["bytes"]
    chunk = 8 << 20
    offsets = range(0, total, chunk)

    def piece(start):
        stop = min(total, start + chunk)
        path = parts / str(start)
        if path.exists() and path.stat().st_size == stop-start:
            return
        for attempt in range(10):
            try:
                r = requests.get(url, headers={"Range": f"bytes={start}-{stop-1}"}, timeout=(20, 75), stream=True)
                with r:
                    r.raise_for_status()
                    if r.headers.get("Content-Range") != f"bytes {start}-{stop-1}/{total}":
                        raise ValueError(f"Unexpected Content-Range {r.headers.get('Content-Range')}")
                    content = r.content
                if len(content) != stop-start:
                    raise ValueError("Missing response bytes")
                part = path.with_suffix(".part")
                part.write_bytes(content)
                part.replace(path)
                return
            except (requests.RequestException, ValueError) as err:
                print(start, type(err).__name__, str(err)[:120], flush=True)
                if attempt == 9:
                    raise
                time.sleep(min(2**attempt, 30))
    with ThreadPoolExecutor(8) as pool:
        for i, _ in enumerate(pool.map(piece, offsets), 1):
            if i % 25 == 0:
                print(f"{i}/{len(offsets)}", flush=True)
    assembled = dest.with_suffix(".verified-download")
    digest = hashlib.sha256()
    with assembled.open("wb") as file:
        for start in offsets:
            data = (parts / str(start)).read_bytes()
            file.write(data)
            digest.update(data)
    if digest.hexdigest() != info["sha256"]:
        raise ValueError("Assembled shard does not match the pinned HF digest")
    if not dest.exists():
        os.link(assembled, dest)
    assembled.unlink()
    for start in offsets:
        (parts / str(start)).unlink()
    parts.rmdir()
    print("Verified", dest, digest.hexdigest(), flush=True)


if __name__ == "__main__":
    main()
