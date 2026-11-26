"""Stage the official Linux wheels from a PyPI mirror; verify upstream lock hashes."""
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import threading
import time
import tomllib
from urllib.parse import unquote

import requests
from packaging.markers import Marker
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


def main():
    root = Path(__file__).resolve().parent
    packages = tomllib.loads((root / "upstream/uv.lock").read_text())["package"]
    targets = ["vllm", "trl", "peft", "datasets", "accelerate", "transformers", "loguru", "python-dotenv"]
    tags = {t: i for i, t in enumerate(sys_tags())}
    pending = [(name, ()) for name in targets]
    seen, by_package = set(), {}
    while pending:
        name, extras = pending.pop()
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        options = [p for p in packages if p["name"] == name and
                   (not p.get("resolution-markers") or any(Marker(m).evaluate() for m in p["resolution-markers"]))]
        if len(options) != 1:
            raise ValueError((name, len(options)))
        pkg = options[0]
        candidates = []
        for w in pkg.get("wheels", []):
            wt = parse_wheel_filename(unquote(w["url"].split("/")[-1]))[-1]
            ranks = [tags[t] for t in wt if t in tags]
            if ranks:
                candidates.append((min(ranks), w))
        if candidates:
            artifact = min(candidates, key=lambda x: x[0])[1]
        else:
            artifact = pkg["sdist"]
        by_package[name] = artifact
        deps = list(pkg.get("dependencies", []))
        for extra in extras:
            deps.extend(pkg.get("optional-dependencies", {}).get(extra, []))
        for dep in deps:
            if not dep.get("marker") or Marker(dep["marker"]).evaluate():
                pending.append((dep["name"], tuple(sorted(dep.get("extra", [])))))
    artifacts = list(by_package.values())
    cache = root / "locked-wheels"
    cache.mkdir(exist_ok=True)
    print(len(artifacts), "artifacts,", round(sum(a["size"] for a in artifacts)/1e9, 2), "GB", flush=True)

    def fetch(a):
        name = unquote(a["url"].rsplit("/", 1)[1])
        path = cache / name
        expected = a["hash"].split(":", 1)[1]
        if path.exists() and path.stat().st_size == a["size"]:
            with path.open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() == expected:
                    return
        mirror = a["url"].replace("https://files.pythonhosted.org/packages/", "http://mirrors.aliyun.com/pypi/packages/")
        tmp = cache / (name + ".part")
        for attempt in range(4):
            try:
                # The pre-existing *.part is a contiguous prefix left by a
                # streaming transfer. Write later ranges to a distinct file.
                base = min(tmp.stat().st_size, a["size"]) if tmp.exists() else 0
                out = cache / (name + ".ranged")
                ledger = cache / (name + ".ranges")
                resume = out.exists() and ledger.exists()
                done = set(map(int, ledger.read_text().split())) if resume else set()
                with out.open("r+b" if resume else "wb") as file, ledger.open("a" if resume else "w") as progress:
                    if base and not resume:
                        with tmp.open("rb") as old:
                            import shutil
                            shutil.copyfileobj(old, file, length=1024*1024)
                    file.flush()
                    lock = threading.Lock()
                    chunk_size = 3 * 1024 * 1024
                    def piece(start):
                        if start in done:
                            return
                        end = min(start + chunk_size, a["size"]) - 1
                        for retry in range(8):
                            try:
                                r = requests.get(mirror, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 60))
                                r.raise_for_status()
                                if r.status_code == 206:
                                    if r.headers.get("Content-Range") != f'bytes {start}-{end}/{a["size"]}':
                                        raise ValueError(f"unexpected Content-Range {r.headers.get('Content-Range')}")
                                elif start != 0 or end+1 != a["size"]:
                                    raise ValueError(f"unexpected status {r.status_code}")
                                if len(r.content) != end-start+1:
                                    raise ValueError("wrong length")
                                if os.pwrite(file.fileno(), r.content, start) != len(r.content):
                                    raise IOError("short write")
                                with lock:
                                    progress.write(f"{start}\n")
                                    progress.flush()
                                return
                            except (requests.RequestException, ValueError) as err:
                                print(name, start, type(err).__name__, str(err)[:200], flush=True)
                                if retry == 7:
                                    raise
                                time.sleep(min(2 ** retry, 20))
                    with ThreadPoolExecutor(5) as ranges:
                        list(ranges.map(piece, range(base, a["size"], chunk_size)))
                with out.open("rb") as file:
                    if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                        tmp.unlink(missing_ok=True)
                        ledger.unlink(missing_ok=True)
                        raise ValueError(f"Hash mismatch for {name}")
                out.replace(path)
                tmp.unlink(missing_ok=True)
                ledger.unlink(missing_ok=True)
                print(name, flush=True)
                return
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(fetch, sorted(artifacts, key=lambda a: -a["size"])))
    dest = sys.argv[1]
    subprocess.run([dest, "-m", "pip", "install", "--no-cache-dir", "--find-links", str(cache),
                    "--no-index", *targets], check=True)


if __name__ == "__main__":
    main()
