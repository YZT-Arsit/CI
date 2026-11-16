"""Restore the original Windows tokenizer audit assets without global installs."""
import hashlib
import json
from pathlib import Path
import platform
import sys
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'artifacts/20261112_cpu_pilot/tokenization'


def fetch(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def ensure(path, url, expected):
    data = path.read_bytes() if path.exists() else fetch(url)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'SHA256 mismatch: {path.name}')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def main():
    if (sys.platform != 'win32' or sys.version_info[:2] != (3, 13)
            or platform.machine().lower() not in ('amd64', 'x86_64')):
        raise SystemExit('The archived wheel contract requires Windows amd64 CPython 3.13.')
    manifest = json.loads((ASSETS/'config.json').read_text(encoding='utf-8'))
    for name, sha in manifest['tokenizer_files'].items():
        url = f"https://huggingface.co/{manifest['official_repo']}/resolve/{manifest['official_revision']}/{name}"
        ensure(ASSETS/'official_tokenizer'/name, url, sha)
    for name, row in manifest['wheels'].items():
        package, version = name.split('-')[:2]
        wheel = ASSETS/'wheels'/name
        if not wheel.exists():
            metadata = json.loads(fetch(f'https://pypi.org/pypi/{package}/{version}/json'))
            url = next(x['url'] for x in metadata['urls'] if x['filename'] == name)
        else:
            url = ''
        ensure(wheel, url, row['sha256'])
        target = (ASSETS/'dependencies').resolve()
        with ZipFile(wheel) as archive:
            for entry in archive.infolist():
                path = (target/entry.filename).resolve()
                if not path.is_relative_to(target):
                    raise ValueError('Wheel path escapes isolated directory')
                if entry.is_dir():
                    continue
                data = archive.read(entry)
                if path.exists() and path.read_bytes() != data:
                    # pip rewrites RECORD on installation; the verifier checks
                    # package bytes against the wheel and metadata versions.
                    if entry.filename.endswith('.dist-info/RECORD'):
                        continue
                    raise ValueError(f'Existing dependency differs: {entry.filename}')
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
    sys.path.insert(0, str(ROOT/'code'))
    from kci.training_data import PinnedTokenizerAssets
    result = PinnedTokenizerAssets.from_manifest(ASSETS/'config.json').verify()
    print(json.dumps({'verified': True, 'assets': result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
