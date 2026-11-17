"""Fetch pinned author-published positive/negative norm data, verifying Git blobs."""
import csv
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parent
REV = '62e5d4529d291ff0eb1f8f9565f035e8a8827589'
FILES = {'README.md': '306453ca688b6a54f8d58df5f2b36c21d5a57cf2',
         'LICENSE': 'f987f3daa1f8637793144323bcd10e330b8db97c',
         'camber/privacylensplus.csv': '46675fc42c8f66bd14a610dda9404454379c1c66'}


def main():
    assert not (R / 'SOURCE_MANIFEST.json').exists()
    records = []
    for name, expected in FILES.items():
        url = f'https://raw.githubusercontent.com/google-parfait/contextual-privacy-and-security/{REV}/{name}'
        raw = urllib.request.urlopen(url, timeout=30).read(10_000_001)
        assert len(raw) <= 10_000_000
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        assert blob == expected
        path = R / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        assert not path.exists()
        path.write_bytes(raw)
        records.append({'path': name, 'url': url, 'git_blob': blob, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
    (R / 'SOURCE_MANIFEST.json').write_text(json.dumps({'revision': REV, 'files': records,
         'license': 'CC-BY-4.0', 'usage': 'EXTERNAL_EVALUATION_ONLY', 'model_calls': 0}, indent=2), encoding='utf-8')
    with (R / 'source/camber/privacylensplus.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    print(json.dumps({'rows': len(rows), 'fields': list(rows[0]),
                      'labels': dict(Counter(r['appropriateness_label'] for r in rows)),
                      'expansions': dict(Counter(r['expansion_type'] for r in rows)), 'example': rows[0]}))


if __name__ == '__main__':
    main()
