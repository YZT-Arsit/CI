"""Pin external evaluation data; never add it to the training corpus."""
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

R = Path(__file__).resolve().parent
REVISION = '9c2ee07b080dc54ed4924af11d9751e81753c94d'
HF_REVISION = '7dc406e8d1e882e5b0c6ea4be697b2b7a5dee805'
FILES = {'data/main_data.json': '7992bddb2638bb1fa595c4bf8fb4c82783209e6f',
         'README.md': '2de1b8f0cafb7bc7bb1e406e59531421cfdbffd0',
         'LICENSE': 'ff234b056d4f332ad972ea70978aa0e684b59b52'}


def pin_card():
    url = f'https://huggingface.co/datasets/SALT-NLP/PrivacyLens/raw/{HF_REVISION}/README.md'
    card = urllib.request.urlopen(url, timeout=20).read()
    assert b'license: cc-by-4.0' in card and b'please do not use it directly for training' in card
    path = R / 'source/DATASET_CARD.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == card
    else:
        path.write_bytes(card)
    return {'dataset_card_url': url, 'dataset_card_revision': HF_REVISION,
            'dataset_card_snapshot_path': path.relative_to(R).as_posix(),
            'dataset_card_snapshot_sha256': hashlib.sha256(card).hexdigest()}


def main():
    assert not (R / 'SOURCE_MANIFEST.json').exists()
    records = []
    for name, expected in FILES.items():
        url = f'https://raw.githubusercontent.com/SALT-NLP/PrivacyLens/{REVISION}/{name}'
        data = urllib.request.urlopen(url, timeout=20).read()
        git_blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert git_blob == expected
        target = R / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == data
        else:
            target.write_bytes(data)
        records.append({'path': target.relative_to(R).as_posix(), 'url': url, 'bytes': len(data),
                        'git_blob': git_blob, 'sha256': hashlib.sha256(data).hexdigest()})
    card = pin_card()
    result = {'repository': 'SALT-NLP/PrivacyLens', 'revision': REVISION, 'files': records,
              'retrieved_utc': datetime.now(timezone.utc).isoformat(),
              'dataset_license': 'CC-BY-4.0', 'code_license': 'MIT',
              **card,
              'use': 'EXTERNAL_EVALUATION_ONLY', 'training_use': False,
              'author_training_recommendation': 'please do not use it directly for training',
              'paper': 'https://arxiv.org/html/2409.00138v3', 'whole_goal_complete': False}
    (R / 'SOURCE_MANIFEST.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'files': len(records), 'bytes': sum(r['bytes'] for r in records), 'revision': REVISION, 'training_use': False}))


if __name__ == '__main__':
    main()
