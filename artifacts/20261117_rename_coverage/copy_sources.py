"""Two bounded source adaptations, verified again by the data auditor."""
from pathlib import Path
R = Path(__file__).resolve().parent


def sources():
    old = (R.parent / '20261117_query_coverage/eval_worlds.py').read_text()
    assert old.count('400000000') == old.count('400010000') == 1
    yield 'fresh_worlds.py', old.replace('400000000', '430000000').replace('400010000', '430010000')
    old = (R.parent / '20261117_teacher_binding/replay.py').read_text()
    text = "pool=json.loads((OLD/'entity_palettes.json').read_text())['fresh']"
    assert old.count(text) == 1
    yield 'native_replay.py', old.replace(text, "pool=json.loads((R/'palettes.json').read_text())['new']")


if __name__ == '__main__':
    for name, text in sources():
        assert not (R / name).exists()
        (R / name).write_text(text, encoding='utf-8')
