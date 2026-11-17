"""Finite counterexample: action consistency is stronger than value consistency."""
import hashlib
import json
import math
from pathlib import Path

R = Path(__file__).resolve().parent


def js(p, q):
    assert len(p) == len(q) and abs(sum(p) - 1) < 1e-12 and abs(sum(q) - 1) < 1e-12
    middle = [(a + b) / 2 for a, b in zip(p, q)]
    return .5 * sum(a * math.log(a / m) for a, m in zip(p, middle) if a) + .5 * sum(b * math.log(b / m) for b, m in zip(q, middle) if b)


def classes(probabilities, values):
    assert len(probabilities) == len(values) + 1
    return [sum(p for p, v in zip(probabilities, values) if v == key) for key in sorted(set(values))] + [probabilities[-1]]


def main():
    good, safe_failure, unsafe = (1, 1, 0), (1, 0, 0), (0, 0, 0)
    p, q = [.9702, .0098, .02], [.0098, .9702, .02]
    tied_full = js(p, q)
    tied_value = js(classes(p, [good, good]), classes(q, [good, good]))
    strict_value = js(classes(p, [good, unsafe]), classes(q, [good, unsafe]))
    assert tied_full > .5 and tied_value == 0 and abs(strict_value - tied_full) < 1e-12
    a, b = [.882, .009, .099, .01], [.009, .882, .099, .01]
    c = [.009, .099, .882, .01]
    assert js(classes(a, [good, good, unsafe]), classes(b, [good, good, unsafe])) == 0
    assert js(classes(a, [good, good, unsafe]), classes(c, [good, good, unsafe])) > 0
    assert len(classes(a, [good, safe_failure, unsafe])) == 4
    result = {'tied_action_js': tied_full, 'tied_value_class_js': tied_value,
              'strict_value_class_js': strict_value, 'unsafe_mass_changes_remain_visible': True,
              'safety_and_completion_classes_not_merged': True, 'OTHER_mass_retained': True,
              'model_calls': 0, 'optimizer_updates': 0,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Mathematical diagnostic only. Penalizing tied-action changes is not implied by equal outcome values; this does not prove it caused model failure or that value-class consistency improves training.'}
    (R / 'VALUE_CLASS_CHECK.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
