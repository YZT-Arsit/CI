"""The actual bit expression must treat opposing labels equally within an action pair.

Evaluate its source AST on conditional binary CE values at fixed binary mass.
No model or replacement implementation is used by this regression.
"""
import ast
import math
import sys
import unittest
from pathlib import Path
import numpy as np

SOURCE = Path(sys.argv.pop(1))
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'loss_for_views')
node = next(n for n in ast.walk(function) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'bit' for t in n.targets))
EXPRESSION = compile(ast.Expression(node.value), str(SOURCE), 'eval')


def value(log_odds, candidates):
    losses, labels = [], []
    for view in range(2):
        for side in range(2):
            for candidate in range(candidates):
                target = int(side == 0 and candidate == 0)
                loss = math.log1p(math.exp(-log_odds if target else log_odds)) if candidate == 0 else 0.
                losses.append([0., loss, 0.]); labels.append(target)
    qce = np.array(losses)
    positive = np.array(labels, dtype=bool)
    return float(eval(EXPRESSION, {}, {'qce': qce, 'positive': positive}))


class BitBalanceTests(unittest.TestCase):
    def test_opposing_action_pair_has_no_positive_log_odds_push(self):
        for candidates in [1, 2, 3]:
            with self.subTest(candidates=candidates):
                h = 1e-5
                self.assertAlmostEqual((value(h, candidates) - value(-h, candidates)) / (2 * h), 0., places=8)

    def test_opposing_action_pair_is_symmetric_after_adding_fixed_negative_actions(self):
        for candidates in [1, 2, 3]:
            with self.subTest(candidates=candidates):
                self.assertAlmostEqual(value(.7, candidates), value(-.7, candidates), places=12)


if __name__ == '__main__':
    unittest.main()
