"""Adversarial checks for the frozen scorer, with fabricated outputs, no model."""
import copy
import unittest
from graph_data import confirmation
from scoring import score


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.obs, self.labels = confirmation()

    def outputs(self, mode):
        return [{'id': o['id'], 'raw': 'ANSWER: ' + str(l['target']) if mode == 'binary' else 'PROOF: ' + l['valid_witnesses'][0],
                 'seconds_batch': .1, 'prompt_tokens': 100, 'padded_width': 100} for o, l in zip(self.obs, self.labels)]

    def test_perfect_both_modes(self):
        for mode in ['binary', 'proof']:
            result, _ = score(self.obs, self.labels, self.outputs(mode), mode)
            self.assertEqual(result['correct'], 480)
            self.assertEqual(result['paired_correct'], 144)
            self.assertEqual(result['random_all_four_groups'], 32)
            self.assertTrue(result['relation_gate_passed'])

    def test_constant_and_multi_answer_fail(self):
        for mode, text in [('binary', 'ANSWER: 1'), ('proof', 'PROOF: none')]:
            raw = self.outputs(mode)
            for r in raw: r['raw'] = text
            result, _ = score(self.obs, self.labels, raw, mode)
            self.assertEqual(result['correct'], 240)
            self.assertEqual(result['paired_correct'], 0)
            self.assertFalse(result['relation_gate_passed'])
        for mode, text in [('binary', 'ANSWER: 0 ANSWER: 1'), ('proof', 'PROOF: none PROOF: extra')]:
            raw = self.outputs(mode)
            for r in raw: r['raw'] = text
            result, _ = score(self.obs, self.labels, raw, mode)
            self.assertEqual(result['invalid'], 480)
            self.assertEqual(result['correct'], 0)

    def test_invalid_identity_timing_and_length(self):
        for mode in ['binary', 'proof']:
            raw = self.outputs(mode); broken = copy.deepcopy(raw); broken[0]['id'] = broken[1]['id']
            with self.assertRaises(AssertionError): score(self.obs, self.labels, broken, mode)
            for r in raw[:4]: r['seconds_batch'] = 9.5
            result, _ = score(self.obs, self.labels, raw, mode)
            self.assertEqual(result['invalid'], 4); self.assertFalse(result['relation_gate_passed'])
            raw = self.outputs(mode); raw[0]['prompt_tokens'] = 99
            result, _ = score(self.obs, self.labels, raw, mode)
            self.assertFalse(result['pair_token_control']); self.assertFalse(result['relation_gate_passed'])

    def test_diagnostic_names_exclude_field_labels(self):
        from diagnostics import present_names
        prompt = 'Known nodes: apple.\nEdges: apple -> pear; plum -> peach.\nTarget: pear. Give one known node'
        self.assertEqual(present_names(prompt), {'apple', 'pear', 'plum', 'peach'})
        self.assertEqual(present_names('Known nodes: none.\nEdges: apple -> pear.\nTarget: plum. Give one known node'), {'apple', 'pear', 'plum'})


if __name__ == '__main__': unittest.main()
