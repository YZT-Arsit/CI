import unittest
from kci.action_binding_metrics import pair_metrics


def rows(left='left',right='right'):
    return [{'pair_id':'p','side':0,'selected':left,'optimal_action':'left','actions':['left','right']},
            {'pair_id':'p','side':1,'selected':right,'optimal_action':'right','actions':['left','right']}]


class BindingMetricTests(unittest.TestCase):
    def test_constant_route_does_not_pass(self):
        result=pair_metrics(rows('left','left'))
        self.assertEqual(result['both_correct'],0)
        self.assertEqual(result['same_action'],1)

    def test_correct_reversal_passes(self):
        self.assertEqual(pair_metrics(rows())['both_correct'],1)

    def test_wrong_reversal_is_not_relational_success(self):
        result=pair_metrics(rows('right','left'))
        self.assertEqual(result['both_correct'],0)
        self.assertEqual(result['reversed_both_wrong'],1)

    def test_invalid_output_cannot_earn_pair_success(self):
        result=pair_metrics(rows(None,'right'))
        self.assertEqual(result['both_correct'],0)
        self.assertEqual(result['invalid_groups'],1)

    def test_misaligned_pair_fails_instead_of_silently_dropping(self):
        data=rows();data[1]['pair_id']='other'
        with self.assertRaises(ValueError):pair_metrics(data)

    def test_duplicate_side_is_rejected(self):
        data=rows();data[1]['side']=0
        with self.assertRaises(ValueError):pair_metrics(data)

    def test_outside_action_set_is_rejected(self):
        with self.assertRaises(ValueError):pair_metrics(rows('not-an-action','right'))


if __name__=='__main__':unittest.main()
