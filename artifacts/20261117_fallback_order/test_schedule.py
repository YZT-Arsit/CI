import unittest
from collections import Counter
from schedules import make_schedule


class ScheduleTests(unittest.TestCase):
    def test_matched_exposure_and_interleaving(self):
        expected=Counter([('correct',i) for i in range(27)]+[('replay',i) for i in range(80)])
        for arm in ['correct_then_replay','replay_then_correct','interleaved']:
            self.assertEqual(Counter(map(tuple,make_schedule(arm))),expected)
        mixed=make_schedule('interleaved')
        self.assertEqual(mixed[:4],[['replay',0],['replay',1],['replay',2],['correct',0]])
        self.assertEqual(mixed[-3:],[['replay',78],['replay',79],['correct',26]])
        self.assertEqual(make_schedule('correct_then_replay')[26],['correct',26])
        self.assertEqual(make_schedule('replay_then_correct')[79],['replay',79])


if __name__=='__main__':unittest.main()
