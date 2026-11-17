import copy
import unittest
from bridge_data import worlds, targets, relation_fact, conditional_target, build_records


class BridgeTest(unittest.TestCase):
    def test_received_history_and_edge_direction_are_necessary(self):
        obs = {'subject': 'person', 'committed_ticket': 'ticket', 'actions': ['PUBLISH_FULL', 'PUBLISH_MINIMAL'],
               'contract': {'providers': [{'ticket': 'ticket', 'links': [['ticket', 'key']]}], 'permission': False, 'binding_required': True},
               'delivered': [{'recipient': 'R', 'content': {'identities': [['person', 'ticket']]}},
                             {'recipient': 'Other', 'content': {'histories': [['key', 'secret']]}}]}
        self.assertEqual(relation_fact(obs, 'PUBLISH_FULL'), 0)
        obs['delivered'][1]['recipient'] = 'R'
        self.assertEqual(relation_fact(obs, 'PUBLISH_FULL'), 1)
        self.assertEqual(conditional_target(obs, 'PUBLISH_FULL', 1), 0)
        obs['contract']['permission'] = True
        self.assertEqual(relation_fact(obs, 'PUBLISH_FULL'), 1)
        self.assertEqual(conditional_target(obs, 'PUBLISH_FULL', 1), 1)
        obs['contract']['binding_required'] = False
        self.assertEqual(conditional_target(obs, 'PUBLISH_MINIMAL', 1), 1)
        obs['contract']['providers'][0]['links'][0].reverse()
        self.assertEqual(relation_fact(obs, 'PUBLISH_FULL'), 0)

    def test_fact_separates_relations_from_norm_and_task(self):
        for group in worlds().values():
            for world in group:
                for case in targets(world):
                    obs = case['observation']
                    for candidate in case['candidates']:
                        action = candidate['action']; fact = relation_fact(obs, action)
                        self.assertEqual(conditional_target(obs, action, fact), candidate['safe_completion'])
                        for flag in ['permission', 'binding_required']:
                            changed = copy.deepcopy(obs); changed['contract'][flag] = not changed['contract'][flag]
                            self.assertEqual(relation_fact(changed, action), fact)
                        changed = copy.deepcopy(obs)
                        changed['delivered'].append({'recipient': 'SomeoneElse', 'content': {'histories': [['not_received', 'secret']]}})
                        self.assertEqual(relation_fact(changed, action), fact)

    def test_complete_support_and_matched_fact_intervention(self):
        states, queries, labels = build_records()
        self.assertEqual((len(states), len(queries), len(labels)), (80, 192, 192))
        self.assertEqual({s['observation']['role'] for s in states}, {'Planner', 'Source', 'Worker'})
        for q, label in zip(queries, labels):
            self.assertEqual(q['id'], label['id'])
            a, b = q['prompts']['truth'], q['prompts']['flip']
            self.assertEqual(len(a), len(b))
            self.assertEqual(sum(x != y for x, y in zip(a, b)), 1)
            self.assertNotIn('Own estimates', a)
            self.assertEqual(label['sensitive'], label['truth_target'] != label['flip_target'])


if __name__ == '__main__': unittest.main()
