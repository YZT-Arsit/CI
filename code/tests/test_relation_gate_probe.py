import unittest
from copy import deepcopy
from dataclasses import asdict
from experiments.relation_gate_probe import BindingProbe,make_map,rename,independent_join,SOURCE
from kci.legacy_dynamic_tasks_v1 import DynamicTask


class RelationProbeTests(unittest.TestCase):
    def test_maps_are_bijections_and_preserve_literal_lengths(self):
        for i in range(34):
            m=make_map(i);self.assertEqual(len(m),len(set(m.values())))
            self.assertTrue(all(len(str(k))==len(str(v)) for k,v in m.items()))
            sample=list(m);self.assertEqual(rename(rename(sample,m),{v:k for k,v in m.items()}),sample)

    def test_identity_transport_matches_old_transition(self):
        for kind in ('meeting','query'):
            old=DynamicTask(kind,SOURCE,related=True,horizon=12)
            new=BindingProbe(kind,True,12);a=old.decision_start();b=new.decision_start()
            self.assertEqual(asdict(a),asdict(b));self.assertEqual(old.observation(a),new.observation(b))
            for action in old.exact(a)['witness']:
                a,x=old.transition(a,action);b,y=new.transition(b,action)
                self.assertEqual(asdict(a),asdict(b));self.assertEqual(x,y)

    def test_uniform_rename_preserves_values_and_full_trace(self):
        for kind in ('meeting','query'):
          for related in (False,True):
            plain=BindingProbe(kind,related,12);m=make_map(2);mapped=BindingProbe(kind,related,12,m)
            a=plain.decision_start();b=mapped.decision_start()
            self.assertEqual(plain.exact(a),mapped.exact(b))
            for action in plain.exact(a)['witness']:
                a,_=plain.transition(a,action);b,_=mapped.transition(b,action)
                self.assertEqual(rename(asdict(a),m),asdict(b))

    def test_binding_reversal_changes_privacy_not_current_legality(self):
        for kind in ('meeting','query'):
            verdicts=[]
            for flip in (False,True):
                e=BindingProbe(kind,True,12,route_flip=flip);s=e.decision_start()
                actions=('SEND_ROSTER','SEND_CASE_LINK') if kind=='meeting' else ('OPEN_INDEX','FETCH_CITY_BUNDLE')
                s,a=e.transition(s,actions[0]);self.assertEqual(a['verdict'],'VALID')
                s,a=e.transition(s,actions[1]);verdicts.append(a['verdict'])
                self.assertEqual(e.base.protected(s.ledger),independent_join(kind,s.ledger))
            self.assertEqual(verdicts,['VIOLATION','VALID'])

    def test_wrapper_does_not_mutate_input(self):
        e=BindingProbe('query',True,12,make_map(1),True);s=e.decision_start();saved=deepcopy(s)
        e.transition(s,'OPEN_INDEX');self.assertEqual(s,saved)


if __name__=='__main__':unittest.main()
