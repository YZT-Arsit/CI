import copy,unittest
from experiments.relation_solution_data import training_world,intervene,environment
from kci.relation_language import prompt,vocabulary
from kci.extended_meeting_tasks import ExtendedMeetingKernel
from kci.dynamic_tasks import TaskEvidenceError

class LanguageInputTests(unittest.TestCase):
    def setUp(self):
        self.world=training_world(0);self.env=environment(self.world);self.obs=self.env.observation(self.env.initial());self.palette=['word'+str(i) for i in range(128)]

    def test_swapped_pair_preserves_vocabulary_and_option_order(self):
        other=environment(intervene(self.world,'swap'));obs=other.observation(other.initial())
        self.assertEqual(vocabulary(self.obs,self.palette,17),vocabulary(obs,self.palette,17))
        a,order=prompt(self.obs,self.palette,17);b,otherorder=prompt(obs,self.palette,17)
        self.assertEqual(order,otherorder)
        # This training cell may contain an undelivered history, so use a delivered one.
        self.world['history_delivered']=True;left=environment(self.world);right=environment(intervene(self.world,'swap'))
        a,_=prompt(left.observation(left.initial()),self.palette,17);b,_=prompt(right.observation(right.initial()),self.palette,17)
        self.assertNotEqual(a,b);self.assertEqual(a.split('R received:')[0],b.split('R received:')[0]);self.assertEqual(a.split('Plans:')[1],b.split('Plans:')[1])

    def test_private_worker_values_and_evaluator_fields_never_rendered(self):
        a,_=prompt(self.obs,self.palette,17);obs=copy.deepcopy(self.obs)
        obs.update(expected='PRIVATE_EXPECTED',safe_success='PRIVATE_LABEL')
        obs['delivered'].append({'recipient':'Worker','content':{'value':'PRIVATE_SECRET_VALUE'}})
        b,_=prompt(obs,self.palette,17);self.assertEqual(a,b)

    def test_no_entity_truncation(self):
        with self.assertRaisesRegex(ValueError,'palette exhausted'):prompt(self.obs,['one'],17)

class ExtendedConstraintTests(unittest.TestCase):
    def setUp(self):
        self.public={'rooms':[{'id':'r','capacity':2,'features':['hybrid']}],
            'slots':[{'id':'m','day':'Monday','start':10.,'end':11.},{'id':'t','day':'Tuesday','start':10.,'end':11.}],
            'meetings':[{'id':'x','candidate_slots':['m','t'],'candidate_rooms':['r'],'required_attendees':['p'],'duration_minutes':60}]}
        self.result={'x':{'slot':'m','room':'r'}}

    def test_room_feature_and_missing_feature(self):
        c={'person':'p','type':'meeting_requires_room_feature','meeting_id':'x','room_feature':'hybrid'}
        ExtendedMeetingKernel.validate_constraints([c]);self.assertTrue(ExtendedMeetingKernel.schedule_valid(self.result,[c],self.public))
        c['room_feature']='quiet';self.assertFalse(ExtendedMeetingKernel.schedule_valid(self.result,[c],self.public))

    def test_deadline_boundary_and_day(self):
        c={'person':'p','type':'must_finish_before_deadline','meeting_id':'x','deadline_day':'Monday','deadline_hour':11.}
        ExtendedMeetingKernel.validate_constraints([c]);self.assertTrue(ExtendedMeetingKernel.schedule_valid(self.result,[c],self.public))
        c['deadline_hour']=10.99;self.assertFalse(ExtendedMeetingKernel.schedule_valid(self.result,[c],self.public))
        c['deadline_hour']=11.;self.result['x']['slot']='t';self.assertFalse(ExtendedMeetingKernel.schedule_valid(self.result,[c],self.public))

    def test_invalid_deadline_and_extra_raw_fields(self):
        c={'person':'p','type':'must_finish_before_deadline','meeting_id':'x','deadline_day':'Monday','deadline_hour':25.}
        with self.assertRaises(TaskEvidenceError):ExtendedMeetingKernel.validate_constraints([c])
        c['deadline_hour']=11.;c['raw']='private reason'
        with self.assertRaises(TaskEvidenceError):ExtendedMeetingKernel.validate_constraints([c])

class SourceEncodingTests(unittest.TestCase):
    def test_future_obligation_visible_at_source(self):
        try:from kci.relation_solution import facts_and_flags
        except ImportError:self.skipTest('requires isolated CPU torch environment')
        from kci.receipt_policy import candidate_view
        w=training_world(0);w['binding_required']=True;env=environment(w);s=env.initial();s,_=env.transition(s,env.actions(s)[0]);obs=env.observation(s)
        self.assertEqual(candidate_view(obs,'DELIVER_INPUTS')[1]['required_future_link'],[])
        facts,_=facts_and_flags(obs,'DELIVER_INPUTS')
        self.assertEqual([(a,b) for k,a,b in facts if k==2],[tuple(e) for e in env.provider(s.ticket)['links']])
        self.assertFalse(any(k==2 for k,a,b in facts_and_flags(obs,'WITHHOLD')[0]))

if __name__=='__main__':unittest.main()
