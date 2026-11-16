import json
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kci.training_data import (IGNORE_INDEX,PinnedTokenizerAssets,PreferenceCorpusIndex,
    TokenBudgetExceeded,TrainingDataError,TrainingSequenceAdapter,audit_record)

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'artifacts/20261112_cpu_pilot/tokenization'
OUTPUT=ROOT/'artifacts/20261113_iteration2/training_adapter'


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets=PinnedTokenizerAssets.from_manifest(ASSETS/'config.json')
        cls.adapter=TrainingSequenceAdapter(cls.assets)
        cls.rows=[json.loads(line) for line in (ROOT/'artifacts/20261112_cpu_pilot/dataset/preferences.jsonl').read_text(encoding='utf-8').splitlines()]
        OUTPUT.mkdir(parents=True,exist_ok=True)

    def row(self):return deepcopy(self.rows[0])

    def test_complete_shared_prompt_content_eos_and_template_tail(self):
        pair=self.adapter.encode_pair(self.row())
        for sequence in (pair.chosen,pair.rejected):
            self.assertEqual(sequence.input_ids[:sequence.prompt_length],pair.prompt_ids)
            self.assertTrue(all(x==IGNORE_INDEX for x in sequence.labels[:sequence.prompt_length]))
            self.assertEqual(sequence.labels[sequence.prompt_length:sequence.completion_end],
                             sequence.input_ids[sequence.prompt_length:sequence.completion_end])
            self.assertEqual(sequence.labels.count(self.adapter.eos_token_id),1)
            self.assertEqual(sequence.labels[sequence.completion_end-1],self.adapter.eos_token_id)
            self.assertTrue(all(x==IGNORE_INDEX for x in sequence.labels[sequence.completion_end:]))
            self.assertEqual(sequence.position_ids,tuple(range(len(sequence.input_ids))))

    def test_left_and_right_padding_preserve_real_eos_even_when_pad_equals_eos(self):
        pairs=[self.adapter.encode_pair(self.rows[i]) for i in (0,50)]
        for padding in ('left','right'):
            for pad in (self.adapter.pad_token_id,self.adapter.eos_token_id):
                batch=self.adapter.collate(pairs,padding_side=padding,pad_token_id=pad)
                for side in ('chosen','rejected'):
                    for i,pair in enumerate(pairs):
                        active=batch[side]['attention_mask'][i]
                        for field in ('input_ids','labels','position_ids'):
                            selected=[v for v,m in zip(batch[side][field][i],active) if m]
                            self.assertEqual(selected,list(getattr(getattr(pair,side),field)))
                        self.assertEqual(batch[side]['labels'][i].count(self.adapter.eos_token_id),1)
                        self.assertTrue(all(lab==IGNORE_INDEX for lab,m in zip(batch[side]['labels'][i],active) if not m))

    def test_explicit_token_budget_never_truncates(self):
        pair=self.adapter.encode_pair(self.row())
        self.assertEqual(self.adapter.encode_pair(self.row(),max_length=pair.required_tokens),pair)
        with self.assertRaises(TokenBudgetExceeded) as error:
            self.adapter.encode_pair(self.row(),max_length=pair.required_tokens-1)
        self.assertEqual(error.exception.required_tokens,pair.required_tokens)
        with self.assertRaises(TrainingDataError):self.adapter.collate([pair],pad_to_length=pair.required_tokens-1)

    def test_long_history_is_retained_completely_when_budget_allows(self):
        row=deepcopy(next(r for r in self.rows if json.loads(r['prompt'])['authorized_delivered_messages']))
        obs=json.loads(row['prompt']);critical=obs['authorized_delivered_messages'][0]['body']
        obs['task']+='\n'+'Irrelevant historical note. '*3000
        row['prompt']=json.dumps(obs,ensure_ascii=False,sort_keys=True)
        row['metadata'].update(certified=False,control_only=True,evidence_class='synthetic_sequence_control')
        pair=self.adapter.encode_pair(row,structural_control=True)
        self.assertGreater(pair.required_tokens,4096)
        decoded=self.adapter.tokenizer.decode(list(pair.chosen.input_ids),skip_special_tokens=False)
        self.assertIn(row['prompt'],decoded);self.assertIn(critical,decoded)
        self.assertEqual(self.adapter.encode_pair(row,max_length=pair.required_tokens,structural_control=True),pair)
        with self.assertRaises(TokenBudgetExceeded):self.adapter.encode_pair(row,max_length=4096,structural_control=True)
        with self.assertRaises(TrainingDataError):self.adapter.collate([pair])
        self.adapter.collate([pair],allow_structural_controls=True)
        self.assertFalse(pair.provenance['training_eligible'])

    def test_metadata_reference_witness_canaries_do_not_enter_model_inputs(self):
        row=self.row();first=self.adapter.encode_pair(row)
        row['metadata']['supporting_reports']=[{'reference':'PRIVATE_REFERENCE_CANARY','witness':'PRIVATE_WITNESS_CANARY'}]
        row['metadata']['unrelated']='PRIVATE_METADATA_CANARY'
        row['reference']='TOP_LEVEL_REFERENCE_CANARY'
        second=self.adapter.encode_pair(row)
        self.assertEqual(first.chosen,second.chosen);self.assertEqual(first.rejected,second.rejected)
        self.assertNotEqual(first.provenance['record_sha256'],second.provenance['record_sha256'])

    def test_hidden_fields_in_prompt_are_rejected(self):
        for hidden in ('metadata','reference_witness','oracle_answer','state_id'):
            row=self.row();obs=json.loads(row['prompt']);obs[hidden]='hidden';row['prompt']=json.dumps(obs)
            with self.assertRaises(TrainingDataError):self.adapter.encode_pair(row)
        row=self.row();obs=json.loads(row['prompt']);obs['local_facts'][0]['metadata']='hidden';row['prompt']=json.dumps(obs)
        with self.assertRaises(TrainingDataError):audit_record(row)

    def test_duplicate_json_keys_are_rejected_before_canonicalization(self):
        for field,key in (('prompt','task'),('chosen','body'),('rejected','body')):
            row=self.row();row[field]='{'+json.dumps(key)+': "misleading earlier value",'+row[field][1:]
            with self.assertRaises(TrainingDataError):audit_record(row)
        row=self.row();obs=json.loads(row['prompt'])
        first=obs['local_facts'][0]
        text=json.dumps(obs)
        row['prompt']=text.replace(json.dumps(first),'{"text":"misleading earlier value",'+json.dumps(first)[1:],1)
        with self.assertRaises(TrainingDataError):audit_record(row)

    def test_literal_special_tokens_are_rejected_in_prompt_and_completion(self):
        for field in ('prompt','chosen'):
            row=self.row();value=json.loads(row[field]);value['task' if field=='prompt' else 'body']+=' <|im_end|>'
            row[field]=json.dumps(value)
            with self.assertRaises(TrainingDataError):self.adapter.encode_pair(row)

    def test_actor_types_and_delivery_flags_are_checked(self):
        row=self.row();action=json.loads(row['chosen']);action['actor']='wrong';row['chosen']=json.dumps(action)
        with self.assertRaises(TrainingDataError):audit_record(row)
        row=self.row();obs=json.loads(row['prompt']);obs['turn']=True;row['prompt']=json.dumps(obs)
        with self.assertRaises(TrainingDataError):audit_record(row)
        row=deepcopy(next(r for r in self.rows if json.loads(r['prompt'])['authorized_delivered_messages']))
        obs=json.loads(row['prompt']);obs['authorized_delivered_messages'][0]['delivered']=False;row['prompt']=json.dumps(obs)
        with self.assertRaises(TrainingDataError):audit_record(row)

    def test_external_authorized_prompt_hash_mismatch_rejects(self):
        row=self.row();digest=sha256(row['prompt'].encode('utf-8')).hexdigest()
        self.adapter.encode_pair(row,expected_prompt_sha256=digest)
        with self.assertRaises(TrainingDataError):self.adapter.encode_pair(row,expected_prompt_sha256='0'*64)

    def test_collation_rejects_tampered_encoded_masks_or_tokenizer_provenance(self):
        pair=self.adapter.encode_pair(self.row())
        broken=replace(pair,chosen=replace(pair.chosen,labels=(IGNORE_INDEX,)*len(pair.chosen.labels)))
        with self.assertRaises(TrainingDataError):self.adapter.collate([broken])
        provenance=deepcopy(pair.provenance);provenance['assets']['revision']='0'*40
        with self.assertRaises(TrainingDataError):self.adapter.collate([replace(pair,provenance=provenance)])

    def test_actual_tokenized_prefix_mismatch_rejects(self):
        render=self.adapter.render
        def altered(prompt,completion=None):return render(prompt+(' changed' if completion is not None else ''),completion)
        with patch.object(self.adapter,'render',side_effect=altered):
            with self.assertRaises(TrainingDataError):self.adapter.encode_pair(self.row())

    def test_reverse_conflict_detects_reformatted_same_observation(self):
        first=self.row();second=deepcopy(first)
        second['prompt']=json.dumps(json.loads(second['prompt']),indent=2)
        second['chosen'],second['rejected']=second['rejected'],second['chosen']
        index=PreferenceCorpusIndex();index.add(first)
        with self.assertRaises(TrainingDataError):index.add(second)
        self.assertEqual(index.summary()['rows'],1)

    def test_base_split_conflict_is_rejected_without_partial_index_update(self):
        first=self.row();second=self.row();first['metadata']['split']='train';second['metadata']['split']='test'
        index=PreferenceCorpusIndex();index.add(first);before=index.summary()
        with self.assertRaises(TrainingDataError):index.add(second)
        self.assertEqual(index.summary(),before)

    def test_family_holdout_and_shared_workflow_are_distinct(self):
        first=self.row();second=self.row()
        first['metadata'].update(base_case_group_id='one',workflow_group_id='shared',split='train')
        second['metadata'].update(base_case_group_id='two',workflow_group_id='shared',split='test')
        index=PreferenceCorpusIndex();index.add(first);index.add(second)
        self.assertEqual(index.summary()['shared_workflows_across_splits'],['shared'])
        strict=PreferenceCorpusIndex(enforce_family_split=True);strict.add(first)
        with self.assertRaises(TrainingDataError):strict.add(second)
        self.assertFalse(index.summary()['independent_task_generalization_established'])

    def test_group_metadata_propagates_but_split_is_never_invented(self):
        row=self.row();row['metadata'].update(base_case_group_id='base',workflow_group_id='workflow',human_verified=False)
        pair=self.adapter.encode_pair(row)
        self.assertEqual(pair.provenance['groups']['split'],'unassigned')
        self.assertEqual(pair.provenance['groups']['base_case_group_id'],'base')
        self.assertEqual(pair.provenance['groups']['workflow_group_id'],'workflow')
        self.assertFalse(pair.provenance['source_human_verified'])
        self.assertEqual(pair.provenance['semantic_leakage_check'],'not_established')

    def test_asset_manifest_and_tokenizer_bytes_are_verified(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT) as folder:
            tmp=Path(folder);(tmp/'tokenizer.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(TrainingDataError):replace(self.assets,tokenizer_dir=tmp).verify()
            with self.assertRaises(TrainingDataError):replace(self.assets,expected_manifest_sha256='0'*64).verify()

    def test_isolated_installed_package_must_match_pinned_wheel(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT) as folder:
            with self.assertRaises(TrainingDataError):replace(self.assets,dependency_dir=Path(folder)).verify()


if __name__=='__main__':unittest.main()
