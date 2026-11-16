"""Never silently fabricate dynamic supervision from static third-party datasets."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Mapping,Any
from .contracts import Preference

class DataContractError(ValueError):
    def __init__(self,message:str,*,conflicting_records:tuple[dict,...]=()):
        super().__init__(message)
        self.conflicting_records=conflicting_records

@dataclass(frozen=True)
class DataReadiness:
    source:str
    reusables:tuple[str,...]
    must_create:tuple[str,...]
    train_ready:bool

# Extracted from actual PiSAs JSON schema, not a claim of model training readiness.
PISAS_REUSABLE={
    'scenario.json':{'scenario_id','task','cast','org','timeline'},
    'utility.json':{'scenario_id','oracle_answer','decision_rule','attribute_values'},
    'appropriateness.json':{'scenario_id','attributes'},
    'visibility.json':{'scenario_id','attributes'},
}
KCI_REQUIRED_NEW={
    'delivered_recipient_ledger',
    'structured_compositional_inference_rules',
    'ci_policy_for_new_derived_facts',
    'agent_observation_filter',
    'legal_action_transition_model',
    'next_step_candidate_actions',
    'action_level_validation',
    'verifiable_task_success',
    'reference_continuation_evidence',
}

def audit_pisas_files(files:Mapping[str,Mapping[str,Any]])->DataReadiness:
    for filename,required in PISAS_REUSABLE.items():
        if filename not in files:raise DataContractError(f'missing {filename}')
        missing=required-set(files[filename])
        if missing:raise DataContractError(f'{filename}: missing fields {sorted(missing)}')
    # A PiSAs `timeline` describes initial artifacts; it is NOT a delivery ledger.
    return DataReadiness('PiSAs static scenario',
        ('roles','initial_artifacts','attribute_appropriateness','visibility','final_task_oracle'),
        tuple(sorted(KCI_REQUIRED_NEW)),False)


def validate_enriched_episode(metadata:Mapping[str,Any])->None:
    """Structural gates; semantic/authorization proof requires environment replay.

    No field is inferred from a static PiSAs `timeline` or MultiAgentPrivacy's
    table sensitivity labels. `reference_continuation_evidence` is mandatory
    only for a claimed future-value preference; this v0.5 contract targets
    that full training pathway.
    """
    missing=KCI_REQUIRED_NEW-set(metadata)
    if missing:raise DataContractError('Missing new KCI fields: '+', '.join(sorted(missing)))
    for field in ('schema_version','scenario_group_id','rule_family_id','norm_version','task_oracle_version'):
        if not isinstance(metadata.get(field),str) or not metadata[field]:
            raise DataContractError('missing nonempty version/group field: '+field)
    ledger=metadata['delivered_recipient_ledger']
    if not isinstance(ledger,list):raise DataContractError('delivered_recipient_ledger must be a list')
    event_ids=set()
    for e in ledger:
        fields={'event_id','sender','recipient','fact_ids','body','purpose','semantic_verifier_version','delivered'}
        if not isinstance(e,dict) or not fields.issubset(e):
            raise DataContractError('delivered event missing ID, routing, body, purpose, semantics or delivery fields')
        for field in fields-{'fact_ids','delivered'}:
            if not isinstance(e[field],str) or not e[field]:raise DataContractError('empty event field: '+field)
        if e['event_id'] in event_ids:raise DataContractError('duplicate delivered event_id')
        event_ids.add(e['event_id'])
        if e['delivered'] is not True:
            raise DataContractError('un-delivered candidate cannot appear in delivered ledger')
        if not isinstance(e['fact_ids'],list) or any(not isinstance(f,str) or not f for f in e['fact_ids']):
            raise DataContractError('event fact_ids must be a list of strings')
    if metadata.get('requires_historical_dependency') and not ledger:
        raise DataContractError('history-dependent example requires delivered messages')
    rules=metadata['structured_compositional_inference_rules']
    if not isinstance(rules,list):raise DataContractError('inference rules must be a list')
    for rule in rules:
        if not isinstance(rule,dict) or not {'premises','conclusion','verifier'}.issubset(rule):
            raise DataContractError('inference rule lacks premises/conclusion/verifier')
    policy=metadata['ci_policy_for_new_derived_facts']
    if not isinstance(policy,list) or not policy:raise DataContractError('derived knowledge needs nonempty scoped CI policy')
    for entry in policy:
        if not isinstance(entry,dict) or not {'policy_id','recipient','purpose','predicate','allowed'}.issubset(entry):
            raise DataContractError('derived-fact CI policy lacks recipient/purpose/attribute/permission')
        if not isinstance(entry['allowed'],bool):raise DataContractError('allowed must be boolean')
    for field in ('agent_observation_filter','legal_action_transition_model','verifiable_task_success'):
        item=metadata[field]
        if not isinstance(item,dict) or not item:
            raise DataContractError(field+' must be a nonempty verifiable object')
    actions=metadata['next_step_candidate_actions']
    if not isinstance(actions,list) or not actions:raise DataContractError('candidate action list missing')
    action_ids=set()
    for action in actions:
        if not isinstance(action,dict) or not isinstance(action.get('action_id'),str) or not action['action_id']:
            raise DataContractError('candidate requires a nonempty action_id')
        if action['action_id'] in action_ids:raise DataContractError('duplicate action_id')
        action_ids.add(action['action_id'])
    if not isinstance(metadata['action_level_validation'],dict) or not metadata['action_level_validation']:
        raise DataContractError('per-action validation evidence missing')
    ref=metadata['reference_continuation_evidence']
    if not isinstance(ref,list) or not ref:raise DataContractError('reference continuation evidence missing')
    for item in ref:
        if not isinstance(item,dict) or not {'action_id','status','verifier_version'}.issubset(item):
            raise DataContractError('continuation evidence lacks action/status/verifier version')
        if item['status'] not in {'exact_within_horizon','proven_unreachable_within_horizon',
                                 'verified_witness_only','unknown','estimated_preference','paired_statistical_estimate'}:
            raise DataContractError('invalid reference evidence status')
        if item['action_id'] not in action_ids:raise DataContractError('reference names an unknown candidate')
        if not isinstance(item['verifier_version'],str) or not item['verifier_version']:
            raise DataContractError('reference verifier_version must be nonempty')
        if item['status'] in {'exact_within_horizon','proven_unreachable_within_horizon'}:
            if not isinstance(item.get('horizon'),int) or item['horizon']<0:
                raise DataContractError('bounded reference requires an explicit horizon')
            budget=item.get('search_budget')
            if (not isinstance(budget,dict) or any(not isinstance(budget.get(k),int) or budget[k]<1
                                                 for k in ('max_nodes','max_actions')) or
                    not isinstance(item.get('generator_version'),str) or not item['generator_version']):
                raise DataContractError('bounded reference requires search budget and generator version')
        if item['status'] in {'exact_within_horizon','verified_witness_only'}:
            if not isinstance(item.get('witness_actions'),list):
                raise DataContractError('successful reference requires a witness')
            cost=item.get('safe_success_cost')
            if not isinstance(cost,(int,float)) or not isfinite(cost) or cost<0:
                raise DataContractError('successful reference requires a finite nonnegative cost')
            if not item['witness_actions']:
                terminal=item.get('terminal_state')
                if (not isinstance(terminal,dict) or terminal.get('done') is not True or
                        terminal.get('success') is not True or terminal.get('any_violation') is not False or
                        not isinstance(item.get('after_state_id'),str) or not item['after_state_id'] or
                        terminal.get('state_id')!=item['after_state_id'] or
                        terminal.get('norm_version')!=metadata['norm_version'] or
                        terminal.get('oracle_version')!=metadata['task_oracle_version'] or
                        terminal.get('cost')!=cost):
                    raise DataContractError('empty success suffix requires matching terminal safe-success state evidence')
        if item['status'] in {'estimated_preference','paired_statistical_estimate'}:
            n=item.get('samples');delta=item.get('delta');seeds=item.get('sample_seeds')
            if not isinstance(n,int) or n<2 or not isinstance(delta,(int,float)) or not 0<delta<1:
                raise DataContractError('statistical evidence requires samples and delta')
            if (not isinstance(seeds,list) or len(seeds)!=n or any(not isinstance(s,int) for s in seeds)
                    or len(set(seeds))!=n):
                raise DataContractError('statistical evidence requires the distinct sampled seed list')
            if not item.get('reference_policy_version') or item.get('epistemic_status')!='statistical_estimate_NOT_optimality_proof':
                raise DataContractError('statistical evidence requires policy version and evidence status')
            interval=item.get('confidence_interval')
            if (not isinstance(item.get('comparison_metric'),str) or not item['comparison_metric'] or
                    not isinstance(interval,list) or len(interval)!=2 or
                    any(not isinstance(v,(int,float)) or not isfinite(v) for v in interval) or interval[0]>interval[1]):
                raise DataContractError('statistical evidence requires comparison metric and confidence interval')
    # Additional imperative checks: replay ledger, verify source permissions,
    # check semantic claims vs body, run task oracle, and re-evaluate Q bounds.


def export_verified_preferences(preferences:list[Preference])->list[dict]:
    seen=set();out=[];orientations={}
    for pref in preferences:
        key=(pref.state_id,pref.chosen.key(),pref.rejected.key())
        if key in seen:continue
        record=pref.to_dpo_record()
        audit=record['metadata']
        audit['export_schema_version']='kci_preferences_0.6_cpu'
        for field in ('prompt','chosen','rejected'):
            audit[field+'_sha256']=sha256(record[field].encode('utf-8')).hexdigest()
        canonical=json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        audit['record_sha256']=sha256(canonical.encode('utf-8')).hexdigest()
        orientation=(record['prompt'],record['chosen'],record['rejected'])
        reverse=(record['prompt'],record['rejected'],record['chosen'])
        if reverse in orientations:
            previous=orientations[reverse]
            raise DataContractError('Same authorized prompt has strictly reversed preferences at states '+
                                    previous['metadata']['state_id']+' and '+pref.state_id,
                                    conflicting_records=(previous,record))
        orientations[orientation]=record
        out.append(record);seen.add(key)
    return out


def deterministic_group_split(records:list[dict],seed:str='2026',train:float=.8,val:float=.1,*,
                              held_out_rule_families:Mapping[str,str]|None=None)->dict[str,list[dict]]:
    """Keep all counterfactual variants of a base world in one split.

    For publication-grade tests, use an explicit held-out *rule family* grouping
    when generalization across new inference mechanisms is the claim.
    """
    if not 0<train<1 or not 0<=val<1 or train+val>=1:
        raise ValueError('Invalid split proportions')
    splits={'train':[],'validation':[],'test':[]}
    group_split={}
    family_targets=dict(held_out_rule_families or {})
    if any(v not in {'validation','test'} for v in family_targets.values()):
        raise DataContractError('held-out families must target validation or test')
    forced_groups={}
    if family_targets:
        for r in records:
            meta=r.get('metadata',{});group=meta.get('scenario_group_id');family=meta.get('rule_family_id')
            if not group or not family:raise DataContractError('explicit family holdout requires world and rule_family_id')
            target=family_targets.get(family)
            if target:
                if group in forced_groups and forced_groups[group]!=target:
                    raise DataContractError('one world spans conflicting family holdouts')
                forced_groups[group]=target
    for r in records:
        group=r.get('metadata',{}).get('scenario_group_id')
        if not group:raise DataContractError('Missing scenario_group_id; cannot split safely')
        if group not in group_split:
            digest=sha256(f'{seed}|{group}'.encode()).digest()
            n=int.from_bytes(digest[:8],'big')/2**64
            group_split[group]=forced_groups.get(group,('train' if n<train else 'validation' if n<train+val else 'test'))
        splits[group_split[group]].append(r)
    return splits
