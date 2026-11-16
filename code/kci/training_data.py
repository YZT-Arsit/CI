"""CPU sequence adapter for pinned tokenizer assets and verified preference rows.

No model, Trainer, semantic judge, or data split is created here. Full prompts,
completion EOS, and template tails are retained; over-budget pairs are rejected.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
from copy import deepcopy
from hashlib import sha256
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Mapping,Sequence
from zipfile import ZipFile


IGNORE_INDEX=-100
OBSERVATION_KEYS={'role','task','purpose','turn','remaining_steps','local_facts',
                  'authorized_delivered_messages','visible_ci_norms','pending_request',
                  'awaiting_notification','instruction'}
ACTION_KEYS={'actor','kind','recipient','fact_ids','body','answer'}
RECEIPT_KEYS={'event_id','sender','recipient','purpose','fact_ids','body','delivered'}
GROUP_FIELDS=('scenario_group_id','base_group_id','base_case_group_id','workflow_group_id',
              'source_world_group_id','rule_family_id','split','split_status','fold_id')
FORBIDDEN_FIELDS={'metadata','reference','reference_witness','witness','supporting_reports',
                  'oracle_answer','final_correct','state_id','evidence','reference_cost','reference_status'}


class TrainingDataError(ValueError):pass


class TokenBudgetExceeded(TrainingDataError):
    def __init__(self,required_tokens:int,max_length:int,prompt_tokens:int):
        super().__init__(f'complete pair requires {required_tokens} tokens, budget is {max_length}; no truncation performed')
        self.required_tokens=required_tokens;self.max_length=max_length;self.prompt_tokens=prompt_tokens


def canonical(value)->str:
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def text_hash(value:str)->str:return sha256(value.encode('utf-8')).hexdigest()


def _unique_json_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise TrainingDataError('duplicate JSON key in model input: '+key)
        result[key]=value
    return result


def audit_record(record:Mapping,*,structural_control:bool=False)->dict:
    for field in ('prompt','chosen','rejected'):
        if not isinstance(record.get(field),str) or not record[field]:raise TrainingDataError('missing text field: '+field)
    try:
        obs=json.loads(record['prompt'],object_pairs_hook=_unique_json_object)
        actions={s:json.loads(record[s],object_pairs_hook=_unique_json_object) for s in ('chosen','rejected')}
    except (ValueError,TypeError) as error:raise TrainingDataError('structured observation/actions require valid JSON') from error
    if not isinstance(obs,dict) or set(obs)!=OBSERVATION_KEYS:
        raise TrainingDataError('prompt must contain exactly authorized Observation fields')
    if any(not isinstance(obs[f],str) or not obs[f] for f in ('role','task','purpose','instruction')):
        raise TrainingDataError('observation text fields must be nonempty strings')
    if any(type(obs[f]) is not int or obs[f]<0 for f in ('turn','remaining_steps')):
        raise TrainingDataError('turn and remaining steps must be nonnegative integers')
    if any(type(obs[f]) is not bool for f in ('pending_request','awaiting_notification')):
        raise TrainingDataError('observation flags must be booleans')
    if not isinstance(obs['visible_ci_norms'],list) or any(not isinstance(n,str) for n in obs['visible_ci_norms']):
        raise TrainingDataError('visible norms must be a list of text strings')
    if not isinstance(obs['local_facts'],list) or any(not isinstance(f,dict) or set(f)!={'id','text'} for f in obs['local_facts']):
        raise TrainingDataError('local facts expose unsupported fields')
    if any(not isinstance(f[k],str) or not f[k] for f in obs['local_facts'] for k in ('id','text')):
        raise TrainingDataError('local fact IDs and text must be nonempty strings')
    if not isinstance(obs['authorized_delivered_messages'],list):raise TrainingDataError('receipt log must be a list')
    event_ids=set()
    for receipt in obs['authorized_delivered_messages']:
        if not isinstance(receipt,dict) or set(receipt)!=RECEIPT_KEYS or receipt['delivered'] is not True:
            raise TrainingDataError('authorized receipt must be a complete delivered message')
        if any(not isinstance(receipt[f],str) or not receipt[f] for f in ('event_id','sender','recipient','purpose','body')):
            raise TrainingDataError('receipt identifiers, scope and body must be nonempty strings')
        if not isinstance(receipt['fact_ids'],list) or any(not isinstance(f,str) or not f for f in receipt['fact_ids']):
            raise TrainingDataError('receipt fact IDs must be a string list')
        if receipt['event_id'] in event_ids:raise TrainingDataError('duplicate receipt event ID')
        event_ids.add(receipt['event_id'])
    def check_keys(value):
        if isinstance(value,dict):
            if FORBIDDEN_FIELDS.intersection(value):raise TrainingDataError('hidden evidence/metadata field in model input')
            for item in value.values():check_keys(item)
        elif isinstance(value,list):
            for item in value:check_keys(item)
    check_keys(obs)
    for side,action in actions.items():
        if not isinstance(action,dict) or set(action)!=ACTION_KEYS:raise TrainingDataError('completion must be a complete Action')
        if any(not isinstance(action[f],str) for f in ACTION_KEYS-{'fact_ids'}) or not action['actor'] or not action['kind']:
            raise TrainingDataError('action fields must have declared string types')
        if not isinstance(action['fact_ids'],list) or any(not isinstance(f,str) or not f for f in action['fact_ids']):
            raise TrainingDataError('action fact IDs must be a string list')
        if action['actor']!=obs['role']:raise TrainingDataError('completion actor does not match visible role')
        check_keys(action)
    if canonical(actions['chosen'])==canonical(actions['rejected']):raise TrainingDataError('identical chosen/rejected actions')
    metadata=record.get('metadata')
    if not isinstance(metadata,Mapping):raise TrainingDataError('preference provenance metadata missing')
    if structural_control:
        if metadata.get('control_only') is not True or metadata.get('certified') is not False or metadata.get('evidence_class')!='synthetic_sequence_control':
            raise TrainingDataError('structural controls must be explicitly uncertified and excluded from training')
    elif metadata.get('certified') is not True:raise TrainingDataError('uncertified preference')
    for field in ('scenario_group_id','rule_family_id'):
        if not isinstance(metadata.get(field),str) or not metadata[field]:raise TrainingDataError('missing group: '+field)
    split=metadata.get('split','unassigned')
    if split not in {'train','validation','test','unassigned'}:raise TrainingDataError('unsupported split value')
    return {'observation':obs,'actions':actions,'prompt_identity':text_hash(canonical(obs)),
            'action_identities':{s:text_hash(canonical(a)) for s,a in actions.items()},
            'split':split,'schema_leakage_check':'passed','semantic_leakage_check':'not_established',
            'training_eligible':not structural_control}


class PreferenceCorpusIndex:
    """Corpus-wide checks must run before splitting work into tokenization chunks."""
    def __init__(self,*,enforce_family_split:bool=False):
        self.enforce_family_split=enforce_family_split
        self.directions={};self.base_splits={};self.family_splits={};self.workflow_splits={}
        self.rows=0;self.duplicates=0

    def add(self,record:Mapping)->None:
        audit=audit_record(record);metadata=record['metadata'];ids=audit['action_identities']
        key=(audit['prompt_identity'],*sorted((ids['chosen'],ids['rejected'])))
        orientation=(ids['chosen'],ids['rejected'])
        if key in self.directions:
            if self.directions[key]!=orientation:raise TrainingDataError('same-prompt reverse preference conflict')
        duplicate=key in self.directions
        base=metadata.get('base_case_group_id',metadata.get('base_group_id',metadata['scenario_group_id']))
        family=metadata['rule_family_id'];split=audit['split']
        if not isinstance(base,str) or not base:raise TrainingDataError('empty base group')
        for mapping,group,enforce in ((self.base_splits,base,True),(self.family_splits,family,self.enforce_family_split)):
            previous=mapping.get(group,set())
            assigned=(previous|{split})-{'unassigned'}
            if enforce and len(assigned)>1:raise TrainingDataError('group appears in incompatible dataset splits: '+group)
        self.directions[key]=orientation;self.rows+=1;self.duplicates+=int(duplicate)
        self.base_splits.setdefault(base,set()).add(split);self.family_splits.setdefault(family,set()).add(split)
        workflow=metadata.get('workflow_group_id')
        if workflow:self.workflow_splits.setdefault(workflow,set()).add(split)

    def summary(self)->dict:
        return {'rows':self.rows,'same_direction_duplicates':self.duplicates,
                'base_groups':len(self.base_splits),'rule_families':len(self.family_splits),
                'family_split_enforced':self.enforce_family_split,
                'shared_workflows_across_splits':[g for g,s in self.workflow_splits.items() if len(s-{'unassigned'})>1],
                'independent_task_generalization_established':False}


@dataclass(frozen=True)
class PinnedTokenizerAssets:
    manifest_path:Path
    tokenizer_dir:Path
    dependency_dir:Path
    wheels_dir:Path
    expected_manifest_sha256:str

    @classmethod
    def from_manifest(cls,manifest_path,*,tokenizer_dir=None,dependency_dir=None,wheels_dir=None,expected_manifest_sha256=None):
        path=Path(manifest_path).resolve();root=path.parent
        return cls(path,Path(tokenizer_dir or root/'official_tokenizer').resolve(),
                   Path(dependency_dir or root/'dependencies').resolve(),Path(wheels_dir or root/'wheels').resolve(),
                   expected_manifest_sha256 or sha256(path.read_bytes()).hexdigest())

    def verify(self)->dict:
        raw=self.manifest_path.read_bytes()
        if sha256(raw).hexdigest()!=self.expected_manifest_sha256:raise TrainingDataError('pinned asset manifest SHA256 mismatch')
        manifest=json.loads(raw.decode('utf-8'))
        if manifest.get('official_repo')!='Qwen/Qwen2.5-7B-Instruct' or len(manifest.get('official_revision',''))!=40:
            raise TrainingDataError('expected pinned official Qwen tokenizer revision')
        hashes={}
        for name in ('tokenizer.json','tokenizer_config.json'):
            actual=sha256((self.tokenizer_dir/name).read_bytes()).hexdigest()
            if actual!=manifest['tokenizer_files'].get(name):raise TrainingDataError('tokenizer asset SHA256 mismatch: '+name)
            hashes[name]=actual
        package_files=0
        for name,expected in manifest['wheels'].items():
            wheel=self.wheels_dir/name
            if sha256(wheel.read_bytes()).hexdigest()!=expected['sha256']:raise TrainingDataError('dependency wheel SHA256 mismatch')
            with ZipFile(wheel) as archive:
                for entry in archive.namelist():
                    if entry.endswith('/') or entry.split('/')[0] not in {'tokenizers','jinja2','markupsafe'}:continue
                    installed=(self.dependency_dir/entry).resolve()
                    if not installed.is_relative_to(self.dependency_dir):raise TrainingDataError('dependency path escaped isolation')
                    if not installed.is_file() or sha256(installed.read_bytes()).digest()!=sha256(archive.read(entry)).digest():
                        raise TrainingDataError('installed isolated dependency differs from pinned wheel: '+entry)
                    package_files+=1
        versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(self.dependency_dir)])}
        if versions!=manifest['dependencies']:raise TrainingDataError('isolated dependency versions differ from pinned manifest')
        return {'repo':manifest['official_repo'],'revision':manifest['official_revision'],'tokenizer_files':hashes,
                'dependency_versions':versions,'verified_installed_package_files':package_files,
                'manifest_sha256':sha256(self.manifest_path.read_bytes()).hexdigest()}


@dataclass(frozen=True)
class EncodedSequence:
    input_ids:tuple[int,...]
    labels:tuple[int,...]
    attention_mask:tuple[int,...]
    position_ids:tuple[int,...]
    prompt_length:int
    completion_end:int


@dataclass(frozen=True)
class EncodedPreference:
    prompt_ids:tuple[int,...]
    chosen:EncodedSequence
    rejected:EncodedSequence
    provenance:dict

    @property
    def required_tokens(self)->int:return max(len(self.chosen.input_ids),len(self.rejected.input_ids))

    def as_dict(self)->dict:return asdict(self)


class TrainingSequenceAdapter:
    def __init__(self,assets:PinnedTokenizerAssets):
        self.asset_provenance=assets.verify()
        old_path=list(sys.path);sys.path.insert(0,str(assets.dependency_dir))
        try:
            tokenizers=importlib.import_module('tokenizers');jinja=importlib.import_module('jinja2.sandbox')
            for name in ('tokenizers','jinja2','markupsafe'):
                module=importlib.import_module(name)
                if not Path(module.__file__).resolve().is_relative_to(assets.dependency_dir):
                    raise TrainingDataError('loaded dependency is outside the pinned isolation directory: '+name)
            self.tokenizer=tokenizers.Tokenizer.from_file(str(assets.tokenizer_dir/'tokenizer.json'))
            self.config=json.loads((assets.tokenizer_dir/'tokenizer_config.json').read_text(encoding='utf-8'))
            self.template=jinja.ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True).from_string(self.config['chat_template'])
        finally:sys.path[:]=old_path
        if self.tokenizer.truncation is not None or self.tokenizer.padding is not None:
            raise TrainingDataError('tokenizer has implicit truncation or padding enabled')
        self.eos_token_id=self.tokenizer.token_to_id(self.config['eos_token'])
        self.pad_token_id=self.tokenizer.token_to_id(self.config['pad_token'])
        if self.eos_token_id is None or self.pad_token_id is None:raise TrainingDataError('EOS/pad token unavailable')
        self.special_strings={x['content'] for x in self.config.get('added_tokens_decoder',{}).values() if x.get('special')}
        self.special_strings.update((self.config['eos_token'],self.config['pad_token']))

    def render(self,prompt:str,completion:str|None=None)->str:
        messages=[{'role':'user','content':prompt}]
        if completion is not None:messages.append({'role':'assistant','content':completion})
        return self.template.render(messages=messages,tools=None,add_generation_prompt=completion is None)

    def encode_pair(self,record:Mapping,*,max_length:int|None=None,expected_prompt_sha256:str|None=None,
                    structural_control:bool=False)->EncodedPreference:
        audit=audit_record(record,structural_control=structural_control);prompt=record['prompt'];prompt_sha=text_hash(prompt)
        if expected_prompt_sha256 is not None and prompt_sha!=expected_prompt_sha256:
            raise TrainingDataError('prompt differs from the externally supplied authorized prompt digest')
        for field in ('prompt','chosen','rejected'):
            if any(token in record[field] for token in self.special_strings):
                raise TrainingDataError('literal tokenizer control token in '+field)
        prompt_ids=tuple(self.tokenizer.encode(self.render(prompt),add_special_tokens=False).ids)
        sequences={}
        for side in ('chosen','rejected'):
            ids=tuple(self.tokenizer.encode(self.render(prompt,record[side]),add_special_tokens=False).ids)
            if ids[:len(prompt_ids)]!=prompt_ids:raise TrainingDataError('tokenized completion does not share the complete prompt prefix')
            stops=[i for i in range(len(prompt_ids),len(ids)) if ids[i]==self.eos_token_id]
            if len(stops)!=1 or stops[0]<=len(prompt_ids):raise TrainingDataError('completion requires content and exactly one terminal EOS')
            end=stops[0]+1
            labels=(IGNORE_INDEX,)*len(prompt_ids)+ids[len(prompt_ids):end]+(IGNORE_INDEX,)*(len(ids)-end)
            sequences[side]=EncodedSequence(ids,labels,(1,)*len(ids),tuple(range(len(ids))),len(prompt_ids),end)
        metadata=record['metadata']
        provenance={'record_sha256':text_hash(canonical(record)),'record_id':metadata.get('record_id'),
                    'prompt_sha256':prompt_sha,'canonical_prompt_sha256':audit['prompt_identity'],
                    'chosen_sha256':text_hash(record['chosen']),'rejected_sha256':text_hash(record['rejected']),
                    'assets':deepcopy(self.asset_provenance),'source_evidence_class':metadata.get('evidence_class'),
                    'source_human_verified':metadata.get('human_verified'),
                    'groups':{field:deepcopy(metadata[field]) for field in GROUP_FIELDS if field in metadata},
                    'schema_leakage_check':audit['schema_leakage_check'],'semantic_leakage_check':audit['semantic_leakage_check'],
                    'training_eligible':audit['training_eligible'],'control_only':structural_control,
                    'model_input_fields':['prompt','chosen','rejected'],'truncated':False}
        provenance['groups'].setdefault('split','unassigned')
        pair=EncodedPreference(prompt_ids,sequences['chosen'],sequences['rejected'],provenance)
        if max_length is not None:
            if not isinstance(max_length,int) or max_length<1:raise TrainingDataError('max_length must be a positive integer')
            if pair.required_tokens>max_length:raise TokenBudgetExceeded(pair.required_tokens,max_length,len(prompt_ids))
        return pair

    def collate(self,pairs:Sequence[EncodedPreference],*,padding_side:str='right',pad_to_length:int|None=None,
                pad_token_id:int|None=None,allow_structural_controls:bool=False)->dict:
        if not pairs:raise TrainingDataError('empty batch')
        if not allow_structural_controls and any(not p.provenance.get('training_eligible') for p in pairs):
            raise TrainingDataError('structural-only sequences cannot enter a training batch')
        if padding_side not in {'left','right'}:raise TrainingDataError('padding side must be left or right')
        width=max(pair.required_tokens for pair in pairs)
        if pad_to_length is not None:
            if not isinstance(pad_to_length,int) or pad_to_length<width:raise TrainingDataError('padding width would truncate a complete pair')
            width=pad_to_length
        pad=self.pad_token_id if pad_token_id is None else pad_token_id
        if not isinstance(pad,int) or pad<0:raise TrainingDataError('invalid padding token ID')
        result={side:{field:[] for field in ('input_ids','labels','attention_mask','position_ids')} for side in ('chosen','rejected')}
        for pair in pairs:
            if pair.provenance.get('assets')!=self.asset_provenance:
                raise TrainingDataError('batch combines unverified or different tokenizer assets')
            for side in ('chosen','rejected'):
                sequence=getattr(pair,side);length=len(sequence.input_ids);padding=width-length
                p=len(pair.prompt_ids);end=sequence.completion_end
                if (sequence.prompt_length!=p or sequence.input_ids[:p]!=pair.prompt_ids or not p<end<=length
                        or sequence.input_ids[end-1]!=self.eos_token_id
                        or sequence.labels!=(IGNORE_INDEX,)*p+sequence.input_ids[p:end]+(IGNORE_INDEX,)*(length-end)
                        or sequence.labels.count(self.eos_token_id)!=1
                        or sequence.attention_mask!=(1,)*length or sequence.position_ids!=tuple(range(length))):
                    raise TrainingDataError('encoded sequence lost prompt, EOS, labels, attention or positions')
                for field,fill in (('input_ids',pad),('labels',IGNORE_INDEX),('attention_mask',0),('position_ids',0)):
                    values=list(getattr(sequence,field));empty=[fill]*padding
                    result[side][field].append(empty+values if padding_side=='left' else values+empty)
        result['provenance']=[deepcopy(p.provenance) for p in pairs]
        result['padding_side']=padding_side
        result['causal_alignment']='logits[:, :-1] score labels[:, 1:]; prompt/padding/template tail are masked'
        return result
