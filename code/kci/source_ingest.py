"""Read-only, pinned source anchors. These records are never dynamic preferences."""
from hashlib import sha256
import json
from pathlib import Path
import re
from .datasets import audit_pisas_files,KCI_REQUIRED_NEW


class SourceIngestError(ValueError):pass


def _read(path:Path,expected_sha256:str):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha256):raise SourceIngestError('expected SHA256 required')
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=expected_sha256:raise SourceIngestError('source bytes differ from pinned manifest')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise SourceIngestError('duplicate source JSON key')
            result[key]=value
        return result
    return json.loads(raw.decode('utf-8'),object_pairs_hook=unique)


def _base(source,revision,workflow,hashes,license_status):
    if not re.fullmatch('[0-9a-f]{40}',revision):raise SourceIngestError('full pinned revision required')
    return {'schema_version':'static_source_anchor_v1','source':source,'source_revision':revision,
            'source_sha256':dict(hashes),'workflow_group_id':workflow,'split':'unassigned',
            'license_status':license_status,'human_verified':False,'certified':False,
            'delivered_events':[],'derived_ci_labels':[],'dynamic_train_ready':False,
            'model_input_ready':False,'observation_filter_implemented':False,
            'missing_dynamic_contract':sorted(KCI_REQUIRED_NEW),
            'evidence_class':'external_static_source_anchor',
            'boundary':'Static access permissions are not delivered knowledge; source annotations are not new derived-CI labels.'}


def load_pisas_anchor(directory,*,revision,workflow_group_id,expected_sha256,license_status):
    """Keep source timeline as initial artifacts, and oracle material in annotations only."""
    directory=Path(directory)
    names=('scenario.json','utility.json','appropriateness.json','visibility.json')
    if set(expected_sha256)!=set(names):raise SourceIngestError('four-file hash manifest required')
    files={name:_read(directory/name,expected_sha256[name]) for name in names}
    audit_pisas_files(files)
    ids={file['scenario_id'] for file in files.values()}
    if len(ids)!=1:raise SourceIngestError('cross-file scenario IDs disagree')
    scenario,utility,appropriate,visibility=(files[n] for n in names)
    values=utility['attribute_values'];labels=appropriate['attributes'];access=visibility['attributes']
    if not all(isinstance(x,dict) for x in (values,labels,access)):raise SourceIngestError('attribute maps required')
    if set(values)!=set(labels) or set(values)!=set(access):raise SourceIngestError('atomic annotation keys disagree')
    if set(labels.values())-{'appropriate','inappropriate'}:raise SourceIngestError('unknown atomic label')
    artifacts=scenario['timeline']
    if not isinstance(artifacts,list) or len({a['id'] for a in artifacts})!=len(artifacts):
        raise SourceIngestError('unique initial artifact IDs required')
    by_id={a['id']:a for a in artifacts};mismatches=[];atomic=[]
    for attribute,value in values.items():
        v=access[attribute]
        if not isinstance(v,dict) or not {'holder','visible_to','hidden_from','value'}<=set(v):
            raise SourceIngestError('incomplete atomic visibility annotation')
        if set(v['visible_to'])&set(v['hidden_from']):raise SourceIngestError('contradictory atomic visibility')
        if value['source'] not in by_id:raise SourceIngestError('attribute references missing initial artifact')
        artifact=by_id[value['source']]
        extra=sorted(set(artifact['visible_to'])-set(v['visible_to']))
        if extra:mismatches.append({'attribute_id':attribute,'artifact_id':artifact['id'],
                                   'artifact_readers_outside_atomic_visibility':extra})
        atomic.append({'attribute_id':attribute,'source_artifact_id':value['source'],
                       'source_value':value['value'],'source_appropriateness':labels[attribute],
                       'source_visibility':v,'label_scope':'source_static_atomic_attribute',
                       'value_matches_visibility':value['value']==v['value']})
    anchor=_base('ServiceNow/PiSAs',revision,workflow_group_id,expected_sha256,license_status)
    anchor.update(source_scenario_id=next(iter(ids)),
        initial_materials={k:scenario[k] for k in ('task','cast','org')},
        source_task_reference=utility,atomic_labels=atomic,
        source_structural_audit={'attribute_count':len(atomic),'initial_artifact_count':len(artifacts),
            'artifact_atomic_visibility_mismatches':mismatches,
            'atomic_values_match_visibility':all(a['value_matches_visibility'] for a in atomic)})
    anchor['initial_materials']['initial_artifacts']=artifacts
    return anchor


def load_multiagentprivacy_anchor(path,*,revision,expected_sha256,license_status):
    """Preserve distributed tables and source goals without claiming an executable oracle."""
    path=Path(path);raw=_read(path,expected_sha256)
    if not {'entities','data_distribution','run_1_benign','run_2_sensitive'}<=set(raw):
        raise SourceIngestError('unsupported MultiAgentPrivacy source schema')
    if set(raw['entities'])!=set(raw['data_distribution']):raise SourceIngestError('agent distribution mismatch')
    tables=[]
    for actor,entry in raw['data_distribution'].items():
        for name,table in entry['table'].items():
            if table['sensitivity'] not in {'sensitive','non-sensitive'} or not isinstance(table['rows'],list):
                raise SourceIngestError('unsupported table schema')
            tables.append({'actor':actor,'table':name,'source_sensitivity':table['sensitivity'],
                           'rows':table['rows'],'label_scope':'source_table_sensitivity_not_CI_authorization'})
    anchor=_base('Vaidehi99/MultiAgentPrivacy',revision,'multiagentprivacy/'+path.stem,
                 {path.name:expected_sha256},license_status)
    anchor.update(source_scenario_id=path.stem,
                  initial_materials={'entities':raw['entities'],'distributed_tables':tables},
                  source_task_reference={k:v for k,v in raw.items() if k not in {'entities','data_distribution'}},
                  atomic_labels=[],source_structural_audit={'agents':len(raw['entities']),
                      'tables':len(tables),'rows':sum(len(t['rows']) for t in tables)},
                  exact_task_oracle_ready=False,
                  task_oracle_limitation='Prose goals/results require independent completeness and threshold validation.')
    return anchor
