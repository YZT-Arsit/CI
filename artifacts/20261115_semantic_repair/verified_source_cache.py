"""Reuse verified immutable source inputs, never policy decisions or outcomes."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from kci.receipt_tasks import ReceiptTask
from kci.ordering_receipt_tasks import OrderingReceiptTask

_templates={};_hashes={}

def build_env(world,source):
    cls=OrderingReceiptTask if world.get('kernel_variant')=='ordering' else ReceiptTask
    key=(cls,world['kind'],str(source))
    if key not in _templates:
        _templates[key]=cls(world,source)
        for relative,expected in _templates[key].kernel.source_hashes.items():
            p=Path(source)/relative
            _hashes[str(p)]=sha256(p.read_bytes()).hexdigest()
    env=deepcopy(_templates[key]);env.world=deepcopy(world);env.binding['world']=deepcopy(world)
    return env

def verify_sources():
    assert all(sha256(Path(p).read_bytes()).hexdigest()==h for p,h in _hashes.items())
    return {'files':len(_hashes),'source_bytes_unchanged':True,'policy_outputs_cached':False}
