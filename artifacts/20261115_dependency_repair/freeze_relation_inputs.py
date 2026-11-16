"""Persist the fixed synthetic graph tensors used by all diagnostic arms."""
import json
from hashlib import sha256
from pathlib import Path
import numpy as np
from relation_graph_probe import graphs

R=Path(__file__).resolve().parent;OUT=R/'graph_data';OUT.mkdir(exist_ok=True)
cases=[('train',51001,2,False),('fresh_binding',62001,2,False),('larger',62002,5,False),('branching',62003,5,True)]
records=[]
for name,seed,n,branching in cases:
    values,meta=graphs(seed,n,branching)
    path=OUT/(name+'.npz');np.savez_compressed(path,x=values[0],adjacency=values[1],root=values[2],label=values[3])
    (OUT/(name+'_metadata.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8')
    records.append({'name':name,'seed':seed,'rows':len(meta),'graph_nodes':values[0].shape[1],
        'file_sha256':sha256(path.read_bytes()).hexdigest(),
        'tensor_sha256':sha256(b''.join(x.tobytes() for x in values)).hexdigest()})
out={'datasets':records,'scope':'synthetic graph tensors, many isomorphic duplicates; not independent tasks or raw LLM prompts',
     'test_protocol':'Three evaluation conditions were fixed before fitting; four permission/match strata in each.',
     'entity_identity':'exact graph binding is supplied by representation; model does not learn string equality',
     'gpu_training_eligible':False}
(OUT/'DATA_CARD.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out))
