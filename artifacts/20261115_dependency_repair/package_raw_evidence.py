"""Lossless deterministic compression for publishing bulky rollout transcripts."""
from pathlib import Path
from hashlib import sha256
import gzip,json

R=Path(__file__).resolve().parent;OUT=R/'published_raw';OUT.mkdir(exist_ok=True)
records=[]
for folder in ('calibration','calibration_final','calibration_bound_v3'):
    target=OUT/folder;target.mkdir(exist_ok=True)
    for path in sorted((R/folder).glob('*.json')):
        data=path.read_bytes();packed=gzip.compress(data,compresslevel=6,mtime=0)
        output=target/(path.name+'.gz');output.write_bytes(packed)
        assert gzip.decompress(packed)==data
        records.append({'source':str(path.relative_to(R)).replace('\\','/'),
            'compressed':str(output.relative_to(R)).replace('\\','/'),
            'raw_sha256':sha256(data).hexdigest(),'compressed_sha256':sha256(packed).hexdigest(),
            'raw_bytes':len(data),'compressed_bytes':len(packed)})
out={'files':records,'roundtrip_all_passed':True,'raw_data_modified':False}
(OUT/'MANIFEST.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'files':len(records),'raw_bytes':sum(x['raw_bytes'] for x in records),
                  'compressed_bytes':sum(x['compressed_bytes'] for x in records),'roundtrip_all_passed':True}))
