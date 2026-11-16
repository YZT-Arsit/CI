"""Fixed-ID source holdout, one <=7s HTTP request per worker; no selection by result."""
import json,sys,time
from pathlib import Path
from urllib.request import Request,urlopen
from hashlib import sha256

R=Path(__file__).resolve().parent
REV='ac6634013dc74b1c9601d9e59248a59465438a43'
scenario=int(sys.argv[1]);kind=sys.argv[2]
assert 2<=scenario<=9 and kind in ('scenario','utility')
remote=f'meeting_allocation/scenario_{scenario:02d}/{kind}.json'
url=f'https://huggingface.co/datasets/ServiceNow/PiSAs/resolve/{REV}/{remote}'
folder=R/'heldout_sources'/f'scenario_{scenario:02d}';folder.mkdir(parents=True,exist_ok=True)
path=folder/(kind+'.json')
if path.exists():raise FileExistsError('source file already exists')
t=time.perf_counter()
with urlopen(Request(url,headers={'User-Agent':'KCI-semantic-audit/1.0'}),timeout=7) as response:
    raw=response.read(500001)
if len(raw)>500000:raise ValueError('file limit')
json.loads(raw);path.write_bytes(raw)
receipt={'url':url,'revision':REV,'original_path':remote,'bytes':len(raw),
         'sha256':sha256(raw).hexdigest(),'seconds':time.perf_counter()-t,'selection':'fixed scenario IDs 02 through 09 before download'}
(folder/(kind+'_receipt.json')).write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
