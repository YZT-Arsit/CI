import sys,json,time
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from verified_source_cache import build_env,verify_sources
from kci.receipt_tasks import ReceiptTask
from kci.ordering_receipt_tasks import OrderingReceiptTask
from experiments.receipt_dataset import SOURCE
from kci.dynamic_tasks import canonical
index=int(sys.argv[1]);worlds=[json.loads(x) for x in (R/'dataset/source_holdout_v3/worlds_0000.jsonl').read_text().splitlines()]
if index<8:chosen=worlds[index*8:index*8+2]
else:chosen=[json.loads(next((R/'dataset/test').glob('worlds_*.jsonl')).read_text().splitlines()[index-8])]
t=time.perf_counter()
for w in chosen:
    source=ROOT/w['source_root_relative'] if 'source_root_relative' in w else SOURCE
    cls=OrderingReceiptTask if w.get('kernel_variant')=='ordering' else ReceiptTask
    fresh=cls(w,source)
    for _ in range(2):
        cached=build_env(w,source)
        assert canonical(fresh.binding)==canonical(cached.binding)
        assert canonical(fresh.exact(fresh.initial()))==canonical(cached.exact(cached.initial()))
out=R/'cache_audits';out.mkdir(exist_ok=True)
report={'index':index,'worlds':len(chosen),'seconds':time.perf_counter()-t,'all_terminal_outcomes_byte_equal':True,**verify_sources()}
(out/f'{index:02d}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
