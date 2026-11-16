"""Independent path argument versus executed zero-cost continuation labels."""
import json,sys,sqlite3
from pathlib import Path
from hashlib import sha256

R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from kci.receipt_tasks import ReceiptTask
from kci.dynamic_tasks import canonical
from experiments.receipt_dataset import SOURCE


def independent_success(world,provider):
    reached={provider['ticket']};remaining=list(provider['links'])
    while remaining:
        used=[(a,b) for a,b in remaining if a in reached]
        if not used:break
        reached.update(b for _,b in used)
        remaining=[(a,b) for a,b in remaining if (a,b) not in used]
    match=world['history_delivered'] and any(k in reached for k,_ in world['histories'])
    return not (world['binding_required'] and not world['allow'] and match)


def independent_task_truth(env):
    if env.world['kind']=='meeting':return env.kernel.utility['oracle_schedule']
    db=sqlite3.connect(':memory:')
    db.executescript('CREATE TABLE names(id INTEGER,name TEXT); CREATE TABLE counts(id INTEGER,n INTEGER); CREATE TABLE cities(id INTEGER,city TEXT);')
    db.executemany('INSERT INTO names VALUES (?,?)',[(r['company_id'],r['company_name']) for r in env.kernel.tables['company_id_map']])
    db.executemany('INSERT INTO counts VALUES (?,?)',[(r['company_id'],r['gas_stations_count']) for r in env.kernel.tables['company_gas_stations']])
    db.executemany('INSERT INTO cities VALUES (?,?)',[(r['company_id'],r['city']) for r in env.kernel.tables['company_locations']])
    result=[{'company':name,'stations':n,'city':city} for name,n,city in db.execute('SELECT names.name,counts.n,cities.city FROM names JOIN counts USING(id) JOIN cities USING(id) ORDER BY names.rowid')]
    db.close();return result


def audit(input_path,output):
    worlds=[json.loads(x) for x in Path(input_path).read_text().splitlines()];rows=[]
    for world in worlds:
        env=ReceiptTask(world,SOURCE);assert env.expected==independent_task_truth(env)
        s=env.initial();proof=env.preference(s,*env.actions(s));assert proof['immediate']==['VALID','VALID']
        expected=[(1,int(independent_success(world,p)),0) for p in world['providers']]
        assert [tuple(v) for v in proof['values']]==expected
        if proof['chosen'] is not None:
            # Even arbitrarily unfavorable finite cost cannot outrank success.
            winner=proof['chosen'];loser=1-winner
            assert (*expected[winner][:2],-10**12)>(*expected[loser][:2],0)
        rows.append({'world_id':world['world_id'],'values':expected,'chosen':proof['chosen'],
                     'independent_source_truth':True,'path_argument_matches_execution':True})
    Path(output).write_text(json.dumps({'input_sha256':sha256(Path(input_path).read_bytes()).hexdigest(),
        'worlds':rows,'all_passed':True,'scope':'explicit synthetic obligation and recipient-access assumptions'},indent=2))
    print(json.dumps({'worlds':len(rows),'all_passed':True}))


if __name__=='__main__':audit(sys.argv[1],sys.argv[2])
