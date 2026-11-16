"""Fixed full-factorial worlds; each invocation generates four pairs."""
import sys,json,itertools,copy
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiments.receipt_dataset import make_world,derive
from kci.dynamic_tasks import canonical

def swapped(world):
    w=copy.deepcopy(world)
    endpoints=[p['links'][-1][1] for p in w['providers']]
    for row in w['histories']:
        if row[0] in endpoints:row[0]=endpoints[1-endpoints.index(row[0])]
    return w

def run(start):
    grid=list(itertools.product(('meeting','query'),(1,2),(0,2),(False,True),(False,True),(False,True),('one','none'),range(4)))
    out=R/'dataset/train_balanced';out.mkdir(exist_ok=True);worlds=[];pairs=[]
    for index in range(start,min(start+4,len(grid))):
        kind,depth,noise,required,allow,delivered,mode,replicate=grid[index]
        w=make_world(120000+index,kind,depth=depth,noise=noise,required=required,allow=allow,delivered=delivered,mode=mode)
        w['factorial_group']=index;w['mode']=mode
        for side,world in enumerate((w,swapped(w))):
            world=copy.deepcopy(world);world['world_id']+=f'_pair{side}'
            world['paired_side']=side;worlds.append(world);pairs.extend(derive(world)[0])
    for name,rows in [('worlds',worlds),('preferences',pairs)]:
        path=out/f'{name}_{start:04d}.jsonl'
        if path.exists():raise FileExistsError(path)
        path.write_text(''.join(canonical(x)+'\n' for x in rows),encoding='utf-8')
    print(json.dumps({'start':start,'worlds':len(worlds),'preferences':len(pairs)}))
if __name__=='__main__':run(int(sys.argv[1]))
