"""Pin official policy source metadata; do not redistribute complete policy text."""
import base64,hashlib,json,urllib.request,urllib.parse,datetime
from pathlib import Path
R=Path(__file__).resolve().parent
REV='e1c706abbe174add18ac09a72976df2a79d0178c'
FILES=['content/handbook/support/workflows/data_handling.md','content/handbook/people-group/acceptable-use-policy.md','content/handbook/support/workflows/ticket_lifecycle.md','LICENSE']

def run():
    output=R/'SOURCE_PIN.json';assert not output.exists();records=[]
    for path in FILES:
        url='https://gitlab.com/api/v4/projects/gitlab-com%2Fcontent-sites%2Fhandbook/repository/files/'+urllib.parse.quote(path,safe='')+'?ref='+REV
        with urllib.request.urlopen(url,timeout=10) as response:data=json.load(response)
        raw=base64.b64decode(data['content']);digest=hashlib.sha256(raw).hexdigest()
        assert data['commit_id']==REV and digest==data['content_sha256']
        lines=raw.decode().splitlines();headings=[{'line':i+1,'heading':line} for i,line in enumerate(lines) if line.startswith('#')]
        records.append({'path':path,'revision':REV,'last_commit_id':data['last_commit_id'],'sha256':digest,'bytes':len(raw),'permalink':'https://gitlab.com/gitlab-com/content-sites/handbook/-/blob/'+REV+'/'+path,'headings':headings})
    result={'status':'SOURCE_PINNED_NOT_LABEL_VALIDATED','retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repository':'https://gitlab.com/gitlab-com/content-sites/handbook','revision':REV,'files':records,'full_text_redistributed':False,'license_scope':'Repository LICENSE recorded separately; not a determination of rights for all handbook or linked material.','independent_normative_review':False,'cross_task_validity_established':False}
    output.write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'files':len(records),'revision':REV}))

if __name__=='__main__':run()
