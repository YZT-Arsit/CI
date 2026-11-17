"""Separate presentation interventions; preserve the existing prompt when seeds match."""
import random
from kci.relation_language import RULES,vocabulary

def render(obs,palette,seeds,question='policy',candidate=None):
    names=vocabulary(obs,palette,seeds['names']);fmt=lambda e:names[e[0]]+' '+names[e[1]];facts=[]
    for message in obs['delivered']:
        if message['recipient']!='R':continue
        for key,tag in [('identities','I'),('links','L'),('histories','H')]:facts += [tag+' '+fmt(e) for e in message['content'].get(key,[])]
    random.Random(seeds['facts']+11).shuffle(facts);plans=[]
    for provider in obs['contract']['providers']:
        links=list(provider['links']);random.Random(seeds['links']+17).shuffle(links);plans.append(names[provider['ticket']]+': '+ '; '.join('L '+fmt(e) for e in links))
    random.Random(seeds['plans']+23).shuffle(plans);actions=list(obs['actions']);random.Random(seeds['options']+31).shuffle(actions)
    def action(a):return 'Commit '+names[a.split(':',1)[1]] if a.startswith('COMMIT:') else {'DELIVER_INPUTS':'Deliver','PUBLISH_FULL':'Full','PUBLISH_MINIMAL':'Minimal','WITHHOLD':'Stop'}[a]
    text=(RULES+'\nRole '+obs['role']+'. Person '+names[obs['subject']]+'. Chosen '+names.get(obs['committed_ticket'],'none')+'. '
          'Permitted '+str(int(obs['contract']['permission']))+'. Full-required '+str(int(obs['contract']['binding_required']))+'.\n'
          'R received: '+('; '.join(facts) or 'nothing')+'.\nPlans: '+' | '.join(plans)+'.\n')
    if question=='policy':text+='Options: '+'; '.join(chr(65+i)+' '+action(a) for i,a in enumerate(actions))+'. Choose the best option.'
    else:
        if candidate not in obs['actions']:raise ValueError('candidate outside authorized actions')
        q='Is this action immediately compliant?' if question=='safe' else 'After this action, is compliant task completion still possible?'
        text+='Action '+action(candidate)+'. '+q+' A yes; B no.'
    return text,actions

def query(obs,palette,seeds,action):
    return render(obs,palette,seeds,'task',action)[0].replace('A yes; B no.','Reply ANSWER: followed by 1 if possible, otherwise 0.')

def actor(obs,palette,seeds,estimates):
    text,actions=render(obs,palette,seeds);text+='Own estimates (1 means compliant completion remains possible): '+ '; '.join(chr(65+i)+'='+estimates[a] for i,a in enumerate(actions))+'.\n'
    return text+'\nReply ANSWER:\nletter.',actions
