"""The existing semantic-set actor loss with fixed targets and no query term."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'20261117_teacher_binding'))
from binding_loss import torch,forward_targets,answer_ids


def correction_loss(tokenizer,model,batch):
    logits,ce,tokens=forward_targets(tokenizer,model,[r['prompt'] for r in batch],[r['answer'] for r in batch])
    letters=answer_ids(tokenizer)
    semantic=torch.stack([torch.logsumexp(logits[i,-2],0)-torch.logsumexp(logits[i,-2,letters[r['good_indices']]],0)
                          for i,r in enumerate(batch)]).mean()
    formatting=.1*torch.cat([ce[:,:-2],ce[:,-1:]],1).mean()
    return semantic+formatting,{'semantic':float(semantic.detach()),'formatting':float(formatting.detach())},tokens
