"""Describe saved errors only; never change the frozen gates or run inference."""
import collections
import json
import re
from graph_data import R, read, rows, sha


def main():
    result = read(R / 'RESULTS.json'); labels = rows(R / 'labels.jsonl'); obs = rows(R / 'observations.jsonl')
    analyses = []
    for summary in result['summaries']:
        arm = summary['arm']; folder = R / 'study' / arm
        binary = rows(folder / 'binary.jsonl'); proof = rows(folder / 'proof.jsonl')
        confusion = collections.Counter(); witness_errors = collections.Counter(); agreement = collections.Counter()
        groups = collections.defaultdict(collections.Counter)
        for o, label, b, p in zip(obs, labels, binary, proof):
            assert o['id'] == label['id'] == b['id'] == p['id']
            matches = re.findall(r'ANSWER:\s*([01])\b', b['raw'])
            bit = int(matches[0]) if len(matches) == 1 and b['seconds_batch'] < 9.5 else None
            match = re.fullmatch(r'\s*PROOF:\s*([A-Za-z]+)\s*', p['raw'])
            word = match.group(1) if match and p['seconds_batch'] < 9.5 else None
            bc = bit == label['target']; pc = word in label['valid_witnesses']
            confusion[f"truth{label['target']}_pred{bit}"] += 1
            agreement[f'binary_{bc}_valid_proof_{pc}'] += 1
            if pc: category = 'valid_witness'
            elif word is None: category = 'invalid_format_or_time'
            elif word == 'none': category = 'false_no_witness'
            else:
                # Search only the data serialization, not explanatory instruction words.
                fields = o['proof_prompt'].split('Known nodes: ', 1)[1].split('. Give one known node', 1)[0]
                category = 'present_name_but_not_witness' if re.search(r'\b' + re.escape(word) + r'\b', fields) else 'not_a_present_name'
            witness_errors[category] += 1
            for key in [f"family={o['family']}", f"noise={o['noise']}", f"vocab={o['vocab']}",
                        f"valid_witness_count={0 if label['valid_witnesses'] == ['none'] else len(label['valid_witnesses'])}"]:
                groups[key].update(queries=1, binary_correct=int(bc), valid_proof=int(pc))
        assert sum(v for k, v in agreement.items() if k.startswith('binary_True_')) == summary['binary']['correct']
        assert witness_errors['valid_witness'] == summary['proof']['correct']
        analyses.append({'arm': arm, 'binary_confusion': dict(confusion), 'witness_categories': dict(witness_errors),
                         'separate_request_agreement': dict(agreement), 'strata': dict(groups)})
    output = {'analyses': analyses, 'scope': 'Descriptive saved-output analysis; no gate changes, causal faithfulness claim or new model calls.',
              'results_sha256': sha(R / 'RESULTS.json'), 'seal_sha256': sha(R / 'SEALED.json'),
              'script_sha256': sha(R / 'diagnostics.py'), 'new_model_calls': 0}
    (R / 'DIAGNOSTICS.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'models': len(analyses), 'new_model_calls': 0, 'errors': 0}))


if __name__ == '__main__': main()
