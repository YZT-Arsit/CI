"""Paired diagnostic scoring; this module is not an action selector."""


def pair_metrics(records):
    grouped={}
    for row in records:
        if row['selected'] is not None and row['selected'] not in row['actions']:
            raise ValueError('selected action outside declared set')
        if row['optimal_action'] not in row['actions']:raise ValueError('invalid target')
        grouped.setdefault(row['pair_id'],[]).append(row)
    result=dict(groups=len(grouped),both_valid=0,both_correct=0,same_action=0,
                reversed_both_wrong=0,invalid_groups=0)
    for rows in grouped.values():
        if len(rows)!=2 or {r['side'] for r in rows}!={0,1}:raise ValueError('incomplete or duplicated pair')
        a,b=rows
        if set(a['actions'])!=set(b['actions']):raise ValueError('different action sets')
        valid=all(r['selected'] is not None for r in rows)
        correct=[r['selected']==r['optimal_action'] for r in rows]
        same=valid and a['selected']==b['selected']
        result['both_valid']+=valid;result['invalid_groups']+=not valid
        result['both_correct']+=valid and all(correct);result['same_action']+=same
        result['reversed_both_wrong']+=valid and not same and not any(correct)
    return result
