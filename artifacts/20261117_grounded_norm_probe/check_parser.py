"""Reject ambiguous, negated, incomplete and explanatory text before model calls."""
import json
from pathlib import Path
from audit import parse

positives={'PASS':'PASS','FAIL':'FAIL',' UNKNOWN\n':'UNKNOWN'}
normalized={'answer: pass.':'PASS','FAIL!':'FAIL','Answer: UNKNOWN':'UNKNOWN'}
negatives=['not PASS','PASS or FAIL','UNKNOWN because facts are missing','FAIL\nPASS','', 'PASSAGE','ANSWER: PASS but FAIL']
for text, expected in positives.items():
    assert parse(text)=={'strict':expected,'secondary':expected}
for text, expected in normalized.items():
    assert parse(text)=={'strict':None,'secondary':expected}
for text in negatives:
    assert parse(text)=={'strict':None,'secondary':None}
assert parse('PASS',False)=={'strict':None,'secondary':None}
result={'cases':14,'all_passed':True,'model_calls':0}
Path(__file__).with_name('PARSER_CHECK.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
