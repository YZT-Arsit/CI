"""Pinned official Qwen tokenizer and explicit chat rendering; no truncation."""
import hashlib
import json
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / '20261112_cpu_pilot/tokenization'
CONFIG = json.loads((ASSETS / 'config.json').read_text())
for name, digest in CONFIG['tokenizer_files'].items():
    assert hashlib.sha256((ASSETS / 'official_tokenizer' / name).read_bytes()).hexdigest() == digest
sys.path.insert(0, str(ASSETS / 'dependencies'))
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment

TOKENIZER = Tokenizer.from_file(str(ASSETS / 'official_tokenizer/tokenizer.json'))
TEMPLATE = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
    json.loads((ASSETS / 'official_tokenizer/tokenizer_config.json').read_text())['chat_template'])
SYSTEM = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'


def render(prompt):
    text = TEMPLATE.render(messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
                           tools=None, add_generation_prompt=True)
    return text, len(TOKENIZER.encode(text, add_special_tokens=False).ids)
