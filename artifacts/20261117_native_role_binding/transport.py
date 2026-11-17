"""Lossless policy pages and role-local schema access for the native interface."""
import hashlib
import json
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / '20261117_native_roles'))
from session import RoleSession


def pages(text, cap=1800):
    result, offset = [], 0
    while offset < len(text):
        end = min(offset + cap, len(text))
        if end < len(text):
            boundary = text.rfind('\n', offset + cap // 2, end)
            if boundary > offset:
                end = boundary + 1
        result.append(text[offset:end])
        offset = end
    assert ''.join(result) == text
    return result


class Transport:
    def __init__(self, task):
        self.session = RoleSession(task)
        self.policy_pages = pages(self.session.view('assistant')['instructions'])
        self.index = []
        heading = 'General policy'
        for i, text in enumerate(self.policy_pages):
            headings = re.findall(r'^#+\s+(.+)$', text, re.M)
            self.index.append({'page': i, 'section': heading, 'headings_in_page': headings,
                               'opening': text[:80]})
            if headings:
                heading = headings[-1]
        self.references = []

    def observation(self, role):
        view = self.session.view(role)
        instructions = view['instructions'] if role == 'user' else (
            'You are the telecom customer-service assistant. The complete native policy is available through read_policy. '
            'Consult relevant policy pages before acting. No policy text has been discarded.')
        return {'role': role, 'instructions': instructions,
                'policy_pages': self.index if role == 'assistant' else [],
                'own_tool_names': [tool['function']['name'] for tool in view['tools']],
                'history': view['history']}

    def prompt(self, role):
        protocol = ('Act only as the role in the supplied observation. Continue its task using its own instructions, tools and history. '
                    'Do not invent tool results. Return one JSON object only. Allowed commands:\n'
                    '{"kind":"message","text":"message to the other role"}; '
                    '{"kind":"schema","name":"one of your own tool names"}; '
                    '{"kind":"policy","page":0} (assistant only); '
                    '{"kind":"tool","name":"own tool name","arguments":{}}; '
                    '{"kind":"finish","text":"final message"}. '
                    'Read a tool schema before calling an unfamiliar tool. A message switches to the other role; a tool/reference call keeps your turn.\n')
        return protocol + json.dumps(self.observation(role), ensure_ascii=False)

    def execute(self, role, command):
        before = self.session.view('user' if role == 'assistant' else 'assistant')
        kind = command.get('kind')
        if kind in ['message', 'finish']:
            text = command.get('text')
            if not isinstance(text, str) or not text.strip():
                result = {'error': 'nonempty text required'}
                self.session.local[role].append({'type': 'protocol_error', 'request': command, **result})
                assert before == self.session.view('user' if role == 'assistant' else 'assistant')
                return result, role, False
            self.session.message(role, text)
            return {'delivered': True}, 'user' if role == 'assistant' else 'assistant', kind == 'finish'
        if kind == 'tool':
            name, args = command.get('name'), command.get('arguments')
            own = {tool['function']['name'] for tool in self.session.view(role)['tools']}
            if name not in own or not isinstance(args, dict):
                result = {'error': 'own tool name and object arguments required'}
                self.session.local[role].append({'type': 'protocol_error', **result})
            else:
                result = self.session.tool(role, name, args)
        elif kind in ['schema', 'policy']:
            if kind == 'schema':
                schemas = {tool['function']['name']: tool for tool in self.session.view(role)['tools']}
                result = schemas.get(command.get('name'), {'error': 'tool unavailable for this role'})
            else:
                page = command.get('page')
                result = {'page': page, 'text': self.policy_pages[page]} if role == 'assistant' and type(page) is int and 0 <= page < len(self.policy_pages) else {'error': 'invalid policy page or role'}
            self.session.local[role].append({'type': 'reference', 'kind': kind, 'request': command, 'result': result})
            self.references.append({'role': role, 'kind': kind, 'request': command, 'result_sha256': hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()})
        else:
            result = {'error': 'unknown command kind'}
            self.session.local[role].append({'type': 'protocol_error', **result})
        assert before == self.session.view('user' if role == 'assistant' else 'assistant')
        return result, role, False
