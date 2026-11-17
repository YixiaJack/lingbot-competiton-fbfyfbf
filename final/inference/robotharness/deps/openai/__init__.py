"""Minimal stand-in for the `openai` client, standard library only, for the 5080 inference env (no `openai` installed).

Covers exactly what harness.Chat uses: OpenAI(api_key, base_url, timeout, max_retries).chat.completions.create(model,
messages, response_format, max_tokens, extra_body) -> obj.choices[0].message.content and obj.usage.<field>. It is on
sys.path only as an appended fallback (harness.Chat appends ../deps), so a real `openai` package always wins.
RH_TIMEOUT (seconds, default 12) caps the request timeout so a slow API cannot stall the robot for long.
"""
import json
import os
import urllib.error
import urllib.request


class _Obj(dict):
    __getattr__ = dict.get


class _Completions:
    def __init__(self, client):
        self._c = client

    def create(self, model, messages, response_format=None, max_tokens=None, extra_body=None, **kw):
        body = {'model': model, 'messages': messages}
        if response_format:
            body['response_format'] = response_format
        if max_tokens:
            body['max_tokens'] = max_tokens
        if extra_body:
            body.update(extra_body)
        body.update({k: v for k, v in kw.items() if v is not None})
        req = urllib.request.Request(self._c.base_url.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self._c.api_key},
                                     method='POST')
        last = None
        for _ in range(self._c.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self._c.timeout) as r:
                    data = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                last = RuntimeError(f'HTTP {e.code}: {e.read()[:300]!r}')
                if e.code < 500 and e.code != 429:
                    raise last
            except Exception as e:  # timeout, connection reset: retry once
                last = e
        else:
            raise last
        choices = [_Obj(message=_Obj(content=(ch.get('message') or {}).get('content'))) for ch in data.get('choices', [])]
        return _Obj(choices=choices, usage=_Obj(**(data.get('usage') or {})))


class OpenAI:
    def __init__(self, api_key, base_url='https://api.deepseek.com', timeout=30.0, max_retries=1, **_):
        self.api_key, self.base_url, self.max_retries = api_key, base_url, int(max_retries)
        self.timeout = min(float(timeout), float(os.environ.get('RH_TIMEOUT', '12')))
        self.chat = _Obj(completions=_Completions(self))
