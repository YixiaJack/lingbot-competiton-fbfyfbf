"""Training progress for the board (https://lingbot-board.vercel.app/train).

Keeps one SSH session to the dev machine open (the user's exact command, then the GateShell menu: Down to row 001,
Enter), every INTERVAL seconds pipes train_probe.py into `python3 -` there, checks the returned zlib+base64 JSON
(size + SHA-256), and publishes it as train_status.json on branch `train` of the private repo
YixiaJack/lingbot-board-data (one amended commit, force-pushed; this branch belongs to this script only, the field
page uses branch field and the cluster job uses main). Reconnects when the session drops.

  python3 train_board.py [--interval 120] [--once] [--no-push]
  bash train_board.sh start|stop|status|log
"""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import pty
import re
import select
import signal
import subprocess
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = Path.home() / '.cache' / 'train_board'
REPO_DIR = STATE / 'repo'
REPO_URL = 'https://github.com/YixiaJack/lingbot-board-data.git'
BRANCH = 'train'
# the user's own login command, unchanged (11/14: "完全按照我这个做法连")
SSH = ['ssh', '-tt', '-i', str(Path.home() / '.ssh' / 'id_ed25519'), '-o', 'IdentitiesOnly=yes', '-p', '60022',
       '<bastion-user>@<bastion-host>']
ANSI = re.compile(rb'\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07')


def log(*a):
    print(dt.datetime.now().strftime('%m-%d %H:%M:%S'), *a, flush=True)


class Session:
    def __init__(self):
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.environ['TERM'] = 'xterm'
            os.execvp(SSH[0], SSH)
        import fcntl, struct, termios
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack('HHHH', 50, 220, 0, 0))
        self.buf = b''

    def read(self, timeout=1.0):
        if select.select([self.fd], [], [], timeout)[0]:
            try:
                part = os.read(self.fd, 131072)
            except OSError as e:
                raise RuntimeError('ssh closed') from e
            if not part:
                raise RuntimeError('ssh closed')
            self.buf += part

    def wait_for(self, pats, timeout):
        end = time.monotonic() + timeout
        while True:
            for p in pats:
                if p in self.buf:
                    return p
            if time.monotonic() > end:
                raise RuntimeError(f'timeout waiting for {pats!r}: {ANSI.sub(b"", self.buf[-400:])!r}')
            self.read(0.5)

    def send(self, data):
        for i in range(0, len(data), 512):
            chunk = memoryview(data[i:i + 512])
            while chunk:
                n = os.write(self.fd, chunk)
                chunk = chunk[n:]
            self.read(0.02)

    def login(self):
        self.wait_for([b'name-asc', b'name-desc'], 90)
        time.sleep(1.5)
        for _ in range(6):   # the cursor starts on the [Search] input; Down moves it to row 001 (status bar 1/1)
            self.buf = b''
            self.send(b'\x1b[B')
            try:
                self.wait_for([b'1/1,'], 3)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError('menu: cursor never reached row 001')
        time.sleep(0.5)
        self.buf = b''
        self.send(b'\r')
        self.wait_for([b':~$ '], 60)
        self.buf = b''
        self.send(b"stty -echo; export PS1='TB> '; unset PROMPT_COMMAND\n")
        self.wait_for([b'TB> '], 20)
        log('logged in to the dev machine')

    def probe(self, source, timeout=240):
        enc = base64.b64encode(zlib.compress(source.encode(), 9)).decode()
        boot = "cd ~ && python3 -I - <<'TB_PROBE_CODE'\nimport base64,zlib\nexec(zlib.decompress(base64.b64decode(\n"
        boot += '\n'.join(repr(enc[i:i + 1000]) for i in range(0, len(enc), 1000))
        boot += "\n)))\nTB_PROBE_CODE\n"
        self.buf = b''
        self.send(boot.encode())
        self.wait_for([b'TRAIN_BEGIN '], timeout)
        self.wait_for([b'TRAIN_END'], 120)
        text = ANSI.sub(b'', self.buf).decode('ascii', 'replace').replace('\r', '')
        head = text[text.index('TRAIN_BEGIN '):]
        first, rest = head.split('\n', 1)
        _, size, sha = first.split()
        body = ''.join(l.strip() for l in rest[:rest.index('TRAIN_END')].split('\n'))
        z = base64.b64decode(body, validate=True)
        if len(z) != int(size) or hashlib.sha256(z).hexdigest() != sha:
            raise RuntimeError('probe output failed size/SHA-256 check')
        self.wait_for([b'TB> '], 30)
        return json.loads(zlib.decompress(z))

    def close(self):
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            os.close(self.fd)
            os.waitpid(self.pid, 0)
        except OSError:
            pass


def git(*a, check=True):
    return subprocess.run(['git', '-C', str(REPO_DIR), *a], check=check, capture_output=True, text=True)


def publish(doc):
    if not (REPO_DIR / '.git').exists():
        REPO_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', '-q', str(REPO_DIR)], check=True)
        git('remote', 'add', 'origin', REPO_URL)
        git('checkout', '-q', '--orphan', BRANCH)
    (REPO_DIR / 'train_status.json').write_text(json.dumps(doc, ensure_ascii=False, separators=(',', ':')))
    git('add', 'train_status.json')
    amend = ['--amend'] if git('rev-parse', '--verify', '-q', 'HEAD', check=False).returncode == 0 else []
    git('-c', 'user.name=YixiaJack', '-c', 'user.email=145562489+YixiaJack@users.noreply.github.com',
        'commit', '-q', *amend, '--allow-empty', '-m', f"train status {doc.get('generated')}", '--date=now')
    git('push', '-q', '--force', 'origin', f'HEAD:{BRANCH}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', type=int, default=120)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--no-push', action='store_true')
    a = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    source = (HERE / 'train_probe.py').read_text()
    sess, fails = None, 0
    while True:
        t0 = time.monotonic()
        try:
            if sess is None:
                sess = Session()
                sess.login()
            doc = sess.probe(source)
            doc['collector'] = {'published': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                                'interval_s': a.interval}
            (STATE / 'last.json').write_text(json.dumps(doc, ensure_ascii=False))
            cur = next((r for r in doc['runs'] if r['name'] == doc.get('current')), {})
            if not a.no_push:
                publish(doc)
            log(f"published {doc.get('current')} step {cur.get('step')}/{cur.get('max_steps')} "
                f"alive={cur.get('alive')} log_age={cur.get('log_age_s')}s")
            fails = 0
        except Exception as e:   # keep running: reconnect next round
            fails += 1
            log(f'round failed ({fails}): {e}')
            if sess is not None:
                sess.close()
                sess = None
            if a.once:
                sys.exit(1)
        if a.once:
            break
        time.sleep(max(5, (a.interval if fails == 0 else min(30 * fails, a.interval)) - (time.monotonic() - t0)))
    if sess is not None:
        sess.close()


if __name__ == '__main__':
    main()
