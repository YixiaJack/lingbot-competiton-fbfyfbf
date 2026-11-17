"""Copy files FROM the dev machine through the GateShell bastion (user's exact ssh command; base64 over the PTY,
size + SHA-256 checked).   python3 devfetch.py REMOTE_PATH LOCAL_PATH [REMOTE_PATH LOCAL_PATH ...]"""
import base64, hashlib, os, re, sys, time
sys.path.insert(0, os.path.expanduser('~/LingBotRoboTwin/train_board'))
from train_board import Session, ANSI

REMOTE = r'''import base64,hashlib,os,sys
p=sys.argv[1];b=open(p,'rb').read()
sys.stdout.write('FETCH_BEGIN %d %s\n'%(len(b),hashlib.sha256(b).hexdigest()))
e=base64.b64encode(b).decode()
for i in range(0,len(e),4000): sys.stdout.write(e[i:i+4000]+'\n')
sys.stdout.write('FETCH_END\n');sys.stdout.flush()'''


def fetch(s, remote, local):
    code = base64.b64encode(REMOTE.encode()).decode()
    s.buf = b''
    s.send(f"python3 -c \"import base64;exec(base64.b64decode('{code}'))\" '{remote}'\n".encode())
    s.wait_for([b'FETCH_BEGIN '], 120)
    data = bytearray(s.buf); s.buf = b''
    t0 = time.time()
    while b'FETCH_END' not in data[-200:]:
        r = select_read(s)
        data += r
    end = bytes(data).rindex(b'FETCH_END'); s.buf = bytes(data[end + 9:])
    text = ANSI.sub(b'', bytes(data[:end + 9])).replace(b'\r', b'')
    head = text[text.index(b'FETCH_BEGIN '):]
    first, rest = head.split(b'\n', 1)
    _, size, sha = first.decode().split()
    body = b''.join(l.strip() for l in rest[:rest.index(b'FETCH_END')].split(b'\n'))
    raw = base64.b64decode(body, validate=True)
    if len(raw) != int(size) or hashlib.sha256(raw).hexdigest() != sha:
        raise RuntimeError(f'{remote}: size/SHA mismatch ({len(raw)} vs {size})')
    open(local, 'wb').write(raw)
    dt = time.time() - t0
    print(f'{remote} -> {local}: {len(raw)} bytes, sha256 ok, {dt:.1f}s ({len(raw)/max(dt,1e-3)/1e6:.2f} MB/s)', flush=True)
    s.wait_for([b'TB> '], 60)


def select_read(s):
    import select
    if select.select([s.fd], [], [], 30)[0]:
        part = os.read(s.fd, 1 << 20)
        if not part:
            raise RuntimeError('ssh closed')
        return part
    raise RuntimeError('transfer stalled for 30 s')


if __name__ == '__main__':
    a = sys.argv[1:]
    s = Session(); s.login()
    try:
        for i in range(0, len(a), 2):
            fetch(s, a[i], a[i + 1])
    finally:
        s.close()
