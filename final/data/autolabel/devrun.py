"""Run one shell command on the LingBot dev machine through the GateShell bastion (own PTY session) and print its output.
  devrun.py 'ls /mnt/forge/datasets/private'
  devrun.py --fetch REMOTE_DIR --out LOCAL_DIR [--include meta data videos/observation.images.camera_top]
"""
import argparse, base64, hashlib, io, os, pty, re, select, signal, sys, tarfile, time, zlib
from pathlib import Path
HOST = '<bastion-user>@<bastion-host>'
ANSI = re.compile(rb'\x1b\[[0-9;?]*[A-Za-z]')

class T:
    def __init__(s, fd): s.fd, s.buf = fd, b''
    def read(s, to=1):
        if select.select([s.fd], [], [], to)[0]:
            p = os.read(s.fd, 1 << 17)
            if not p: raise RuntimeError('closed')
            s.buf += p
    def expect(s, x, to=60):
        dl = time.monotonic() + to
        while x not in s.buf:
            if time.monotonic() > dl: raise RuntimeError('timeout waiting for %r: %r' % (x, s.buf[-400:]))
            s.read()
        i = s.buf.index(x) + len(x); pre, s.buf = s.buf[:i], s.buf[i:]; return pre
    def send(s, d):
        for i in range(0, len(d), 512):
            m = memoryview(d[i:i + 512])
            while m: n = os.write(s.fd, m); m = m[n:]
            s.read(0.02)
    def line(s, to=600):
        dl = time.monotonic() + to
        while b'\n' not in s.buf:
            if time.monotonic() > dl: raise RuntimeError('line timeout')
            s.read()
        l, s.buf = s.buf.split(b'\n', 1); return ANSI.sub(b'', l).rstrip(b'\r')

def connect():
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp('ssh', ['ssh', '-tt', '-i', str(Path.home() / '.ssh/id_ed25519'), '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
                          '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30', '-p', '60022', HOST])
    t = T(fd)
    t.expect(b'0/1, name-asc'); t.send(b'jjj'); t.expect(b'1/1, name-asc'); t.send(b'\r'); t.expect(b':~$ ')
    t.send(b'stty -echo\n'); t.expect(b':~$ ')
    return pid, fd, t

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', nargs='?'); ap.add_argument('--fetch'); ap.add_argument('--out', type=Path)
    ap.add_argument('--include', nargs='*', default=['meta', 'data', 'videos/observation.images.camera_top'])
    a = ap.parse_args()
    pid, fd, t = connect()
    try:
        if a.cmd:
            t.send(f"echo __B__; {{ {a.cmd}; }} 2>&1; echo __E__\n".encode())
            t.expect(b'__B__'); print(ANSI.sub(b'', t.expect(b'__E__')[:-5]).decode(errors='replace').strip())
        if a.fetch:
            code = ("import base64,hashlib,io,os,sys,tarfile\nroot=%r\ninc=%r\nb=io.BytesIO()\n"
                    "with tarfile.open(fileobj=b,mode='w:gz',compresslevel=1) as tf:\n"
                    " for x in inc:\n  p=os.path.join(root,x)\n  if os.path.exists(p): tf.add(p,arcname=x)\n"
                    "d=b.getvalue();print('RF_BEGIN',len(d),hashlib.sha256(d).hexdigest(),flush=True)\n"
                    "e=base64.b64encode(d).decode()\nfor i in range(0,len(e),4000): sys.stdout.write(e[i:i+4000]+'\\n')\n"
                    "print('RF_END',flush=True)\n") % (a.fetch, a.include)
            enc = base64.b64encode(zlib.compress(code.encode())).decode()
            t.send(("python3 - <<'RFX'\nimport base64,zlib\nexec(zlib.decompress(base64.b64decode(%r)))\nRFX\n" % enc).encode())
            while True:
                l = t.line()
                if l.startswith(b'RF_BEGIN'): _, n, h = l.decode().split(); n = int(n); break
                if b'Traceback' in l or b'Error' in l: raise RuntimeError(l.decode())
            buf, dg, t0 = io.BytesIO(), hashlib.sha256(), time.time()
            while True:
                l = t.line(120)
                if l == b'RF_END': break
                p = base64.b64decode(l, validate=True); buf.write(p); dg.update(p)
                if buf.tell() // (20 << 20) != (buf.tell() - len(p)) // (20 << 20):
                    print(f'{buf.tell() >> 20} / {n >> 20} MB  {buf.tell() / (time.time() - t0) / 1e6:.1f} MB/s', flush=True)
            if buf.tell() != n or dg.hexdigest() != h: raise RuntimeError('size/sha mismatch')
            buf.seek(0); a.out.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=buf, mode='r:gz') as tf:
                for m in tf.getmembers():
                    if Path(m.name).is_absolute() or '..' in Path(m.name).parts: raise RuntimeError('bad member ' + m.name)
                tf.extractall(a.out)
            print('FETCH_OK', n, h, a.out)
        t.send(b'stty echo\nexit\n')
    finally:
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
        os.close(fd); os.waitpid(pid, 0)

if __name__ == '__main__':
    main()
