"""Local live viewer of what the VLA is being told on the 5080 (harness prompts, session state, LLM view, yolocue).
Runs on the Mac: keeps `ssh lbreal python3 -u - < follow_remote.py` alive and pushes its records to the browser (SSE).
  python3 server.py            -> http://127.0.0.1:8766   (PORT env to change)
  BIND=0.0.0.0 python3 server.py -> also reachable on the LAN; non-local visitors need the access code once:
                                 http://<mac-ip>:8766/?k=<code>  (code in access_code, kept in a cookie afterwards)
Read-only on the 5080; no third-party packages.
/topref: live top camera (cam_remote.py over ssh, started only while the page polls /live.jpg) with ghosts of past
scenes built by mk_topref.py (topref/)."""
import base64, collections, http.server, json, os, queue, re, secrets, socketserver, subprocess, threading, time, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get('PORT', '8766'))
HOST = os.environ.get('HOST', 'lbreal')
BIND = os.environ.get('BIND', '127.0.0.1')
CODE_FILE = os.path.join(HERE, 'access_code')
if not os.path.exists(CODE_FILE):
    with open(CODE_FILE, 'w') as f:
        f.write(secrets.token_hex(3) + '\n')
    os.chmod(CODE_FILE, 0o600)
CODE = open(CODE_FILE).read().strip()
YIMG = re.compile(r'\d{8}_\d{6}_[0-9a-f]{8}_(cue|nocue|nocue_raw|grasp|regrasp\d)\.jpg')
YCACHE = os.path.join(HERE, 'yolo_cache')
TOPREF = os.path.join(HERE, 'topref')
TREF = re.compile(r'((ref|obj|edge|line)_\d{8}_\d{6}_[0-9a-f]{8}\.(jpg|png))|refs\.json')
CAM_IDLE_S = 20                                   # stop the camera reader this long after the last /live.jpg request
cam = {'jpg': None, 't': 0.0, 'stamp': 0.0, 'req': 0.0, 'proc': None, 'hb': 0.0, 'seen': 0, 'err': '', 'started': 0.0}
backlog = collections.deque(maxlen=6000)
clients, lock = set(), threading.Lock()
link = {'ok': False, 'since': time.time(), 'err': ''}


def push(msg, keep=True):
    data = json.dumps(msg, ensure_ascii=False)
    with lock:
        if keep:
            backlog.append(data)
        for q in list(clients):
            try:
                q.put_nowait(data)
            except queue.Full:
                pass


def reader():
    while True:
        try:
            p = subprocess.Popen(['ssh', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
                                  '-o', 'BatchMode=yes', HOST, 'python3 -u -'],
                                 stdin=open(os.path.join(HERE, 'follow_remote.py')), stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, bufsize=1)
        except OSError as e:
            link.update(ok=False, err=str(e)); time.sleep(5); continue
        with lock:
            backlog.clear()
        push({'src': 'reset'}, keep=False)
        got = False
        for line in p.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if not got:
                got = True; link.update(ok=True, since=time.time(), err='')
            push(msg, keep=msg.get('src') != 'hb')
        err = (p.stderr.read() or '').strip()[-300:]
        link.update(ok=False, since=time.time(), err=err or 'ssh ended')
        push({'src': 'link', 'rec': dict(link)}, keep=False)
        time.sleep(3)


def cam_reader(p):
    for line in p.stdout:
        q = line.rstrip('\n').split('\t')
        if q[0] == 'f' and len(q) == 3:
            try:
                cam.update(jpg=base64.b64decode(q[2]), t=time.time(), stamp=float(q[1]), err='')
            except ValueError:
                pass
        elif q[0] == 'hb' and len(q) == 3:
            cam.update(hb=time.time(), seen=int(q[2]))
    err = (p.stderr.read() or '').strip()[-200:]; p.wait()
    if p.returncode != -15:                       # -15 = stopped by cam_manager when nobody watches: not an error
        cam['err'] = err or 'camera reader ended'


def cam_manager():
    """keeps cam_remote.py running on the 5080 only while /live.jpg is being requested"""
    while True:
        p = cam['proc']; active = time.time() - cam['req'] < CAM_IDLE_S
        if p is not None and p.poll() is not None:
            cam['proc'] = p = None
        if active and p is None and time.time() - cam['started'] > 5:
            try:
                p = subprocess.Popen(['ssh', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
                                      '-o', 'BatchMode=yes', HOST,
                                      'source ~/workspace/lingbot/inference/env.sh >/dev/null 2>&1; exec nice -n 19 python3 -u -'],
                                     stdin=open(os.path.join(HERE, 'cam_remote.py')), stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
                cam.update(proc=p, started=time.time(), err='')
                threading.Thread(target=cam_reader, args=(p,), daemon=True).start()
            except OSError as e:
                cam['err'] = str(e)
        elif not active and p is not None:
            p.terminate(); cam.update(proc=None, jpg=None)
        time.sleep(0.5)


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def allowed(self):
        if self.client_address[0] in ('127.0.0.1', '::1'):
            return True
        ck = dict(p.strip().split('=', 1) for p in (self.headers.get('Cookie') or '').split(';') if '=' in p)
        return secrets.compare_digest(ck.get('hv', ''), CODE)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); qk = urllib.parse.parse_qs(u.query).get('k', [''])[0]
        if qk:
            if secrets.compare_digest(qk, CODE):
                self.send_response(303); self.send_header('Set-Cookie', f'hv={CODE}; Max-Age=2592000; Path=/; SameSite=Strict; HttpOnly')
                self.send_header('Location', u.path if u.path in ('/', '/yolo', '/topref') else '/'); self.end_headers()
            else:
                self.send_error(403, 'wrong code')
            return
        if not self.allowed():
            body = '<meta charset="utf-8"><p style="font:16px sans-serif;padding:24px">需要访问码：请用队友发给你的完整链接打开（形如 /?k=xxxxxx）。</p>'.encode()
            self.send_response(403); self.send_header('Content-Type', 'text/html; charset=utf-8'); self.end_headers(); self.wfile.write(body); return
        self.path = u.path
        if self.path in ('/', '/index.html'):
            body = open(os.path.join(HERE, 'index.html'), 'rb').read()
            self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body); return
        if self.path in ('/yolo', '/yolo.html'):          # task-1/2 YOLO view (same access rules, same event stream)
            body = open(os.path.join(HERE, 'yolo.html'), 'rb').read()
            self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body); return
        if self.path in ('/topref', '/topref.html'):      # live top camera + ghosts of a past scene
            body = open(os.path.join(HERE, 'topref.html'), 'rb').read()
            self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith('/topref/'):
            name = self.path[8:]
            f = os.path.join(TOPREF, name)
            if not TREF.fullmatch(name) or not os.path.isfile(f):
                self.send_error(404); return
            body = open(f, 'rb').read()
            ct = 'application/json' if name.endswith('.json') else 'image/png' if name.endswith('.png') else 'image/jpeg'
            self.send_response(200); self.send_header('Content-Type', ct)
            self.send_header('Cache-Control', 'no-cache'); self.end_headers(); self.wfile.write(body); return   # references get rebuilt in place
        if self.path == '/live.jpg':
            cam['req'] = time.time(); age = time.time() - cam['t']
            if cam['jpg'] is not None and age < 5:
                body = cam['jpg']
                self.send_response(200); self.send_header('Content-Type', 'image/jpeg'); self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Frame-Age', f'{age:.2f}'); self.end_headers(); self.wfile.write(body); return
            st = 'starting' if cam['proc'] is not None and time.time() - cam['started'] < 15 else \
                 'no frames' if cam['proc'] is not None else 'connecting'
            body = json.dumps({'state': st, 'err': cam['err'], 'seen': cam['seen']}, ensure_ascii=False).encode()
            self.send_response(503); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-store')
            self.end_headers(); self.wfile.write(body); return
        if self.path.startswith('/yimg/'):                # yolocue images, fetched read-only from the 5080 once, then cached
            name = self.path[6:]
            if not YIMG.fullmatch(name):
                self.send_error(404); return
            f = os.path.join(YCACHE, name)
            if not os.path.exists(f):
                try:
                    r = subprocess.run(['ssh', '-o', 'ConnectTimeout=8', '-o', 'BatchMode=yes', HOST,
                                        f'cat ~/workspace/lingbot/inference/logs/yolocue/{name}'], capture_output=True, timeout=20)
                except subprocess.TimeoutExpired:
                    self.send_error(504); return
                if r.returncode != 0 or not r.stdout.startswith(b'\xff\xd8'):
                    self.send_error(404); return
                os.makedirs(YCACHE, exist_ok=True)
                with open(f + '.part', 'wb') as fh:
                    fh.write(r.stdout)
                os.replace(f + '.part', f)
            body = open(f, 'rb').read()
            self.send_response(200); self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Cache-Control', 'max-age=86400'); self.end_headers(); self.wfile.write(body); return
        if self.path != '/events':
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream'); self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        q = queue.Queue(maxsize=20000)
        with lock:
            snap = list(backlog); clients.add(q)
        try:
            self.wfile.write(b'data: {"src":"reset"}\n\n')
            self.wfile.write(('data: ' + json.dumps({'src': 'link', 'rec': dict(link)}) + '\n\n').encode())
            for d in snap:
                self.wfile.write(('data: ' + d + '\n\n').encode())
            self.wfile.flush()
            while True:
                try:
                    d = q.get(timeout=15)
                    self.wfile.write(('data: ' + d + '\n\n').encode())
                except queue.Empty:
                    self.wfile.write(b': keepalive\n\n')
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with lock:
                clients.discard(q)


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == '__main__':
    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=cam_manager, daemon=True).start()
    print(f'http://127.0.0.1:{PORT}  bind {BIND}' + (f'  LAN: http://<mac-ip>:{PORT}/?k={CODE}' if BIND != '127.0.0.1' else ''), flush=True)
    S((BIND, PORT), H).serve_forever()
