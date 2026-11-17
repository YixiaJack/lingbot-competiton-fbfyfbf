// 现场作战台的共享记录：自测打分和提交前检查清单，存在私有 Vercel Blob（store lingbot-field）的 field/state.json。
// GET 读全部；POST 发一个操作（{op, ...}），服务器在最新版本上改，写回时带 ETag（ifMatch），两个人同时保存时后到的重试，
// 不会互相覆盖。每次修改都记下是谁（GitHub 登录名）。只给登录用户。
import { get, put, BlobPreconditionFailedError } from "@vercel/blob";
import { readSession } from "../lib/session.js";

const PATH = "field/state.json";
const RUBRIC_LEN = { 1: 4, 2: 5, 3: 6 };
const ROUNDS = ["1", "2", "3"];
const MAX_TRIALS = 40, MAX_CKS = 60, MAX_LOG = 300;

const empty = () => ({ v: 1, selftest: {}, check: {}, log: [] });

async function readState() {
  const r = await get(PATH, { access: "private", useCache: false });
  if (!r || r.statusCode !== 200) return { state: empty(), etag: null };
  const text = await new Response(r.stream).text();
  // get() returns a weak ETag (W/"…"), put(ifMatch) compares the strong one: strip the prefix, or every write fails
  return { state: JSON.parse(text), etag: String(r.blob.etag || "").replace(/^W\//, "") || null };
}

function bad(msg) { const e = new Error(msg); e.bad = true; return e; }
const int = (x, lo, hi) => { const n = Number(x); if (!Number.isInteger(n) || n < lo || n > hi) throw bad("bad number"); return n; };
function ckName(x) {
  const s = String(x ?? "").trim();
  if (!s || s.length > 40 || /[\u0000-\u001f]/.test(s)) throw bad("checkpoint 名字要 1–40 个字");
  return s;
}

function apply(st, body, who) {
  const at = new Date().toISOString();
  const op = body.op;
  const trials = (ck, t) => {
    if (!st.selftest[ck]) throw bad("没有这个 checkpoint");
    return (st.selftest[ck][t] ||= []);
  };
  if (op === "ck_add") {
    const ck = ckName(body.ck);
    if (!st.selftest[ck]) {
      if (Object.keys(st.selftest).length >= MAX_CKS) throw bad("checkpoint 太多了");
      st.selftest[ck] = { 1: [], 2: [], 3: [], by: who, at };
    }
  } else if (op === "ck_del") {
    const ck = ckName(body.ck), c = st.selftest[ck];
    if (c && [1, 2, 3].some((t) => (c[t] || []).length)) throw bad("这个 checkpoint 还有记录，先把记录删掉");
    delete st.selftest[ck];
  } else if (op === "trial_add") {
    const ck = ckName(body.ck), t = int(body.task, 1, 3), L = trials(ck, t);
    if (L.length >= MAX_TRIALS) throw bad("这个任务记得太多了");
    L.push({ s: Array(RUBRIC_LEN[t]).fill(0), by: who, at });
  } else if (op === "trial_set") {
    const ck = ckName(body.ck), t = int(body.task, 1, 3), L = trials(ck, t);
    const i = int(body.i, 0, L.length - 1), j = int(body.j, 0, RUBRIC_LEN[t] - 1);
    L[i].s[j] = body.v ? 1 : 0; L[i].by = who; L[i].at = at;
  } else if (op === "trial_del") {
    const ck = ckName(body.ck), t = int(body.task, 1, 3), L = trials(ck, t);
    L.splice(int(body.i, 0, L.length - 1), 1);
  } else if (op === "check_set") {
    const r = String(body.round); if (!ROUNDS.includes(r)) throw bad("bad round");
    const i = int(body.i, 0, 19);
    (st.check[r] ||= {})[i] = { v: !!body.v, by: who, at };
  } else if (op === "check_reset") {
    const r = String(body.round); if (!ROUNDS.includes(r)) throw bad("bad round");
    st.check[r] = {};
  } else {
    throw bad("unknown op");
  }
  st.log = [...(st.log || []), { at, by: who, op, ck: body.ck, task: body.task, round: body.round }].slice(-MAX_LOG);
  st.updated = at;
  return st;
}

async function readBody(req) {
  if (req.body && typeof req.body === "object") return req.body;
  let raw = "";
  for await (const c of req) { raw += c; if (raw.length > 20000) throw bad("too large"); }
  return JSON.parse(raw || "{}");
}

export default async function handler(req, res) {
  res.setHeader("Cache-Control", "private, no-store");
  res.setHeader("X-Robots-Tag", "noindex, nofollow");
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  const s = readSession(req.headers.cookie);
  if (!s) { res.statusCode = 401; return res.end(JSON.stringify({ error: "unauthorized" })); }
  try {
    if (req.method === "GET") {
      const { state } = await readState();
      return res.end(JSON.stringify(state));
    }
    if (req.method !== "POST") { res.statusCode = 405; return res.end(JSON.stringify({ error: "method" })); }
    if (!String(req.headers["content-type"] || "").startsWith("application/json")) throw bad("json only");
    const body = await readBody(req);
    for (let attempt = 0; attempt < 8; attempt++) {
      if (attempt) await new Promise((r) => setTimeout(r, 60 + Math.random() * 240 * attempt));   // spread retries out
      const { state, etag } = await readState();
      const next = apply(state, body, s.login || String(s.uid));
      try {
        await put(PATH, JSON.stringify(next), {
          access: "private", contentType: "application/json", addRandomSuffix: false,
          ...(etag ? { allowOverwrite: true, ifMatch: etag } : { allowOverwrite: false }),
        });
        return res.end(JSON.stringify(next));
      } catch (e) {
        // someone else wrote in between: ETag mismatch, a concurrent conditional write, or the file appeared meanwhile
        const clash = e instanceof BlobPreconditionFailedError || /already exists|precondition|conflict/i.test(String(e && e.message));
        if (!clash || attempt === 7) throw e;
      }
    }
  } catch (e) {
    if (!e.bad) console.error("field-state", req.method, e && e.name, e && e.message);
    res.statusCode = e.bad ? 400 : 500;
    return res.end(JSON.stringify({ error: e.bad ? e.message : `保存失败，请再试一次（${e && e.name}: ${String(e && e.message).slice(0, 160)}）` }));
  }
}
