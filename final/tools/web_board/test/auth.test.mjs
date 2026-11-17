// 本地测试：node --test test/auth.test.mjs   （GitHub 的接口用假的 fetch 代替）
import { test } from "node:test";
import assert from "node:assert/strict";
import { randomBytes, createHash } from "node:crypto";

process.env.SESSION_SECRET = randomBytes(48).toString("base64");
process.env.ALLOWED_GITHUB_IDS = "145562489";
process.env.GITHUB_OAUTH_CLIENT_ID = "Iv1.testclient";
process.env.GITHUB_OAUTH_CLIENT_SECRET = "testsecret";
process.env.GITHUB_TOKEN = "data-token";

const S = await import("../lib/session.js");
const middleware = (await import("../middleware.js")).default;
const login = (await import("../api/auth/login.js")).default;
const callback = (await import("../api/auth/callback.js")).default;
const logout = (await import("../api/auth/logout.js")).default;
const board = (await import("../api/board.js")).default;
const fieldImage = (await import("../api/field-img.js")).default;

function mockRes() {
  const h = {};
  return {
    statusCode: 200, headers: h, body: "",
    setHeader(k, v) { h[k.toLowerCase()] = v; },
    end(b) { this.body = b || ""; this.ended = true; },
  };
}
const cookies = (res) => [].concat(res.headers["set-cookie"] || []);
const cookieVal = (res, name) => {
  const c = cookies(res).find((x) => x.startsWith(name + "="));
  return c ? c.slice(name.length + 1).split(";")[0] : null;
};
const session = (over = {}) => S.sign({ k: "session", uid: 145562489, login: "YixiaJack", exp: S.now() + 3600, ...over });
const req = (url, cookie, method = "GET") => new Request("https://lingbot-board.vercel.app" + url, { method, headers: cookie ? { cookie } : {} });

test("middleware: no cookie -> 302 to login with next", () => {
  const r = middleware(req("/?x=1"));
  assert.equal(r.status, 302);
  assert.equal(r.headers.get("location"), "/api/auth/login?next=%2F%3Fx%3D1");
});

test('field images require login and only allow exact thumbnail/background paths',async()=>{
  let res=mockRes();
  await fieldImage({url:'/api/field-img?f=bg_private_t1.jpg',headers:{}},res);
  assert.equal(res.statusCode,401);
  const cookie=`${S.SESSION_COOKIE}=${session()}`;
  for(const name of ['../config.json','bg_private_t4.jpg','bg_private_t1.jpg/../../secret','bg_unknown_t1.jpg','thumbs/647.jpg.exe']){
    res=mockRes();await fieldImage({url:'/api/field-img?f='+encodeURIComponent(name),headers:{cookie}},res);
    assert.equal(res.statusCode,400,name);
  }
  const original=globalThis.fetch;
  try{
    globalThis.fetch=async()=>new Response(new Uint8Array([255,216,255]),{status:200});
    for(const name of ['bg_private_t1.jpg','bg_public_t3.jpg','bg_t1.jpg','thumbs/758_end.jpg']){
      res=mockRes();await fieldImage({url:'/api/field-img?f='+name,headers:{cookie}},res);
      assert.equal(res.statusCode,200,name);assert.equal(res.headers['content-type'],'image/jpeg');
    }
  }finally{globalThis.fetch=original}
});

test("middleware: valid session passes", () => {
  const r = middleware(req("/", `${S.SESSION_COOKIE}=${session()}`));
  assert.equal(r.headers.get("x-middleware-next"), "1");
});

test("middleware: forged / expired / wrong kind / not allowed / other key -> blocked", () => {
  const good = session();
  const forged = good.slice(0, -3) + (good.endsWith("AAA") ? "BBB" : "AAA");
  const payloadSwap = Buffer.from(JSON.stringify({ k: "session", uid: 1, exp: S.now() + 3600 })).toString("base64url") + "." + good.split(".")[1];
  for (const c of [forged, payloadSwap, session({ exp: S.now() - 1 }), session({ k: "oauth" }), session({ uid: 42 }), "garbage", ""]) {
    assert.equal(middleware(req("/", `${S.SESSION_COOKIE}=${c}`)).status, 302, c);
  }
  assert.equal(middleware(req("/api/board", `${S.SESSION_COOKIE}=${forged}`, "POST")).status, 401);
});

test("middleware: secret missing -> nothing passes", () => {
  const keep = process.env.SESSION_SECRET;
  const good = session();
  process.env.SESSION_SECRET = "short";
  assert.equal(middleware(req("/", `${S.SESSION_COOKIE}=${good}`)).status, 302);
  process.env.SESSION_SECRET = keep;
});

test("safeNext rejects open redirects", () => {
  for (const bad of ["//evil.com", "/\\evil.com", "/\t/evil.com", "/\n/evil.com", "https://evil.com", "evil", "/ space", "", null])
    assert.equal(S.safeNext(bad), "/", String(bad));
  assert.equal(S.safeNext("/api/board?x=1"), "/api/board?x=1");
});

test("login: redirects to GitHub with state + S256 PKCE, sets signed oauth cookie", () => {
  const res = mockRes();
  login({ url: "/api/auth/login?next=%2F%3Fq%3D1", headers: {} }, res);
  assert.equal(res.statusCode, 302);
  const a = new URL(res.headers.location);
  assert.equal(a.origin + a.pathname, "https://github.com/login/oauth/authorize");
  assert.equal(a.searchParams.get("client_id"), "Iv1.testclient");
  assert.equal(a.searchParams.get("redirect_uri"), "https://lingbot-board.vercel.app/api/auth/callback");
  assert.equal(a.searchParams.get("code_challenge_method"), "S256");
  assert.equal(a.searchParams.get("scope"), null);
  const o = S.unsign(cookieVal(res, S.OAUTH_COOKIE));
  assert.equal(o.k, "oauth");
  assert.equal(o.n, "/?q=1");
  assert.equal(a.searchParams.get("state"), o.st);
  assert.equal(a.searchParams.get("code_challenge"), createHash("sha256").update(o.v).digest("base64url"));
  assert.match(cookies(res)[0], /HttpOnly; Secure; SameSite=Lax; Max-Age=600/);
});

function fakeGitHub(userId, calls) {
  return async (url, init = {}) => {
    calls.push([String(url), init]);
    if (String(url).startsWith("https://github.com/login/oauth/access_token")) {
      return new Response(JSON.stringify({ access_token: "gho_test", token_type: "bearer", scope: "" }), { status: 200 });
    }
    if (String(url) === "https://api.github.com/user") {
      return new Response(JSON.stringify({ id: userId, login: userId === 145562489 ? "YixiaJack" : "someone" }), { status: 200 });
    }
    if (String(url).includes("/applications/")) return new Response(null, { status: 204 });
    throw new Error("unexpected " + url);
  };
}

async function startLogin() {
  const res = mockRes();
  login({ url: "/api/auth/login?next=%2F", headers: {} }, res);
  const oc = cookieVal(res, S.OAUTH_COOKIE);
  return { oc, state: new URL(res.headers.location).searchParams.get("state"), verifier: S.unsign(oc).v };
}

test("callback: allowed user gets a session; token exchange sends the PKCE verifier; token revoked", async () => {
  const { oc, state, verifier } = await startLogin();
  const calls = [];
  globalThis.fetch = fakeGitHub(145562489, calls);
  const res = mockRes();
  await callback({ url: `/api/auth/callback?code=abc&state=${state}`, headers: { cookie: `${S.OAUTH_COOKIE}=${oc}` } }, res);
  assert.equal(res.statusCode, 302);
  assert.equal(res.headers.location, "/");
  const s = S.readSession(`${S.SESSION_COOKIE}=${cookieVal(res, S.SESSION_COOKIE)}`);
  assert.equal(s.uid, 145562489);
  assert.equal(cookieVal(res, S.OAUTH_COOKIE), "");  // oauth cookie cleared
  const body = new URLSearchParams(String(calls[0][1].body));
  assert.equal(body.get("code_verifier"), verifier);
  assert.equal(body.get("redirect_uri"), "https://lingbot-board.vercel.app/api/auth/callback");
  await new Promise((r) => setTimeout(r, 10));
  assert.ok(calls.some(([u, i]) => u.includes("/applications/Iv1.testclient/token") && i.method === "DELETE"));
});

test("callback: other GitHub user -> 403, no session", async () => {
  const { oc, state } = await startLogin();
  globalThis.fetch = fakeGitHub(999, []);
  const res = mockRes();
  await callback({ url: `/api/auth/callback?code=abc&state=${state}`, headers: { cookie: `${S.OAUTH_COOKIE}=${oc}` } }, res);
  assert.equal(res.statusCode, 403);
  assert.equal(cookieVal(res, S.SESSION_COOKIE), null);
});

test("callback: wrong state / no oauth cookie / session cookie used as oauth -> 400, GitHub not called", async () => {
  const { oc } = await startLogin();
  const calls = [];
  globalThis.fetch = fakeGitHub(145562489, calls);
  for (const [qs, cookie] of [["code=abc&state=nope", `${S.OAUTH_COOKIE}=${oc}`], ["code=abc&state=x", ""],
                              ["code=abc&state=x", `${S.OAUTH_COOKIE}=${session()}`]]) {
    const res = mockRes();
    await callback({ url: `/api/auth/callback?${qs}`, headers: { cookie } }, res);
    assert.equal(res.statusCode, 400, qs + " " + cookie.slice(0, 20));
  }
  assert.equal(calls.length, 0);
});

test("board: needs a session, never cacheable publicly, reads board_v2.html", async () => {
  const calls = [];
  globalThis.fetch = async (url, init) => { calls.push([String(url), init]); return new Response("<html>board</html>", { status: 200 }); };
  let res = mockRes();
  await board({ url: "/", headers: {} }, res);
  assert.equal(res.statusCode, 302);
  assert.equal(res.headers["cache-control"], "private, no-store");
  assert.equal(calls.length, 0);
  res = mockRes();
  await board({ url: "/", headers: { cookie: `${S.SESSION_COOKIE}=${session()}` } }, res);
  assert.equal(res.statusCode, 200);
  assert.equal(res.body, "<html>board</html>");
  assert.equal(res.headers["cache-control"], "private, no-store");
  assert.equal(calls[0][0], "https://api.github.com/repos/YixiaJack/lingbot-board-data/contents/board_v2.html?ref=main");
});

test("logout clears the session cookie", () => {
  const res = mockRes();
  logout({ url: "/api/auth/logout", headers: {} }, res);
  assert.match(cookies(res)[0], new RegExp(`^${S.SESSION_COOKIE}=; .*Max-Age=0`));
});

const presence = (await import("../api/presence.js")).default;
const { getCache } = await import("@vercel/functions");
const { SEEN_KEY } = await import("../lib/presence.js");
const ask = async (auth) => {
  const res = mockRes();
  await presence({ url: "/api/presence", headers: auth ? { authorization: auth } : {} }, res);
  return res;
};
const tick = () => new Promise((r) => setTimeout(r, 20));

test("presence: refuses without a configured token or with a wrong one", async () => {
  delete process.env.PRESENCE_TOKEN;
  assert.equal((await ask("Bearer anything")).statusCode, 401);
  process.env.PRESENCE_TOKEN = randomBytes(32).toString("hex");
  assert.equal((await ask(null)).statusCode, 401);
  assert.equal((await ask("Bearer " + "0".repeat(64))).statusCode, 401);
});

test("presence: only logged-in board views count", async () => {
  process.env.PRESENCE_TOKEN = randomBytes(32).toString("hex");
  const auth = "Bearer " + process.env.PRESENCE_TOKEN;
  globalThis.fetch = async () => new Response("<html>board</html>", { status: 200 });
  await getCache().delete(SEEN_KEY);
  let res = await ask(auth);
  assert.equal(res.statusCode, 200);
  assert.deepEqual(JSON.parse(res.body), { age_s: null });
  await board({ url: "/", headers: {} }, mockRes());  // not logged in: redirected, not counted
  await tick();
  assert.deepEqual(JSON.parse((await ask(auth)).body), { age_s: null });
  await board({ url: "/", headers: { cookie: `${S.SESSION_COOKIE}=${session()}` } }, mockRes());
  await tick();
  res = await ask(auth);
  assert.equal(res.headers["cache-control"], "no-store");
  assert.ok(JSON.parse(res.body).age_s <= 1, res.body);
});
