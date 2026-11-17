// 登录第二步：核对 state，用 code + PKCE verifier 换 GitHub token，查是谁，只放行 ALLOWED_GITHUB_IDS 里的账号。
// 查完就把这个 GitHub token 作废（之后用不到），只发看板自己的 session cookie。
import { waitUntil } from "@vercel/functions";
import {
  ORIGIN, CALLBACK, OAUTH_COOKIE, SESSION_COOKIE, SESSION_DAYS,
  now, sign, unsign, same, getCookie, setCookie, allowedIds, page, redirect,
} from "../../lib/session.js";

const GH = { "User-Agent": "lingbot-board", "X-GitHub-Api-Version": "2022-11-28" };

export default async function handler(req, res) {
  const clear = setCookie(OAUTH_COOKIE, "", 0);
  const q = new URL(req.url, ORIGIN).searchParams;
  if (q.get("error")) return page(res, 403, "没有完成 GitHub 授权", q.get("error_description") || q.get("error"), [clear]);
  const o = unsign(getCookie(req.headers.cookie, OAUTH_COOKIE));
  const code = q.get("code"), state = q.get("state");
  if (!o || o.k !== "oauth" || !code || !state || !same(state, o.st)) {
    return page(res, 400, "登录已过期", "这次登录的校验信息对不上，或者已经超过 10 分钟。", [clear]);
  }
  const id = (process.env.GITHUB_OAUTH_CLIENT_ID || "").trim();
  const secret = (process.env.GITHUB_OAUTH_CLIENT_SECRET || "").trim();
  if (!id || !secret) return page(res, 503, "登录还没配置好", "GitHub OAuth App 的 Client ID 或 Secret 还没设置。", [clear]);
  let token, user;
  try {
    const r = await fetch("https://github.com/login/oauth/access_token", {
      method: "POST",
      headers: { ...GH, Accept: "application/json", "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ client_id: id, client_secret: secret, code, redirect_uri: ORIGIN + CALLBACK, code_verifier: o.v }),
    });
    const j = await r.json().catch(() => ({}));
    token = j.access_token;
    if (!token) throw new Error(j.error_description || j.error || `token HTTP ${r.status}`);
    const u = await fetch("https://api.github.com/user", {
      headers: { ...GH, Accept: "application/vnd.github+json", Authorization: `Bearer ${token}` },
    });
    user = await u.json().catch(() => ({}));
    if (!u.ok || !user.id) throw new Error(`GitHub /user HTTP ${u.status}`);
  } catch (e) {
    return page(res, 502, "GitHub 登录失败", e.message, [clear]);
  } finally {
    if (token) waitUntil(revoke(id, secret, token));
  }
  if (!allowedIds().has(String(user.id))) {
    return page(res, 403, "这个 GitHub 账号没有权限", `@${user.login} 不在允许名单里。`, [clear],
      { href: "https://github.com/logout", label: "换一个 GitHub 账号" });
  }
  const session = sign({ k: "session", uid: user.id, login: user.login, exp: now() + SESSION_DAYS * 86400 });
  redirect(res, o.n || "/", [clear, setCookie(SESSION_COOKIE, session, SESSION_DAYS * 86400)]);
}

// 作废这次的 token；对这个 App 的授权保留，下次登录不用再点"授权"
function revoke(id, secret, token) {
  return fetch(`https://api.github.com/applications/${encodeURIComponent(id)}/token`, {
    method: "DELETE",
    headers: {
      ...GH, Accept: "application/vnd.github+json", "Content-Type": "application/json",
      Authorization: "Basic " + Buffer.from(`${id}:${secret}`).toString("base64"),
    },
    body: JSON.stringify({ access_token: token }),
  }).catch(() => {});
}
