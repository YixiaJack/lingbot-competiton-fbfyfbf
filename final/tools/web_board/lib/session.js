// 看板登录的共用部分：GitHub OAuth 登录后发一个签名的 session cookie。
// cookie 值 = base64url(JSON) + "." + HMAC-SHA256(SESSION_SECRET)，不存任何服务器端状态；换 SESSION_SECRET 即全部失效。
// 只认 ALLOWED_GITHUB_IDS 里的 GitHub 用户数字 ID（用户名可以改，数字 ID 不变），名单里删掉的 ID 下次部署后立即失效。
// 环境变量没配好时一律当作未登录（宁可打不开，也不放行）。
import { createHmac, timingSafeEqual } from "node:crypto";

export const ORIGIN = "https://lingbot-board.vercel.app";  // GitHub OAuth App 的回调地址必须正好是 ORIGIN + CALLBACK
export const CALLBACK = "/api/auth/callback";
export const SESSION_COOKIE = "__Host-lb_session";  // __Host- 前缀：浏览器只接受 Secure、Path=/、不带 Domain 的写法
export const OAUTH_COOKIE = "__Host-lb_oauth";      // 登录跳转到 GitHub 期间的 state 和 PKCE verifier，10 分钟有效
export const SESSION_DAYS = 7;

function key() {
  const s = (process.env.SESSION_SECRET || "").trim();
  return s.length >= 32 ? s : null;
}

export function allowedIds() {
  return new Set((process.env.ALLOWED_GITHUB_IDS || "").split(/[\s,]+/).filter((x) => /^\d+$/.test(x)));
}

export const now = () => Math.floor(Date.now() / 1000);

export function sign(obj) {
  const k = key();
  if (!k) throw new Error("SESSION_SECRET 没有配置");
  const p = Buffer.from(JSON.stringify(obj)).toString("base64url");
  return `${p}.${createHmac("sha256", k).update(p).digest("base64url")}`;
}

// 签名对、没过期 -> 对象；否则 null
export function unsign(token) {
  const k = key();
  if (!k || typeof token !== "string") return null;
  const i = token.lastIndexOf(".");
  if (i <= 0) return null;
  const p = token.slice(0, i);
  const mac = Buffer.from(token.slice(i + 1));
  const want = Buffer.from(createHmac("sha256", k).update(p).digest("base64url"));
  if (mac.length !== want.length || !timingSafeEqual(mac, want)) return null;
  let obj;
  try { obj = JSON.parse(Buffer.from(p, "base64url").toString("utf8")); } catch { return null; }
  return obj && typeof obj.exp === "number" && obj.exp > now() ? obj : null;
}

export function same(a, b) {  // 定长时间比较两个字符串
  const x = Buffer.from(String(a)), y = Buffer.from(String(b));
  return x.length === y.length && timingSafeEqual(x, y);
}

export function getCookie(header, name) {
  for (const part of String(header || "").split(";")) {
    const j = part.indexOf("=");
    if (j > 0 && part.slice(0, j).trim() === name) return part.slice(j + 1).trim();
  }
  return null;
}

export function setCookie(name, value, maxAge) {
  return `${name}=${value}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${maxAge}`;
}

// 有效的登录 -> {uid, login, exp}；否则 null。k 区分用途，OAuth 过程中的 cookie 不能当登录用。
export function readSession(cookieHeader) {
  const s = unsign(getCookie(cookieHeader, SESSION_COOKIE));
  return s && s.k === "session" && allowedIds().has(String(s.uid)) ? s : null;
}

// 登录后回到哪：只收本站路径。拒绝 "//x"、"/\x"、控制字符和空格（浏览器会删掉 tab/换行，"/\t/x" 会变成 "//x"）。
export function safeNext(p) {
  return typeof p === "string" && p.length < 2000 && /^\/(?![/\\])[\x21-\x5b\x5d-\x7e]*$/.test(p) ? p : "/";
}

export function loginPath(pathAndQuery) {
  return `/api/auth/login?next=${encodeURIComponent(safeNext(pathAndQuery))}`;
}

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// Node 函数里回一个简单的说明页
export function page(res, status, title, text, cookies = [], link = { href: "/", label: "重新登录" }) {
  res.statusCode = status;
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  if (cookies.length) res.setHeader("Set-Cookie", cookies);
  res.end(`<!doctype html><html lang="zh"><head><meta charset="utf-8">` +
    `<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex, nofollow">` +
    `<title>${esc(title)}</title><style>body{font:16px/1.6 -apple-system,"PingFang SC",sans-serif;max-width:32rem;` +
    `margin:15vh auto;padding:0 16px;color:#1d2128;background:#f6f7f9}@media(prefers-color-scheme:dark){body{` +
    `color:#e7e9ee;background:#14161a}a{color:#5b93f0}}h1{font-size:20px}p{color:#6b7380}</style></head><body>` +
    `<h1>${esc(title)}</h1><p>${esc(text)}</p><p><a href="${esc(link.href)}">${esc(link.label)}</a></p></body></html>`);
}

export function redirect(res, location, cookies = []) {
  res.statusCode = 302;
  res.setHeader("Location", location);
  res.setHeader("Cache-Control", "no-store");
  if (cookies.length) res.setHeader("Set-Cookie", cookies);
  res.end();
}
