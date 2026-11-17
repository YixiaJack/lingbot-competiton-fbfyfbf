// 登录第一步：生成 state 和 PKCE verifier，签名后放进短期 cookie，再跳到 GitHub 授权页。不申请任何 scope，
// 只用来确认是谁（GitHub 用户数字 ID）。
import { randomBytes, createHash } from "node:crypto";
import { ORIGIN, CALLBACK, OAUTH_COOKIE, now, sign, setCookie, safeNext, page, redirect } from "../../lib/session.js";

export default function handler(req, res) {
  const clientId = (process.env.GITHUB_OAUTH_CLIENT_ID || "").trim();
  if (!clientId) return page(res, 503, "登录还没配置好", "GitHub OAuth App 的 Client ID 还没设置。");
  let token;
  const state = randomBytes(32).toString("base64url");
  const verifier = randomBytes(32).toString("base64url");
  try {
    const next = safeNext(new URL(req.url, ORIGIN).searchParams.get("next"));
    token = sign({ k: "oauth", st: state, v: verifier, n: next, exp: now() + 600 });
  } catch (e) {
    return page(res, 503, "登录还没配置好", e.message);
  }
  const a = new URL("https://github.com/login/oauth/authorize");
  a.search = new URLSearchParams({
    client_id: clientId,
    redirect_uri: ORIGIN + CALLBACK,
    state,
    code_challenge: createHash("sha256").update(verifier).digest("base64url"),
    code_challenge_method: "S256",
    allow_signup: "false",
  }).toString();
  redirect(res, a.toString(), [setCookie(OAUTH_COOKIE, token, 600)]);
}
