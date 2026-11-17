// 第一道登录检查：在 CDN 缓存和函数之前运行（Vercel Routing Middleware，Node.js）。没登录的 GET 跳到 GitHub 登录，
// 其他方法回 401。登录流程本身（/api/auth/*）、集群 用的 /api/presence（自带令牌鉴权）和 robots.txt 不检查。
// api/board.js 自己还会再验一次。
import { next } from "@vercel/functions";
import { readSession, loginPath } from "./lib/session.js";

export const config = { matcher: ["/((?!api/auth/|api/presence$|robots\\.txt$).*)"] };

export default function middleware(request) {
  if (readSession(request.headers.get("cookie"))) return next();
  if (request.method !== "GET" && request.method !== "HEAD") {
    return new Response("unauthorized", { status: 401, headers: { "Cache-Control": "no-store" } });
  }
  const u = new URL(request.url);
  return new Response(null, { status: 302, headers: { Location: loginPath(u.pathname + u.search), "Cache-Control": "no-store" } });
}
