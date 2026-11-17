// 给 集群 的 cron 用：最近一次有人打开看板是多少秒前（没有记录 -> null）。不走登录中间件，用 PRESENCE_TOKEN 鉴权
// （Vercel 环境变量；集群 上存在 ~/.config/lingbot-board/presence_token）。只回一个数字，不含看板内容。
import { getCache } from "@vercel/functions";
import { same } from "../lib/session.js";
import { SEEN_KEY } from "../lib/presence.js";

export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  res.setHeader("Content-Type", "application/json");
  const want = (process.env.PRESENCE_TOKEN || "").trim();
  const got = String(req.headers.authorization || "").replace(/^Bearer\s+/i, "").trim();
  if (want.length < 32 || !same(got, want)) {
    res.statusCode = 401;
    return res.end('{"error":"unauthorized"}');
  }
  let t;
  try { t = await getCache().get(SEEN_KEY); } catch { t = undefined; }
  res.statusCode = 200;
  res.end(JSON.stringify({ age_s: typeof t === "number" ? Math.max(0, Math.round((Date.now() - t) / 1000)) : null }));
}
