// 看板页面：读 集群 推到私有仓库 YixiaJack/lingbot-board-data 的 board_v2.html（集群 每 2 分钟推一次）。
// 只给登录用户：middleware.js 先验一次，这里再验一次。响应是 private + no-store，CDN 不缓存
// （以前 s-maxage=15，加了登录以后缓存过的页面会被原样发给没登录的人）。同一个函数实例 15 秒内复用上次取到的页面，
// 少调 GitHub API。文件名从 board.html 改成了 board_v2.html：没有登录的旧版代码再被部署上来，只会取不到数据，
// 不会把看板公开出去。
// GITHUB_TOKEN：Vercel 项目的环境变量，只读、只对这个仓库的 fine-grained token。
// 每次登录用户打开都记一下时间（lib/presence.js），集群 据此决定推送频率。
import { getCache, waitUntil } from "@vercel/functions";
import { readSession, loginPath } from "../lib/session.js";
import { SEEN_KEY, SEEN_TTL_S } from "../lib/presence.js";

const REPO = "YixiaJack/lingbot-board-data";
const FILE = "board_v2.html";
let memo = { html: null, at: 0 };

export default async function handler(req, res) {
  res.setHeader("Cache-Control", "private, no-store");
  res.setHeader("X-Robots-Tag", "noindex, nofollow");
  if (!readSession(req.headers.cookie)) {
    res.statusCode = 302;
    res.setHeader("Location", loginPath(req.url || "/"));
    return res.end();
  }
  waitUntil(Promise.resolve().then(() => getCache().set(SEEN_KEY, Date.now(), { ttl: SEEN_TTL_S, name: "board-last-seen" }))
    .catch(() => {}));
  if (!memo.html || Date.now() - memo.at > 15000) {
    let r;
    try {
      r = await fetch(`https://api.github.com/repos/${REPO}/contents/${FILE}?ref=main`, {
        headers: {
          Authorization: `Bearer ${(process.env.GITHUB_TOKEN || "").trim()}`,
          Accept: "application/vnd.github.raw+json",
          "X-GitHub-Api-Version": "2022-11-28",
          "User-Agent": "lingbot-board",
        },
      });
    } catch (e) {
      r = { ok: false, status: String(e) };
    }
    if (!r.ok) {
      res.statusCode = 502;
      res.setHeader("Content-Type", "text/html; charset=utf-8");
      return res.end(`<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">` +
        `<title>LingBot 看板</title><p>看板数据暂时取不到（GitHub ${r.status}），过一分钟刷新试试。</p>`);
    }
    memo = { html: await r.text(), at: Date.now() };
  }
  res.statusCode = 200;
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.end(withFieldNav(memo.html));
}

// 入口：左边栏最上面加一个"现场"分组，点进现场作战台（/field）。看板页面本身在 集群 上生成，这里在返回前插进去，
// 不用改 集群 的构建脚本；万一页面结构变了找不到左边栏，就退回右下角的浮动按钮。
const FIELD_NAV = `<nav id="fieldnav"><div class="ngroup">现场</div><a class="nitem" href="/field">` +
  `<span class="nt">现场作战台 →</span><span class="ns">数据覆盖 · 质量 · 评测 · 自测打分</span></a>` +
  `<a class="nitem" href="/train"><span class="nt">训练进度 →</span><span class="ns">训练机当前训练 · 曲线 · checkpoint · GPU</span></a></nav>` +
  `<style>#fieldnav{display:flex;flex-direction:column;gap:4px;margin-top:14px}#fieldnav .ngroup{margin-top:0}` +
  `@media (max-width:900px){#fieldnav{flex-direction:row;margin:10px 0 0}}</style>`;
const FIELD_BUTTON = `<a href="/field" style="position:fixed;right:14px;bottom:14px;z-index:50;background:#2f6fdb;color:#fff;` +
  `padding:8px 14px;border-radius:999px;font:600 13px -apple-system,'PingFang SC',sans-serif;text-decoration:none;` +
  `box-shadow:0 2px 8px rgba(0,0,0,.25)">现场作战台 →</a>`;
function withFieldNav(html) {
  const anchor = '<nav id="projnav"></nav>';
  return html.includes(anchor) ? html.replace(anchor, FIELD_NAV + anchor) : html.replace("</body>", FIELD_BUTTON + "</body>");
}
