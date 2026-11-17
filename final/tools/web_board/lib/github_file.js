// 从私有仓库 YixiaJack/lingbot-board-data 读文件（GITHUB_TOKEN：只读、只对这个仓库的 fine-grained token）。
// 现场页的数据在 field 分支：集群 的看板 cron 每轮 force-push main，放在 main 上会被冲掉。
// 同一个函数实例里缓存 ttlMs，少调 GitHub API。
const REPO = "YixiaJack/lingbot-board-data";
const memo = new Map();

export async function readRepoFile(path, { ref = "field", ttlMs = 15000, binary = false } = {}) {
  const key = `${ref}:${path}`;
  const hit = memo.get(key);
  if (hit && Date.now() - hit.at < ttlMs) return hit;
  const r = await fetch(`https://api.github.com/repos/${REPO}/contents/${path}?ref=${ref}`, {
    headers: {
      Authorization: `Bearer ${(process.env.GITHUB_TOKEN || "").trim()}`,
      Accept: "application/vnd.github.raw+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "lingbot-board",
    },
  });
  if (!r.ok) return { ok: false, status: r.status };
  const body = binary ? Buffer.from(await r.arrayBuffer()) : await r.text();
  const out = { ok: true, body, at: Date.now() };
  if (memo.size > 300) memo.clear();
  memo.set(key, out);
  return out;
}
