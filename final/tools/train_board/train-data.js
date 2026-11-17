// 训练进度页（/train）的数据：train 分支的 train_status.json。队员电脑上的 train_board.py 每 2 分钟经堡垒机读一次训练机，
// 推到这个分支（只有它写，force-push 单个提交）。只给登录用户。
import { readSession } from "../lib/session.js";
import { readRepoFile } from "../lib/github_file.js";

export default async function handler(req, res) {
  res.setHeader("Cache-Control", "private, no-store");
  res.setHeader("X-Robots-Tag", "noindex, nofollow");
  if (!readSession(req.headers.cookie)) {
    res.statusCode = 401;
    return res.end("unauthorized");
  }
  let f;
  try {
    f = await readRepoFile("train_status.json", { ref: "train", ttlMs: 15000 });
  } catch (e) {
    f = { ok: false, status: String(e) };
  }
  if (!f.ok) {
    res.statusCode = 502;
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    return res.end(JSON.stringify({ error: `GitHub ${f.status}` }));
  }
  res.statusCode = 200;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(f.body);
}
