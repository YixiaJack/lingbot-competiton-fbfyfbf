// 现场页的数据：field 分支的 field_data.json（由 field_build.py 生成，现场从队员电脑推送）。只给登录用户。
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
    f = await readRepoFile("field_data.json", { ttlMs: 15000 });
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
