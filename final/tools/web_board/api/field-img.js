// 现场页的图片：field 分支里的缩略图（thumbs/<n>.jpg）和各任务的空桌面底图（bg_t<k>.jpg）。只给登录用户，只认这两种文件名。
import { readSession } from "../lib/session.js";
import { readRepoFile } from "../lib/github_file.js";

const OK = /^(thumbs\/\d{1,6}(_end)?\.jpg|bg_t[0-3]\.jpg|bg_(public|private)_t[1-3]\.jpg)$/;   // fixed image names only

export default async function handler(req, res) {
  res.setHeader("X-Robots-Tag", "noindex, nofollow");
  if (!readSession(req.headers.cookie)) {
    res.statusCode = 401;
    res.setHeader("Cache-Control", "private, no-store");
    return res.end("unauthorized");
  }
  const name = new URL(req.url, "http://x").searchParams.get("f") || "";
  if (!OK.test(name)) {
    res.statusCode = 400;
    return res.end("bad name");
  }
  let f;
  try {
    f = await readRepoFile(name, { ttlMs: name.startsWith('bg_') ? 15000 : 300000, binary: true });
  } catch (e) {
    f = { ok: false, status: String(e) };
  }
  if (!f.ok) {
    res.statusCode = f.status === 404 ? 404 : 502;
    res.setHeader("Cache-Control", "private, no-store");
    return res.end(`GitHub ${f.status}`);
  }
  res.statusCode = 200;
  res.setHeader("Content-Type", "image/jpeg");
  res.setHeader("Cache-Control", name.startsWith('bg_') ? "private, max-age=30" : "private, max-age=300");
  res.end(f.body);
}
