// 有没有人在看看板：登录用户每打开一次（页面开着时每分钟自动重载一次），api/board.js 就把时间记进 Vercel Runtime Cache
// （同一区域的函数共享；看板和 /api/presence 都在 iad1）。集群 的 cron 每 2 分钟问一次 /api/presence：最近 10 分钟
// 有人看就照常每 2 分钟推，没人看就最多每 10 分钟推一次。缓存丢了只会让推送变慢，不影响别的。
export const SEEN_KEY = "board:last-seen";
export const SEEN_TTL_S = 1800;
