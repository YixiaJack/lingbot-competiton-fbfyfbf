// 退出：清掉这台设备上的 session cookie。GitHub 本身的登录不受影响，所以"重新登录"通常一跳就回来。
import { SESSION_COOKIE, setCookie, page } from "../../lib/session.js";

export default function handler(req, res) {
  page(res, 200, "已退出登录", "这台设备上的看板登录已经清掉。", [setCookie(SESSION_COOKIE, "", 0)]);
}
