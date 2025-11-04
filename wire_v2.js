export const PROTO = 2;
export const BUILD = "2025-09-09-v14";

export const MSG = {
  HELLO: "hello",
  INIT_MODEL: "init_model",
  LOAD_SHARD: "load_shard",
  SHARD_READY: "shard_ready",
  DECODE_STEP: "decode_step",
  STATE_OUT: "state_out",
  NEXT_TOKEN: "next_token",
  PARTIAL_OUT: "partial_out",
  TELEMETRY: "telemetry",
  PING: "ping",
  PONG: "pong"
};

export function addLog(line) {
  const el = document.getElementById("log");
  if (el) el.innerHTML += line + "<br>";
  try { console.log(line.replaceAll("<br>","\n")); } catch {}
}

export function bootBanner(who) {
  addLog(`✅ BOOT OK: ${who} [PROTO=${PROTO} | BUILD=${BUILD}]`);
  window.__WIRE_V2_MARKER__ = { PROTO, BUILD };
}

export async function antiCacheEnsureLatest() {
  try {
    const seen = sessionStorage.getItem("__build_seen__");
    if (seen && seen === BUILD) return;
    if ("serviceWorker" in navigator) {
      const regs = await navigator.serviceWorker.getRegistrations();
      for (const r of regs) { try { await r.unregister(); } catch {} }
    }
    if ("caches" in window) {
      const keys = await caches.keys();
      await Promise.all(keys.map(k => caches.delete(k)));
    }
    sessionStorage.setItem("__build_seen__", BUILD);
  } catch {}
}

export function wsURL(roomId, peerId, role) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}?room=${encodeURIComponent(roomId)}&peer=${encodeURIComponent(peerId)}&role=${encodeURIComponent(role)}`;
}
