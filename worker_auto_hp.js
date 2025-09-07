// Worker (head-parallel) – diagnostics + safe reconnects (no cache-busting query needed)

const logDiv = document.getElementById("log");
function log(s, cls="") {
  const d = document.createElement("div");
  if (cls) d.className = cls;
  d.textContent = s;
  logDiv.appendChild(d);
  if (logDiv.childElementCount > 300) logDiv.removeChild(logDiv.firstChild);
}

window.addEventListener("error", (e)=>log(`JS error: ${e.message}`, "err"));
window.addEventListener("unhandledrejection", (e)=>log(`Promise rejection: ${e.reason}`, "err"));

log("✅ BOOT OK: worker_auto_hp.js loaded", "ok");
log(`Origin: ${location.origin}`);
if (location.protocol !== "https:" && location.hostname !== "localhost") {
  log("⚠️ Not HTTPS; WebRTC may be blocked on iOS. Use https://…/worker_auto_hp.html", "warn");
}

// UI
const roomInput     = document.getElementById("room");
const joinBtn       = document.getElementById("join");
const reconnectBtn  = document.getElementById("reconnect");
const hardResetBtn  = document.getElementById("hardReset");
const nukeBtn       = document.getElementById("nuke");

// State
let ws=null, pc=null, chan=null;
let roomId="default";
let peerId=null;
let wantWS=false;
let keepWS=true;
let wsTimer=null;

// helpers
const safeClose = (x, fn="close") => { try { x?.[fn]?.(); } catch {} };
const wsURL = () => `wss://${location.host}?room=${encodeURIComponent(roomId)}&peer=${encodeURIComponent(peerId)}&role=worker`;

function scheduleWSReconnect(ms=1500){
  clearTimeout(wsTimer);
  if (!wantWS || !keepWS) return;
  wsTimer = setTimeout(connectWS, ms);
}

async function nukeCaches() {
  try {
    if ('serviceWorker' in navigator) {
      const regs = await navigator.serviceWorker.getRegistrations();
      for (const r of regs) { try { await r.unregister(); } catch {} }
    }
    if (window.caches) {
      const names = await caches.keys();
      for (const n of names) { try { await caches.delete(n); } catch {} }
    }
    log("🧨 Cache nuked — reloading…", "warn");
    setTimeout(()=>location.reload(), 200);
  } catch (e) {
    log("Cache nuke error: "+(e?.message||e), "err");
  }
}
nukeBtn.onclick = nukeCaches;

// signaling
function connectWS(){
  safeClose(ws);
  const url = wsURL();
  log(`🔌 WS connect → ${url}`);
  try { ws = new WebSocket(url); } catch (e) { log("WS ctor error: " + (e?.message||e), "err"); scheduleWSReconnect(2500); return; }

  ws.onopen = () => {
    log("🔗 WS open", "ok");
    ws.send(JSON.stringify({ type:"join", role:"worker", roomId, peerId }));
    try { ws.send(JSON.stringify({ type:"hello", role:"worker", peerId })); } catch {}
  };

  ws.onmessage = async (ev) => {
    let msg=null; try { msg = JSON.parse(ev.data); } catch {}
    if (!msg) return;

    if (msg.type === "offer" && msg.to === peerId) {
      log(`📨 Offer from ${msg.from}`);
      safeClose(pc);
      pc = new RTCPeerConnection({ iceServers: [{ urls: "stun:stun.l.google.com:19302"}] });

      pc.ondatachannel = (ev) => {
        chan = ev.channel;
        chan.binaryType = "arraybuffer";
        chan.onopen  = () => { log("🟢 DataChannel open", "ok"); keepWS = false; };
        chan.onclose = () => { log("⚠️ DataChannel closed", "warn"); keepWS = true; scheduleWSReconnect(500); };
        chan.onerror = (e) => log("DC error: " + (e?.message||e), "err");

        chan.onmessage = (e) => {
          try {
            const m = typeof e.data === "string" ? JSON.parse(e.data) : null;
            if (m?.test === "ping") {
              chan?.send(JSON.stringify({ test: "pong", from: peerId }));
              log("↩️ pong", "ok");
            }
          } catch {}
        };
      };

      pc.onicecandidate = (e) => { if (e.candidate) ws?.send(JSON.stringify({ type:"ice", to: msg.from, from: peerId, candidate: e.candidate })); };
      pc.onconnectionstatechange = () => log(`PC state: ${pc.connectionState}`);

      await pc.setRemoteDescription(msg.sdp);
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      ws.send(JSON.stringify({ type:"answer", to: msg.from, from: peerId, sdp: answer }));
      log("✅ Answer sent", "ok");
      return;
    }

    if (msg.type === "ice" && msg.to === peerId && pc) {
      try { await pc.addIceCandidate(msg.candidate); } catch (e) { log("ICE add error: "+(e?.message||e), "err"); }
    }
  };

  ws.onerror = (e) => log("WS error: " + (e?.message || "see console"), "err");
  ws.onclose = () => { log("⚠️ WS closed", "warn"); scheduleWSReconnect(1500); };
}

// buttons
joinBtn.onclick = () => {
  roomId = roomInput.value || "default";
  if (!peerId) peerId = "w-" + Math.random().toString(36).slice(2);
  wantWS = true;
  keepWS = true;
  log(`Join requested: room="${roomId}" as ${peerId}`);
  connectWS();
};
reconnectBtn.onclick = () => {
  wantWS = true; keepWS = true;
  log("🔄 Reconnect (same ID)");
  safeClose(chan); safeClose(pc); safeClose(ws);
  setTimeout(connectWS, 200);
};
hardResetBtn.onclick = () => {
  wantWS = false; keepWS = false;
  log("🧹 Hard reset (stop auto-rejoin)");
  peerId = null;
  safeClose(chan); safeClose(pc); safeClose(ws);
};
