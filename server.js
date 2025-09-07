// HTTP static + WebSocket signaling (no-cache for assets)
const path = require("path");
const http = require("http");
const express = require("express");
const { WebSocketServer } = require("ws");
const { URL } = require("url");

const app = express();
const PORT = process.env.PORT || 8080;

// ---- No-cache headers for frontend assets ----
const NO_STORE_EXT = new Set([".html", ".js", ".css", ".wgsl", ".json", ".webmanifest"]);
app.use((req, res, next) => {
  const ext = path.extname(req.path).toLowerCase();
  if (NO_STORE_EXT.has(ext)) {
    res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate");
    res.setHeader("Pragma", "no-cache");
    res.setHeader("Expires", "0");
    res.setHeader("Surrogate-Control", "no-store");
  }
  next();
});

// Static files (from repo root)
app.use(express.static(path.join(__dirname)));

const server = http.createServer(app);

// ---- WebSocket signaling (rooms + relay) ----
const wss = new WebSocketServer({ server });
const rooms = new Map(); // room -> Map(peerId -> ws)

function getRoom(name) {
  if (!rooms.has(name)) rooms.set(name, new Map());
  return rooms.get(name);
}
function putPeer(roomName, peerId, ws) {
  const room = getRoom(roomName);
  const old = room.get(peerId);
  if (old && old !== ws) { try { old.terminate(); } catch {} }
  room.set(peerId, ws);
}
function delPeer(roomName, peerId, ws) {
  const room = rooms.get(roomName);
  if (!room) return;
  const cur = room.get(peerId);
  if (cur === ws) room.delete(peerId);
}
function routeTo(roomName, to, payload) {
  const room = rooms.get(roomName);
  if (!room) return false;
  const ws = room.get(to);
  if (!ws || ws.readyState !== ws.OPEN) return false;
  try { ws.send(JSON.stringify(payload)); } catch {}
  return true;
}

wss.on("connection", (ws, req) => {
  const u = new URL(req.url, `http://${req.headers.host}`);
  const roomId = u.searchParams.get("room") || "default";
  const role = u.searchParams.get("role") || "worker";
  let peerId = u.searchParams.get("peer") || null;

  ws.isAlive = true;
  ws._meta = { roomId, role, peerId };

  ws.on("pong", () => { ws.isAlive = true; });

  ws.on("message", (raw) => {
    let msg = null;
    try { msg = JSON.parse(raw); } catch { return; }

    if (msg.type === "ka" || msg.type === "hello") return;

    if (msg.type === "join") {
      if (!msg.peerId) msg.peerId = "w-" + Math.random().toString(36).slice(2);
      peerId = msg.peerId;
      ws._meta.peerId = peerId;
      putPeer(roomId, peerId, ws);
      return;
    }

    if (msg.type === "offer" || msg.type === "answer" || msg.type === "ice") {
      if (!msg.to || !msg.from) return;
      msg.roomId = roomId;
      routeTo(roomId, msg.to, msg);
      return;
    }
  });

  ws.on("close", () => delPeer(roomId, peerId, ws));
  ws.on("error", () => {});
});

// heartbeat
const interval = setInterval(() => {
  wss.clients.forEach((ws) => {
    if (ws.isAlive === false) { try { ws.terminate(); } catch {}; return; }
    ws.isAlive = false;
    try { ws.ping(); } catch {}
  });
}, 30000);
wss.on("close", () => clearInterval(interval));

server.listen(PORT, () => {
  console.log(`HTTP+WS listening on http://localhost:${PORT}`);
});
