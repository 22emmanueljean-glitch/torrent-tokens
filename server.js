// HTTP static + WebSocket signaling with heartbeats + worker discovery
const path = require("path");
const http = require("http");
const express = require("express");
const { WebSocketServer } = require("ws");
const { URL } = require("url");

const app = express();
const PORT = process.env.PORT || 8080;

// serve everything from repo root (adjust if you keep assets elsewhere)
app.use(express.static(path.join(__dirname)));

const server = http.createServer(app);
const wss = new WebSocketServer({ server });

// room -> Map(peerId -> ws)
const rooms = new Map();

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
function broadcastToRole(roomName, role, payload) {
  const room = rooms.get(roomName);
  if (!room) return;
  for (const [, sock] of room) {
    if (sock.readyState !== sock.OPEN) continue;
    if (sock._meta?.role === role) {
      try { sock.send(JSON.stringify(payload)); } catch {}
    }
  }
}

wss.on("connection", (ws, req) => {
  const u = new URL(req.url, `http://${req.headers.host}`);
  const roomId = u.searchParams.get("room") || "default";
  const role   = u.searchParams.get("role") || "worker";
  let peerId   = u.searchParams.get("peer") || null;

  ws.isAlive = true;
  ws._meta = { roomId, role, peerId };

  ws.on("pong", () => { ws.isAlive = true; });

  ws.on("message", (raw) => {
    let msg = null;
    try { msg = JSON.parse(raw); } catch { return; }

    // Heartbeats
    if (msg.type === "ka") return;

    // JOIN: register peer and notify coordinators (discovery)
    if (msg.type === "join") {
      if (!msg.peerId) msg.peerId = (role === "worker" ? "w-" : "coord-") + Math.random().toString(36).slice(2);
      peerId = msg.peerId;
      ws._meta.peerId = peerId;
      putPeer(roomId, peerId, ws);

      // tell all coordinators in the room that someone joined
      broadcastToRole(roomId, "coord", {
        type: "hello",
        role,
        peerId,
        roomId
      });
      return;
    }

    // HELLO: relay to coordinators so they can connect
    if (msg.type === "hello") {
      // ensure we know this socket's peerId (some workers send hello before join)
      if (msg.peerId && !ws._meta.peerId) {
        ws._meta.peerId = msg.peerId;
        putPeer(roomId, msg.peerId, ws);
      }
      broadcastToRole(roomId, "coord", {
        type: "hello",
        role: msg.role || ws._meta.role || role,
        peerId: msg.peerId || ws._meta.peerId,
        roomId
      });
      return;
    }

    // Relay SDP / ICE between peers
    if (msg.type === "offer" || msg.type === "answer" || msg.type === "ice") {
      if (!msg.to || !msg.from) return;
      msg.roomId = roomId;
      routeTo(roomId, msg.to, msg);
      return;
    }
  });

  ws.on("close", () => {
    delPeer(roomId, peerId, ws);
  });
  ws.on("error", () => {});
});

// Heartbeat killer every 30s
const interval = setInterval(() => {
  wss.clients.forEach((ws) => {
    if (ws.isAlive === false) { try { ws.terminate(); } catch {} return; }
    ws.isAlive = false;
    try { ws.ping(); } catch {}
  });
}, 30000);

wss.on("close", () => clearInterval(interval));

server.listen(PORT, () => {
  console.log(`HTTP+WS listening on http://localhost:${PORT}`);
});
