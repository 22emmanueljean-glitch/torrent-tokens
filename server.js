// HTTP static + WebSocket signaling with heartbeats + "joined" broadcast
const path = require("path");
const http = require("http");
const express = require("express");
const { WebSocketServer } = require("ws");
const { URL } = require("url");

const app = express();
const PORT = process.env.PORT || 8080;

// serve the repo dir (index.html, worker_auto_hp.html, assets, etc.)
app.use(express.static(path.join(__dirname)));

const server = http.createServer(app);
const wss = new WebSocketServer({ server });

// roomName -> Map(peerId -> { ws, role })
const rooms = new Map();

function getRoom(name) {
  if (!rooms.has(name)) rooms.set(name, new Map());
  return rooms.get(name);
}

function putPeer(roomName, peerId, role, ws) {
  const room = getRoom(roomName);
  const prev = room.get(peerId);
  if (prev && prev.ws !== ws) {
    try { prev.ws.terminate(); } catch {}
  }
  room.set(peerId, { ws, role });
}

function delPeer(roomName, peerId, ws) {
  const room = rooms.get(roomName);
  if (!room) return;
  const cur = room.get(peerId);
  if (cur && cur.ws === ws) room.delete(peerId);
}

function broadcastToCoords(roomName, payloadObj) {
  const room = rooms.get(roomName);
  if (!room) return;
  for (const [_, entry] of room) {
    if (entry.role === "coord" && entry.ws.readyState === entry.ws.OPEN) {
      try { entry.ws.send(JSON.stringify(payloadObj)); } catch {}
    }
  }
}

function routeTo(roomName, toPeerId, payloadObj) {
  const room = rooms.get(roomName);
  if (!room) return false;
  const entry = room.get(toPeerId);
  if (!entry || entry.ws.readyState !== entry.ws.OPEN) return false;
  try { entry.ws.send(JSON.stringify(payloadObj)); } catch {}
  return true;
}

wss.on("connection", (ws, req) => {
  const u = new URL(req.url, `http://${req.headers.host}`);
  const roomId = u.searchParams.get("room") || "default";
  let role = u.searchParams.get("role") || "worker";
  let peerId = u.searchParams.get("peer") || null;

  ws.isAlive = true;
  ws._meta = { roomId, role, peerId };

  ws.on("pong", () => { ws.isAlive = true; });

  ws.on("message", (raw) => {
    let msg = null;
    try { msg = JSON.parse(raw); } catch { return; }

    // keep-alives
    if (msg.type === "ka" || msg.type === "hello") return;

    if (msg.type === "join") {
      // allow client to omit peerId; we’ll assign
      peerId = msg.peerId || (role === "coord" ? ("coord-" + Math.random().toString(36).slice(2))
                                                : ("w-" + Math.random().toString(36).slice(2)));
      role = msg.role || role;
      ws._meta.peerId = peerId;
      ws._meta.role = role;
      putPeer(roomId, peerId, role, ws);

      // tell all coordinators someone joined
      broadcastToCoords(roomId, { type: "joined", roomId, peerId, role });

      // optional: ack to the joiner
      try { ws.send(JSON.stringify({ type: "joined_ack", roomId, peerId, role })); } catch {}

      return;
    }

    // relay SDP/ICE
    if (msg.type === "offer" || msg.type === "answer" || msg.type === "ice") {
      if (!msg.to || !msg.from) return;
      msg.roomId = roomId;
      routeTo(roomId, msg.to, msg);
      return;
    }
  });

  ws.on("close", () => {
    delPeer(roomId, peerId, ws);
    // inform coordinators a peer left (optional but useful)
    if (peerId) broadcastToCoords(roomId, { type: "left", roomId, peerId, role });
  });

  ws.on("error", () => {});
});

// heartbeat pings
const interval = setInterval(() => {
  wss.clients.forEach((ws) => {
    if (ws.isAlive === false) {
      try { ws.terminate(); } catch {}
      return;
    }
    ws.isAlive = false;
    try { ws.ping(); } catch {}
  });
}, 30000);

wss.on("close", () => clearInterval(interval));

server.listen(PORT, '0.0.0.0', () => {
  console.log(`HTTP+WS listening on http://localhost:${PORT}`);
});
