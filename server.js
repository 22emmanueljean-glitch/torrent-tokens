// server.js — static + WS signaling with join notifications (debounced)
const path = require("path");
const http = require("http");
const express = require("express");
const { WebSocketServer } = require("ws");
const { URL } = require("url");

const app = express();
const PORT = process.env.PORT || 8080;

// serve static from repo root
app.use(express.static(path.join(__dirname)));

const server = http.createServer(app);
const wss = new WebSocketServer({ server });

// roomId -> { worker: Map<peerId, ws>, coord: Map<peerId, ws>, lastJoinMs: Map<peerId, number> }
const rooms = new Map();
function getRoom(name){
  if (!rooms.has(name)) rooms.set(name, { worker:new Map(), coord:new Map(), lastJoinMs:new Map() });
  return rooms.get(name);
}
function sendJSON(ws, o){ try{ ws.send(JSON.stringify(o)); }catch{} }
function routeTo(room, toId, payload){
  const dest = room.worker.get(toId) || room.coord.get(toId);
  if (dest && dest.readyState === dest.OPEN){ sendJSON(dest, payload); return true; }
  return false;
}
function notifyCoords(room, payload){
  for (const ws of room.coord.values()){
    if (ws.readyState === ws.OPEN) sendJSON(ws, payload);
  }
}

wss.on("connection", (ws, req) => {
  const u = new URL(req.url, `http://${req.headers.host}`);
  const roomId = u.searchParams.get("room") || "default";
  const role = (u.searchParams.get("role") || "worker").toLowerCase();
  let peerId = u.searchParams.get("peer") || null;
  const room = getRoom(roomId);

  ws.isAlive = true;
  ws._meta = { roomId, role, peerId };
  ws.on("pong", () => { ws.isAlive = true; });

  ws.on("message", (raw) => {
    let msg = null; try { msg = JSON.parse(raw); } catch { return; }

    if (msg.type === "join") {
      peerId = msg.peerId || peerId || `${role}-${Math.random().toString(36).slice(2)}`;
      ws._meta.peerId = peerId;

      // Replace any old conn for this peer
      const old = room[role].get(peerId);
      if (old && old !== ws) { try{ old.terminate(); }catch{} }
      room[role].set(peerId, ws);

      // Debounce "joined" spam: only notify coords if last join was > 3s ago
      if (role === "worker") {
        const now = Date.now();
        const last = room.lastJoinMs.get(peerId) || 0;
        if (now - last > 3000) {
          room.lastJoinMs.set(peerId, now);
          notifyCoords(room, { type:"joined", peerId });
        }
      }

      sendJSON(ws, { type:"joined-ack", peerId, role, roomId });
      return;
    }

    if (["offer","answer","ice"].includes(msg.type)) {
      msg.roomId = roomId;
      routeTo(room, msg.to, msg);
      return;
    }
  });

  ws.on("close", () => {
    // do not delete lastJoinMs so we can still debounce on immediate reconnects
    const m = room[role];
    const cur = m.get(peerId);
    if (cur === ws) m.delete(peerId);
  });
  ws.on("error", () => {});
});

// server heartbeats so sockets don’t zombie
setInterval(() => {
  wss.clients.forEach((ws) => {
    if (ws.isAlive === false) { try{ ws.terminate(); }catch{}; return; }
    ws.isAlive = false; try { ws.ping(); } catch {}
  });
}, 30000);

server.listen(PORT, () => {
  console.log(`HTTP+WS listening on http://localhost:${PORT}`);
});
