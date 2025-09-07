// Coordinator — auto-offer + visible "Commit" progress
const logDiv = document.getElementById("log");
const log = (s)=>{ logDiv.textContent += s + "\n"; };

const roomInput = document.getElementById("room");
const startBtn = document.getElementById("start");
const runBtn = document.getElementById("run");
const reofferBtn = document.getElementById("reoffer");
const resetBtn = document.getElementById("reset");
const clearBtn = document.getElementById("clear");

let ws=null, roomId="default", coordId=null;
const peers = new Map();        // peerId -> { pc, chan }
const inflight = new Map();     // stepId -> { expected, got, results: [{peerId, tileId, hash}] }

const wsURL = ()=> {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}?room=${encodeURIComponent(roomId)}&peer=${encodeURIComponent(coordId)}&role=coord`;
};

function safeClose(x){ try{x?.close();}catch{} }

function renderCommit(stepId){
  const s = inflight.get(stepId);
  if (!s) return;
  const line = `Step ${stepId} — ${s.got}/${s.expected} results`;
  // live progress line:
  const id = `commit-${stepId}`;
  let el = document.getElementById(id);
  if (!el) {
    el = document.createElement('div');
    el.id = id;
    el.style.margin = '6px 0';
    logDiv.parentNode.insertBefore(el, logDiv);
  }
  el.textContent = line;

  if (s.got === s.expected) {
    el.textContent = `✅ Commit: step ${stepId} — ${s.got}/${s.expected} results`;
    el.style.color = '#22c55e';
  }
}

function startWS(){
  safeClose(ws);
  ws = new WebSocket(wsURL());
  ws.onopen = ()=>{
    log(`Signaling connected (${wsURL()})`);
    ws.send(JSON.stringify({ type:"join", role:"coord", roomId, peerId: coordId }));
  };
  ws.onmessage = async (ev)=>{
    let msg; try{ msg=JSON.parse(ev.data);}catch{}
    if (!msg) return;

    if (msg.type === "joined") {
      log(`worker joined: ${msg.peerId}`);
      connectTo(msg.peerId);
      return;
    }
    if (msg.type === "answer" && msg.to === coordId) {
      const p = peers.get(msg.from); if (!p) return;
      await p.pc.setRemoteDescription(msg.sdp);
      log(`Answer from ${msg.from} applied`);
      return;
    }
    if (msg.type === "ice" && msg.to === coordId) {
      const p = peers.get(msg.from); if (!p) return;
      try{ await p.pc.addIceCandidate(msg.candidate); }catch{}
      return;
    }
  };
  ws.onclose = ()=> log("⚠️ WS closed (coord)");
  ws.onerror = (e)=> log("❌ WS error (coord): "+(e?.message||""));
}

async function connectTo(peerId){
  if (peers.has(peerId)) return;
  const pc = new RTCPeerConnection({ iceServers:[{urls:"stun:stun.l.google.com:19302"}] });
  const chan = pc.createDataChannel("tiles");
  chan.onopen = ()=> log(`🟢 DataChannel → ${peerId} open`);
  chan.onmessage = (e)=>{
    try{
      const m = JSON.parse(e.data);
      if (m.result) {
        // record result for commit display
        const step = inflight.get(m.stepId);
        if (step && !step.seen?.has(`${m.stepId}:${m.tileId}:${peerId}`)) {
          step.seen.add(`${m.stepId}:${m.tileId}:${peerId}`);
          step.got++;
          step.results.push({ peerId, tileId: m.tileId, hash: m.result });
          renderCommit(m.stepId);
        }
        log(`result from ${peerId} tile ${m.tileId} step ${m.stepId}: ${m.result.slice(0,8)}…`);
      }
    }catch{}
  };
  pc.onicecandidate = (e)=>{ if (e.candidate) ws?.send(JSON.stringify({type:"ice",to:peerId,from:coordId,candidate:e.candidate})); };
  pc.onconnectionstatechange = ()=>{
    const st = pc.connectionState;
    if (st === "disconnected" || st === "failed" || st === "closed"){
      peers.delete(peerId);
      log(`⚠️ peer ${peerId} ${st} (removed)`);
    }
  };
  peers.set(peerId, { pc, chan });

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  ws?.send(JSON.stringify({ type:"offer", to:peerId, from:coordId, sdp:offer }));
  log(`Offer → ${peerId}`);
}

function runTest(){
  const openPeers = [...peers.values()].filter(p => p.chan?.readyState === "open");
  if (!openPeers.length){ log("No workers connected"); return; }
  const stepId = Date.now();
  const act = new TextEncoder().encode("Once upon a time");

  const s = { expected: openPeers.length, got: 0, results: [], seen: new Set() };
  inflight.set(stepId, s);
  renderCommit(stepId);

  let i=0;
  for (const p of openPeers){
    p.chan.send(JSON.stringify({ type:"ACTV_JSON", stepId, tileId:i++, actBlob: Array.from(act) }));
  }
  log(`Broadcast ACTV to ${openPeers.length} worker(s) (step ${stepId})`);
}

startBtn.onclick = ()=>{
  roomId = roomInput.value || "default";
  if (!coordId) coordId = "coord-"+Math.random().toString(36).slice(2);
  startWS();
  log(`Coordinator ready as ${coordId}`);
};
runBtn.onclick = ()=> runTest();
reofferBtn.onclick = ()=> {
  log("Re-offer all");
  for (const [pid,p] of peers){ try{p.chan?.close();}catch{}; try{p.pc?.close();}catch{} peers.delete(pid); connectTo(pid); }
};
resetBtn.onclick = ()=>{
  log("Reset signaling + peers");
  for (const [pid,p] of peers){ try{p.chan?.close();}catch{}; try{p.pc?.close();}catch{} }
  peers.clear(); startWS();
};
clearBtn.onclick = ()=>{ logDiv.textContent=""; };
