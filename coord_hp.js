// Coordinator – Head-Parallel (real weights by default, proper logits + top-p)
// Drop this in place of your current coord_hp.js

const logDiv = document.getElementById("log");
const log = (s) => { logDiv.innerHTML += s + "\n"; logDiv.scrollTop = logDiv.scrollHeight; };

const roomInput = document.getElementById("room");
const startBtn  = document.getElementById("start");
const initBtn   = document.getElementById("initModel");
const loadBtn   = document.getElementById("loadShards");
const runBtn    = document.getElementById("startDecode");
const stopBtn   = document.getElementById("stopDecode");
const clearBtn  = document.getElementById("clear");
const promptEl  = document.getElementById("prompt");
const maxTokEl  = document.getElementById("maxTokens");
const tempEl    = document.getElementById("temp");
const toppEl    = document.getElementById("topp");
const timeoutEl = document.getElementById("timeoutMs");
const useRealEl = document.getElementById("useReal");    // checkbox

// Defaults that actually work
useRealEl.checked = true;                   // real weights ON by default
if (!timeoutEl.value) timeoutEl.value = 5000;  // 5s default

let ws = null;
let roomId = "default";
let coordId = null;

const peers = new Map();  // peerId -> { pc, chan, heads:[start,end] }
let stepId = 0;
let running = false;

// Model dims (filled at init from manifest or defaults)
let dModel = 768, nHeads = 12, dHead = 64, vocab = 50257;

// WTE for logits (Float32Array of size vocab*dModel, row-major [vocab, dModel])
let WTE = null;

// -------------- signaling -----------------
const wsURL = () => {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}?room=${encodeURIComponent(roomId)}&peer=${encodeURIComponent(coordId)}&role=coord`;
};
function safeClose(x){ try { x?.close(); } catch {} }

function startWS(){
  safeClose(ws);
  ws = new WebSocket(wsURL());
  ws.onopen = () => {
    log(`Signaling connected (${wsURL()})`);
    ws.send(JSON.stringify({ type:"join", role:"coord", roomId, peerId: coordId }));
  };
  ws.onmessage = async (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch {}
    if (!msg) return;

    if (msg.type === "answer" && msg.to === coordId) {
      const p = peers.get(msg.from);
      if (!p) return;
      await p.pc.setRemoteDescription(msg.sdp);
      log(`Answer from ${msg.from} applied`);
      return;
    }
    if (msg.type === "ice" && msg.to === coordId) {
      const p = peers.get(msg.from);
      if (!p) return;
      try { await p.pc.addIceCandidate(msg.candidate); } catch {}
      return;
    }
    if (msg.type === "joined" && msg.role === "worker" && msg.peerId) {
      // optional broadcast from server; not required
      log(`worker joined: ${msg.peerId}`);
    }
  };
  ws.onclose = () => log("⚠️ WS closed (coord)");
  ws.onerror = (e) => log("❌ WS error (coord): " + (e?.message || "see console"));
}

async function connectTo(peerId){
  if (peers.has(peerId)) return;
  const pc = new RTCPeerConnection({ iceServers:[{urls:"stun:stun.l.google.com:19302"}] });
  const chan = pc.createDataChannel("tiles");
  chan.onopen = () => log(`🟢 DC → ${peerId} open`);
  chan.onmessage = (e) => onWorkerMessage(peerId, e);
  pc.onicecandidate = (e) => { if (e.candidate) ws?.send(JSON.stringify({ type:"ice", to: peerId, from: coordId, candidate: e.candidate })); };
  pc.onconnectionstatechange = () => {
    const st = pc.connectionState;
    if (st === "disconnected" || st === "failed" || st === "closed") {
      peers.delete(peerId);
      log(`⚠️ peer ${peerId} ${st} (removed)`);
    }
  };
  peers.set(peerId, { pc, chan, heads:null });

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  ws?.send(JSON.stringify({ type:"offer", to: peerId, from: coordId, sdp: offer }));
  log(`Offer → ${peerId}`);
}

// Discover workers by asking server for all in room (simple approach: rely on join clicks)
// Provide manual connect:
window.addEventListener("message", (ev) => {
  // not used; kept for extensibility
});

// -------------- wire v2 shapes -----------------
const MSG = {
  INIT_MODEL: "init_model",
  LOAD_SHARD: "load_shard",
  SHARD_READY:"shard_ready",
  DECODE_STEP:"decode_step",
  PARTIAL_OUT:"partial_out",
  TELEMETRY:  "telemetry",
};

// -------------- worker message handling -----------------
const pendingPartials = new Map(); // stepId -> {needed: Set(peerIds), parts: []}

function onWorkerMessage(peerId, e){
  if (typeof e.data === "string"){
    let m; try{ m = JSON.parse(e.data); } catch { return; }
    if (m.type === MSG.SHARD_READY){
      peers.get(peerId).heads = m.headRange; // [start,end)
      log(`✅ shard_ready from ${peerId} heads=${m.headRange[0]}-${m.headRange[1]}`);
      return;
    }
    if (m.type === MSG.PARTIAL_OUT){
      const ps = pendingPartials.get(m.stepId);
      if (!ps) return;
      ps.parts.push({ y: new Float32Array(m.yBuffer), headRange: m.headRange, from: peerId });
      ps.needed.delete(peerId);
      const got = Array.from(ps.parts).length;
      const need = ps.needed.size;
      log(`Step ${m.stepId}: ${got}/${got+need} (gen)`);
      if (need === 0){
        pendingPartials.delete(m.stepId);
        onAllPartials(m.stepId, ps.parts);
      }
      return;
    }
    if (m.type === MSG.TELEMETRY){
      // optional GPU/CPU report
      return;
    }
  }
}

// -------------- model init + assets -----------------
async function fetchJSON(url){
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url} ${r.status}`);
  return r.json();
}
async function fetchBIN(url){
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url} ${r.status}`);
  const buf = await r.arrayBuffer();
  return new Float32Array(buf);
}

async function loadManifest(){
  const man = await fetchJSON(`/assets/weights/manifest.json`);
  dModel = man.dims.dModel;
  nHeads = man.dims.nHeads;
  dHead  = man.dims.dHead;
  vocab  = man.dims.vocab;
  if (!man.tensors?.wte) throw new Error("manifest missing tensors.wte path");
  // absolute to avoid relative confusion
  const wteUrl = man.tensors.wte.startsWith("/")
    ? man.tensors.wte
    : `/` + man.tensors.wte.replace(/^\.?\/*/, "");
  log("Loading WTE (for logits)…");
  const arr = await fetchBIN(wteUrl);
  if (arr.length !== vocab * dModel) {
    log(`⚠️ WTE size mismatch: got ${arr.length}, expected ${vocab*dModel}`);
  }
  WTE = arr; // row-major [vocab, dModel]
  log("Tokenizer: loaded • Weights: manifest loaded");
  return { man, wteUrl };
}

// -------------- sampling -----------------
function softmaxTopP(logits, temp=1.0, topP=0.9){
  // logits: Float32Array length vocab
  // temperature
  const invT = 1.0 / Math.max(1e-6, temp);
  let maxLogit = -1e30;
  for (let i=0;i<logits.length;i++){ const v = logits[i]*invT; if (v>maxLogit) maxLogit=v; logits[i]=v; }
  // exp + sum
  let sum = 0.0;
  for (let i=0;i<logits.length;i++){ const e = Math.exp(logits[i]-maxLogit); logits[i]=e; sum+=e; }
  // sort indices by prob desc (avoid copying big array if you want; simple path here)
  const probs = logits; // reuse
  for (let i=0;i<probs.length;i++) probs[i] = probs[i]/sum;

  const idx = Array.from({length: probs.length}, (_,i)=>i);
  idx.sort((a,b)=>probs[b]-probs[a]);

  // cumulative until topP
  let cum = 0.0;
  let cutoff = 0;
  while (cutoff < idx.length && cum + probs[idx[cutoff]] < topP) {
    cum += probs[idx[cutoff]];
    cutoff++;
  }
  const pool = idx.slice(0, Math.max(1, cutoff+1));
  // sample from pool
  let poolSum = 0;
  for (const i of pool) poolSum += probs[i];
  let r = Math.random() * poolSum;
  for (const i of pool){
    const p = probs[i];
    if (r <= p) return i;
    r -= p;
  }
  return pool[pool.length-1];
}

// -------------- merge & logits -----------------
function mergePartials(parts){
  // Each part.y is a Float32Array length dModel contribution from some head range that
  // your worker computed. v0: just sum them (assuming workers returned full dModel slice).
  const y = new Float32Array(dModel);
  for (const p of parts){ const v = p.y; for (let i=0;i<dModel;i++) y[i]+=v[i]; }
  return y;
}

function logitsFromY(y){
  // y [dModel]; WTE [vocab, dModel] row-major — logits[i] = dot(y, WTE[i,:])
  const logits = new Float32Array(vocab);
  let off = 0;
  for (let i=0;i<vocab;i++){
    let s = 0.0;
    for (let j=0;j<dModel;j++) s += y[j] * WTE[off + j];
    logits[i] = s;
    off += dModel;
  }
  return logits;
}

// -------------- decode loop -----------------
function broadcast(type, body){
  const msg = JSON.stringify({ type, ...body });
  for (const [pid,p] of peers){
    try { p.chan?.send(msg); } catch {}
  }
}

async function initModel(){
  // Load manifest + WTE (for logits)
  await loadManifest();

  // Ask all connected workers to announce; coordinator builds offers lazily
  // (You already click Start Coordinator; this only sends a broadcast init)
  const cfg = {
    modelId: "distilgpt2-layer0",
    dModel, nHeads, dHead, mlpHidden: 3072, nLayers: 1, vocab
  };
  broadcast(MSG.INIT_MODEL, cfg);
  log("INIT_MODEL broadcast");
}

async function loadShards(){
  // Split heads across workers and send either real weight URLs or "synthetic" flag
  const plist = [...peers.entries()].filter(([_,p])=>p.chan && p.chan.readyState==="open");
  if (plist.length === 0){ log("No workers connected"); return; }

  // head ranges
  const per = Math.ceil(nHeads / plist.length);
  const man = await fetchJSON(`/assets/weights/manifest.json`);
  const base = (s) => s.startsWith("/") ? s : "/" + s.replace(/^\.?\/*/, "");

  let idx=0;
  for (const [pid,p] of plist){
    const start = idx*per;
    const end   = Math.min(nHeads, start+per);
    idx++;
    if (start>=end) { p.heads=[start,end]; continue; }

    if (useRealEl.checked){
      // send URLs
      const payload = {
        type: MSG.LOAD_SHARD,
        layer: 0,
        heads: [start, end],
        urls: {
          qkv: base(man.tensors.qkv),
          o:   base(man.tensors.o),
          ff1: base(man.tensors.ff1),
          ff2: base(man.tensors.ff2),
        }
      };
      p.chan.send(JSON.stringify(payload));
      log(`LOAD_SHARD → ${pid} heads=${start}..${end-1} (with urls)`);
    } else {
      // synthetic weights
      const payload = { type: MSG.LOAD_SHARD, layer: 0, heads: [start,end], synthetic: true };
      p.chan.send(JSON.stringify(payload));
      log(`LOAD_SHARD → ${pid} heads=${start}..${end-1} (synthetic)`);
    }
  }
}

function startDecode(){
  const plist = [...peers.entries()].filter(([_,p])=>p.chan && p.chan.readyState==="open");
  if (plist.length === 0){ log("No workers connected"); return; }
  if (!WTE){ log("⚠️ WTE not loaded; click Init Model first"); return; }

  running = true;
  const text = (promptEl.value || "<BOS>").trim();
  // prime with UTF-8 bytes as stub tokenizer (your real tokenizer is already loaded,
  // but if your v2 path still uses byte-priming, keep it for now)
  const enc = new TextEncoder().encode(text);
  stepId = 0;
  // feed a few byte tokens to populate KV on workers
  const primeSteps = Math.min(enc.length, 16);
  for (let i=0;i<primeSteps;i++){
    decodeStep(enc[i], true);
  }
  // then start generation (we’ll call decodeStep again from onAllPartials)
  decodeStep(1, false); // 1 ~ BOS-ish
}

function stopDecode(){ running = false; }

function decodeStep(tokenId, priming){
  const plist = [...peers.entries()].filter(([_,p])=>p.chan && p.chan.readyState==="open");
  const need = new Set(plist.map(([pid])=>pid));
  pendingPartials.set(stepId, { needed: need, parts: [] });

  const msg = { type: MSG.DECODE_STEP, stepId, tokenId, pos: stepId };
  for (const [pid,p] of plist){
    try { p.chan.send(JSON.stringify(msg)); } catch {}
  }
  const label = priming ? "priming" : "gen";
  log(`DECODE_STEP → ${plist.length} workers (step ${stepId}, token ${tokenId}${priming?", priming":""})`);

  // timeout
  const tmo = parseInt(timeoutEl.value || "5000", 10);
  const thisStep = stepId;
  setTimeout(()=>{
    const ps = pendingPartials.get(thisStep);
    if (!ps) return;
    const got = ps.parts.length, needLeft = ps.needed.size;
    log(`⏱️ step ${thisStep} timeout — got ${got}/${got+needLeft}`);
    // proceed with what we have if any
    pendingPartials.delete(thisStep);
    if (got>0) onAllPartials(thisStep, ps.parts);
    else if (running) {
      // advance with same token to avoid deadlock
      decodeStep(tokenId, false);
    }
  }, tmo);

  stepId++;
}

function onAllPartials(id, parts){
  // Merge head outputs into y
  const y = mergePartials(parts);

  // Compute logits and sample next token
  const logits = logitsFromY(y);
  const temp = parseFloat(tempEl.value || "0.8");
  const tp   = parseFloat(toppEl.value || "0.9");
  const nextTok = softmaxTopP(logits, temp, tp);

  // Append to UI
  try {
    const dec = new TextDecoder();
    const ch = nextTok < 256 ? String.fromCharCode(nextTok) : "";
    logDiv.innerHTML += (ch || "□");
  } catch {}

  if (!running) return;

  // Continue decoding until max tokens
  const maxT = parseInt(maxTokEl.value || "64", 10);
  if (id >= maxT) {
    running = false;
    return;
  }
  decodeStep(nextTok, false);
}

// -------------- UI buttons -----------------
startBtn.onclick = () => {
  roomId = roomInput.value || "default";
  if (!coordId) coordId = "coord-" + Math.random().toString(36).slice(2);
  startWS();
  log(`Coordinator ready as ${coordId}`);
};

initBtn.onclick = () => { initModel().catch(e=>log("❌ init: "+e.message)); };
loadBtn.onclick = () => { loadShards().catch(e=>log("❌ load: "+e.message)); };
runBtn.onclick  = () => { running=true; startDecode(); };
stopBtn.onclick = () => stopDecode();
clearBtn.onclick= () => { logDiv.textContent=""; };
