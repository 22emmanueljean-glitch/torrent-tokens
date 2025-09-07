// Worker – Head-parallel shard loader (uses absolute /assets URLs)
// Drop this in place of your current worker_auto_hp.js

const logDiv = document.getElementById("log");
const log = (s) => { logDiv.innerHTML += s + "\n"; logDiv.scrollTop = logDiv.scrollHeight; };

const joinBtn = document.getElementById("join");
const roomInput = document.getElementById("room");

let ws, pc, chan, peerId, roomId;
let helloTimer = null;

// current shard
let HEADS = [0,0];
let weights = null; // { qkv: Float32Array, o: Float32Array, ff1: Float32Array, ff2: Float32Array }

function wsURL(){ const proto = (location.protocol === "https:") ? "wss" : "ws"; return `${proto}://${location.host}`; }
function sendWS(o){ if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }

function startHello(){
  if (helloTimer) clearInterval(helloTimer);
  helloTimer = setInterval(() => sendWS({ type: "hello", role: "worker", peerId }), 2000);
  sendWS({ type: "hello", role: "worker", peerId });
}
function stopHello(){ if (helloTimer){ clearInterval(helloTimer); helloTimer=null; } }

function connectWS(){
  try { ws?.close(); } catch {}
  ws = new WebSocket(wsURL());
  ws.onopen = () => {
    log(`🔗 WS open (${wsURL()})`);
    sendWS({ type: "join", role: "worker", roomId, peerId });
    startHello();
  };
  ws.onmessage = async (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "offer" && msg.to === peerId) {
      stopHello();
      if (pc) { try { pc.close(); } catch {} }
      pc = new RTCPeerConnection({ iceServers: [{urls: "stun:stun.l.google.com:19302"}] });
      pc.ondatachannel = (ev) => {
        chan = ev.channel;
        chan.binaryType = "arraybuffer";
        chan.onopen = () => log("🟢 DataChannel open");
        chan.onclose = () => log("⚠️ DataChannel closed");
        chan.onmessage = onChannelMessage;
        chan.onerror = (e) => log("❌ DataChannel error: " + (e?.message || e));
      };
      pc.onicecandidate = (e) => { if (e.candidate) sendWS({ type: "ice", to: msg.from, from: peerId, candidate: e.candidate }); };
      pc.onconnectionstatechange = () => {
        const st = pc.connectionState;
        if (st === "disconnected" || st === "failed" || st === "closed") {
          log(`⚠️ PeerConnection ${st} — awaiting new offer`);
          startHello();
        }
      };
      await pc.setRemoteDescription(msg.sdp);
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      sendWS({ type: "answer", to: msg.from, from: peerId, sdp: answer });
      log("Answer sent");
    }
    else if (msg.type === "ice" && msg.to === peerId && pc) {
      try { await pc.addIceCandidate(msg.candidate); } catch {}
    }
  };
  ws.onerror = (e) => log("❌ WS error (worker): " + (e?.message || "see console"));
  ws.onclose = () => { log("⚠️ WS closed (worker)"); stopHello(); };
}

// ----------- math stubs (keep until your WGSL kernels are wired) -----------
function rmsnorm(x){ let s=0; for (let i=0;i<x.length;i++) s+=x[i]*x[i]; const g=1/Math.sqrt(s/x.length+1e-5); for (let i=0;i<x.length;i++) x[i]*=g; return x; }
function mul(a,b,outCols){ const out = new Float32Array(outCols); const cols = b.length/outCols; for(let j=0;j<outCols;j++){ let s=0; for(let i=0;i<cols;i++){ s += a[i]*b[i*outCols+j]; } out[j]=s; } return out; }
function silu(x){ for(let i=0;i<x.length;i++){ const v=x[i]; x[i]=v/(1+Math.exp(-v)); } }

// ----------- channel handler -----------
async function fetchBINabs(url){
  // url may be "/assets/weights/qkv.bin" or "./assets/weights/qkv.bin"
  const abs = url.startsWith("/") ? url : ("/" + url.replace(/^\.?\/*/,""));
  const r = await fetch(abs);
  if (!r.ok) throw new Error(`fetch ${abs} ${r.status}`);
  const buf = await r.arrayBuffer();
  return new Float32Array(buf);
}

async function onChannelMessage(ev){
  if (typeof ev.data !== "string") return;
  let msg; try { msg = JSON.parse(ev.data); } catch { return; }

  if (msg.type === "load_shard"){
    HEADS = msg.heads || [0,0];
    if (msg.synthetic){
      // build deterministic random weights shaped for dModel=768 etc (quick stub)
      // (left as-is from your previous worker; this path is rarely used now)
      weights = { qkv:new Float32Array(768*3*768), o:new Float32Array(768*768),
                  ff1:new Float32Array(768*3072), ff2:new Float32Array(3072*768) };
      chan?.send(JSON.stringify({ type:"shard_ready", headRange: HEADS }));
      log(`shard_ready (synthetic) heads=${HEADS[0]}-${HEADS[1]}`);
      return;
    }
    try{
      const { urls } = msg;
      weights = {
        qkv: await fetchBINabs(urls.qkv),
        o:   await fetchBINabs(urls.o),
        ff1: await fetchBINabs(urls.ff1),
        ff2: await fetchBINabs(urls.ff2),
      };
      chan?.send(JSON.stringify({ type:"shard_ready", headRange: HEADS }));
      log(`shard_ready (real) heads=${HEADS[0]}-${HEADS[1]}`);
    }catch(e){
      chan?.send(JSON.stringify({ type:"error", error: String(e) }));
      log("worker error: " + e);
    }
    return;
  }

  if (msg.type === "decode_step"){
    const { stepId, tokenId } = msg;
    // 1) embed token (super simple byte embed for now)
    const x = new Float32Array(768);
    const base = (tokenId & 0xff) / 255;
    for (let i=0;i<x.length;i++) x[i] = base;

    // 2) RMSNorm
    rmsnorm(x);

    // 3) QKV (use whole matrices for now; true head-slicing comes later)
    const qkv = mul(x, weights.qkv, 3*768);
    // 4) (skip rope & real attn in this stub)
    // 5) O projection
    const attnOut = qkv.subarray(0, 768); // stub: pretend attention produced 768-dim
    const o = mul(attnOut, weights.o, 768);
    // 6) MLP
    let h = mul(o, weights.ff1, 3072); silu(h);
    const y = mul(h, weights.ff2, 768); // final per-layer output contribution

    // Send partial_out (full dModel slice; coordinator will sum)
    const yBuffer = y.buffer.slice(0);
    chan?.send(JSON.stringify({ type:"partial_out", stepId, headRange: HEADS, yBuffer }));
  }
}

// -------------- UI --------------
joinBtn.onclick = () => {
  roomId = roomInput.value || "default";
  if (!peerId) peerId = "w-" + Math.random().toString(36).slice(2);
  log(`Join requested: room="${roomId}" as ${peerId}`);
  connectWS();
};
