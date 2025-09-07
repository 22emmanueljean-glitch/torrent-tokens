// Worker — stop WS reconnects while RTC is up; no visibility spam

let decACTV = null, encRESULT = null;
(async () => {
  try { const mod = await import('./wire.js'); decACTV = mod.decACTV; encRESULT = mod.encRESULT; } catch {}
})().catch(()=>{});

const joinBtn = document.getElementById("join");
const reconnectBtn = document.getElementById("reconnect");
const newIdBtn = document.getElementById("newIdReconnect");
const refreshBtn = document.getElementById("refresh");
const hardResetBtn = document.getElementById("hardReset");
const roomInput = document.getElementById("room");
const logDiv = document.getElementById("log");
const wsDot = document.getElementById("wsDot");
const rtcDot = document.getElementById("rtcDot");
const log = (s) => { logDiv.textContent += s + "\n"; };

const setWSDot  = (on)=> wsDot?.classList.toggle("ws",  !!on);
const setRTCDot = (on)=> rtcDot?.classList.toggle("rtc", !!on);

// --- state
let ws=null, pc=null, chan=null;
let peerId=null, roomId=null;
let wantWS=false, keepWS=true;                 // keepWS=false while RTC is open
let helloTimer=null, wsTimer=null;
let isConnectingWS=false;
let reconnectDelay=1000, reconnectDelayMax=20000, retries=0;

// --- helpers
function fnv32(bytes){ let h=0x811c9dc5>>>0,p=0x01000193; for(let i=0;i<bytes.length;i++){ h^=bytes[i]; h=Math.imul(h,p)>>>0; } const out=new Uint8Array(4); new DataView(out.buffer).setUint32(0,h); return out; }
const hexOf = (bytes)=> Array.from(bytes).map(b=>b.toString(16).padStart(2,"0")).join("");
function mulberry32(seed){ let t=seed>>>0; return ()=>{ t+=0x6D2B79F5; let r=Math.imul(t^(t>>>15),1|t); r^=r+Math.imul(r^(r>>>7),61|r); return ((r^(r>>>14))>>>0)/4294967296; }; }
function buildWeightsDeterministic(K, OUT){ const W=new Float32Array(OUT*K); const rnd=mulberry32(0xC0FFEE^(K*1315423911)^(OUT*2654435761)); for(let o=0;o<OUT;o++) for(let i=0;i<K;i++) W[o*K+i]=(rnd()-0.5); return W; }
function bytesToFloatAct(actBytes,K){ const A=new Float32Array(K); const n=actBytes.length; for(let i=0;i<n;i++) A[i]=(actBytes[i]/255)-0.5; for(let i=n;i<K;i++) A[i]=0.0; return A; }
function quantizeUint8(f32){ const out=new Uint8Array(f32.length); const scale=64,eps=1e-7; for(let i=0;i<f32.length;i++){ let q=Math.round((f32[i]+eps)*scale)+128; out[i]=q<0?0:(q>255?255:q);} return out; }
function matmulCPU(W,A,OUT,K){ const C=new Float32Array(OUT); for(let o=0;o<OUT;o++){ let sum=0.0,base=o*K; for(let i=0;i<K;i++) sum+=A[i]*W[base+i]; C[o]=sum; } return C; }

let gpuReady=false, device=null, pipeline=null, bindLayout=null;
let cachedK=0,cachedOUT=0,Wbuf=null,outBuf=null,readBuf=null,paramBuf=null;

async function ensureGPU(){
  if (gpuReady) return;
  if (!navigator.gpu) throw new Error("WebGPU not available");
  const adapter = await navigator.gpu.requestAdapter(); if (!adapter) throw new Error("No GPU adapter");
  device = await adapter.requestDevice();
  const code = /* wgsl */`
    struct Params { K: u32, OUTDIM: u32 };
    @group(0) @binding(0) var<storage, read>      A : array<f32>;
    @group(0) @binding(1) var<storage, read>      W : array<f32>;
    @group(0) @binding(2) var<storage, read_write> C : array<f32>;
    @group(0) @binding(3) var<uniform>            P : Params;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let o = gid.x; if (o >= P.OUTDIM) { return; }
      var sum: f32 = 0.0;
      for (var i: u32 = 0u; i < P.K; i = i + 1u) { sum = sum + A[i] * W[o * P.K + i]; }
      C[o] = sum;
    }
  `;
  pipeline = device.createComputePipeline({ layout:"auto", compute:{ module: device.createShaderModule({ code }), entryPoint:"main" } });
  bindLayout = pipeline.getBindGroupLayout(0);
  gpuReady = true;
}
async function matmulGPU(actBytes){
  await ensureGPU(); const OUT=32; const K=(actBytes.length+3)&~3;
  if (K!==cachedK||OUT!==cachedOUT){
    const W=buildWeightsDeterministic(K,OUT);
    Wbuf?.destroy(); outBuf?.destroy(); readBuf?.destroy(); paramBuf?.destroy();
    Wbuf=device.createBuffer({size:W.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    outBuf=device.createBuffer({size:OUT*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    readBuf=device.createBuffer({size:OUT*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    paramBuf=device.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(Wbuf,0,W); cachedK=K; cachedOUT=OUT;
  }
  const A=bytesToFloatAct(actBytes, ((actBytes.length+3)&~3));
  const Abuf=device.createBuffer({size:A.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(Abuf,0,A);
  device.queue.writeBuffer(paramBuf,0,new Uint32Array([((actBytes.length+3)&~3),32]));
  const bind=device.createBindGroup({ layout:bindLayout, entries:[
    {binding:0,resource:{buffer:Abuf}},
    {binding:1,resource:{buffer:Wbuf}},
    {binding:2,resource:{buffer:outBuf}},
    {binding:3,resource:{buffer:paramBuf}},
  ]});
  const enc=device.createCommandEncoder(); const pass=enc.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0,bind); pass.dispatchWorkgroups(Math.ceil(32/64)); pass.end();
  enc.copyBufferToBuffer(outBuf,0,readBuf,0,32*4); device.queue.submit([enc.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ); const outCopy=readBuf.getMappedRange().slice(0); readBuf.unmap();
  return new Float32Array(outCopy);
}
async function computeResultBytes(actBytes){
  const OUT=32, K=(actBytes.length+3)&~3;
  try { const outF32=await matmulGPU(actBytes); log("🔧 compute mode: GPU"); return fnv32(quantizeUint8(outF32)); }
  catch(e){ log(`🧮 compute mode: CPU (fallback${e?.message?": "+e.message:""})`);
    const W=buildWeightsDeterministic(K,OUT); const A=bytesToFloatAct(actBytes,K);
    return fnv32(quantizeUint8(matmulCPU(W,A,OUT,K)));
  }
}

// --- wake lock
let wakeLock=null;
async function ensureWakeLock(){ try{ if('wakeLock' in navigator && !wakeLock){ wakeLock=await navigator.wakeLock.request('screen'); log('🔒 Wake Lock acquired'); wakeLock.addEventListener('release',()=>log('🔓 Wake Lock released')); } }catch{} }

// --- signaling
const wsURL = ()=> {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const q = `?room=${encodeURIComponent(roomId||"default")}&peer=${encodeURIComponent(peerId||"")}&role=worker`;
  return `${proto}://${location.host}${q}`;
};
const sendWS = (o)=> { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); };

function startHello(){ clearInterval(helloTimer); helloTimer=setInterval(()=>sendWS({type:"hello",role:"worker",peerId}),2000); sendWS({type:"hello",role:"worker",peerId}); }
function stopHello(){ if (helloTimer){ clearInterval(helloTimer); helloTimer=null; } }
function clearWSReconnect(){ if (wsTimer){ clearTimeout(wsTimer); wsTimer=null; } isConnectingWS=false; }
function scheduleWSReconnect(delay=1500){
  clearWSReconnect();
  if (!wantWS || !keepWS) return;                       // <- HARD GATE: don't reconnect WS if RTC is up
  if (document.visibilityState !== 'visible') return;
  wsTimer=setTimeout(connectWS, delay);
}

function connectWS(){
  if (!wantWS || !keepWS) return;                        // <- HARD GATE: only connect WS when we actually want it
  if (isConnectingWS) return;
  isConnectingWS = true;

  try { ws?.close(); } catch {}
  const url = wsURL();
  log(`(dial) ${url}`);
  ws = new WebSocket(url);

  ws.onopen = () => {
    isConnectingWS = false; setWSDot(true);
    retries = 0; reconnectDelay = 1000;
    log(`🔗 WS open`);
    sendWS({ type:"join", role:"worker", roomId, peerId });
    startHello();
  };

  ws.onmessage = async (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }

    if (msg.type === "joined-ack") { log(`✅ joined-ack: ${msg.peerId}`); return; }

    if (msg.type === "offer" && msg.to === peerId) {
      stopHello();
      try { pc?.close(); } catch {}
      pc = new RTCPeerConnection({ iceServers:[{urls:"stun:stun.l.google.com:19302"}] });

      pc.ondatachannel = (ev) => {
        chan = ev.channel; chan.binaryType = "arraybuffer";
        chan.onopen  = () => { log("🟢 DataChannel open"); setRTCDot(true); keepWS=false; /* stop future WS dials */ };
        chan.onclose = () => { log("⚠️ DataChannel closed"); setRTCDot(false); keepWS=true; scheduleWSReconnect(600); };
        chan.onerror = (e) => log("❌ DataChannel error: " + (e?.message || e));
        chan.onmessage = onChannelMessage;
      };
      pc.onicecandidate = (e)=>{ if (e.candidate) sendWS({type:"ice",to:msg.from,from:peerId,candidate:e.candidate}); };
      pc.onconnectionstatechange = ()=>{
        const st = pc.connectionState;
        if (st === "disconnected" || "failed" === st || "closed" === st) {
          setRTCDot(false); keepWS=true; scheduleWSReconnect(500);
        }
      };

      await pc.setRemoteDescription(msg.sdp);
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      sendWS({ type:"answer", to:msg.from, from:peerId, sdp:answer });
      log("Answer sent");
      return;
    }

    if (msg.type === "ice" && msg.to === peerId && pc) { try { await pc.addIceCandidate(msg.candidate); } catch {} }
  };

  ws.onerror = (e)=> log("❌ WS error (worker): " + (e?.message || "see console"));
  ws.onclose = () => {
    isConnectingWS = false; setWSDot(false);
    stopHello(); log("⚠️ WS closed (worker)");
    if (wantWS && keepWS && document.visibilityState === 'visible') {
      retries++; if (retries <= 6) {
        reconnectDelay = Math.min(Math.round(reconnectDelay*1.7)+300, reconnectDelayMax);
        scheduleWSReconnect(reconnectDelay);
        log(`⏳ WS reconnect in ~${Math.round(reconnectDelay/1000)}s (try ${retries}/6)`);
      } else {
        log("🛑 WS retries exhausted — tap Reconnect or Refresh Page");
      }
    }
  };
}

// ONLY try to reconnect on visibility if WS is desired AND RTC is down
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') {
    if (!keepWS) return;                 // <- NEW: if RTC is up, do nothing (no join spam)
    if (!ws || ws.readyState !== 1) { connectWS(); } else { startHello(); }
  }
});

// --- channel handler
async function onChannelMessage(ev){
  if (typeof ev.data === "string") {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.test === "ping") { chan?.send(JSON.stringify({ test:"pong", from:peerId })); }
      else if (msg.type === "ACTV_JSON") {
        const act = new Uint8Array(msg.actBlob || []);
        log(`⬇️ ACTV(JSON) step ${msg.stepId} tile ${msg.tileId} bytes=${act.length}`);
        const resultBytes = await computeResultBytes(act);
        chan?.send(JSON.stringify({ stepId: msg.stepId, tileId: msg.tileId, result: hexOf(resultBytes) }));
        log(`⬆️ RESULT(JSON) step ${msg.stepId} tile ${msg.tileId}`);
      }
    } catch {}
    return;
  }
  if (decACTV && encRESULT) {
    try { const { session_id, step_id, tile_id, body } = decACTV(ev.data);
      log(`⬇️ ACTV(BIN) step ${step_id} tile ${tile_id} bytes=${body.length}`);
      const resultBytes = await computeResultBytes(body);
      const out = encRESULT({ session_id, step_id, tile_id, vote_group: 0, resultBytes });
      chan?.send(out);
      log(`⬆️ RESULT(BIN) step ${step_id} tile ${tile_id}`);
    } catch (e) { log("⚠️ non-ACTV frame or parse error: " + e); }
  }
}

// --- buttons
joinBtn.onclick = async () => {
  await ensureWakeLock();
  roomId = roomInput.value || "default";
  if (!peerId) peerId = "w-" + Math.random().toString(36).slice(2);
  wantWS = true; keepWS = true; setWSDot(false); setRTCDot(false);
  log(`Join requested: room="${roomId}" as ${peerId}`);
  connectWS();
};
reconnectBtn.onclick = () => {
  wantWS=true; keepWS=true;
  log("🔄 Reconnect (same ID)");
  try{ chan?.close(); }catch{}; try{ pc?.close(); }catch{}; try{ ws?.close(); }catch{};
  setRTCDot(false); setWSDot(false); retries=0; reconnectDelay=500;
  scheduleWSReconnect(150);
};
newIdBtn.onclick = () => {
  wantWS=true; keepWS=true; peerId = "w-" + Math.random().toString(36).slice(2);
  log(`🆕 Reconnect (new ID=${peerId})`);
  try{ chan?.close(); }catch{}; try{ pc?.close(); }catch{}; try{ ws?.close(); }catch{};
  setRTCDot(false); setWSDot(false); retries=0; reconnectDelay=500;
  scheduleWSReconnect(150);
};
refreshBtn.onclick = () => location.reload();
hardResetBtn.onclick = () => {
  log("🧹 Hard reset"); wantWS=false; keepWS=false;
  try{ chan?.close(); }catch{}; try{ pc?.close(); }catch{}; try{ ws?.close(); }catch{};
  setRTCDot(false); setWSDot(false);
};
