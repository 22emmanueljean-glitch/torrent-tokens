import { PROTO, BUILD, MSG, addLog, wsURL } from "./wire_v2.js";

const roomInput = document.getElementById("room");
const joinBtn = document.getElementById("join");
const reconnectBtn = document.getElementById("reconnect");
const hardResetBtn = document.getElementById("hardReset");
const nukeBtn = document.getElementById("nuke");

let ws=null, pc=null, chan=null;
let roomId="default", peerId=null;

function log(s){ addLog(s); }

let dims=null;
let W=[]; // Array of 6 layers
let kvCaches = Array(6).fill(null); // Separate KV cache per layer

async function fetchMaybe(url){ try{ const r=await fetch(url,{cache:"force-cache"}); if(!r.ok) return null; const b=await r.arrayBuffer(); return new Float32Array(b); }catch{ return null; } }
function zeros(n){ return new Float32Array(n); }
function ones(n){ const a=new Float32Array(n); a.fill(1); return a; }
function gelu(x){ const c=Math.sqrt(2/Math.PI); return 0.5*x*(1+Math.tanh(c*(x+0.044715*x*x*x))); }
function layernorm_inplace(x,g,b){ const d=g.length; let mu=0; for(let i=0;i<d;i++) mu+=x[i]; mu/=d; let vs=0; for(let i=0;i<d;i++){ const t=x[i]-mu; vs+=t*t; } const inv=1/Math.sqrt(vs/d+1e-5); for(let i=0;i<d;i++) x[i]=(x[i]-mu)*inv*g[i]+b[i]; }
function gemv_right_rowmajor(x,M,rows,cols,out){ for(let c=0;c<cols;c++){ let acc=0; for(let r=0;r<rows;r++) acc+=x[r]*M[r*cols+c]; out[c]=acc; } }
function add_inplace(y,b){ for(let i=0;i<y.length;i++) y[i]+=b[i]; }
function add_residual_inplace(y,x){ for(let i=0;i<y.length;i++) y[i]+=x[i]; }
function split_qkv(v,d){ return { q:v.subarray(0,d), k:v.subarray(d,2*d), v:v.subarray(2*d,3*d) }; }

function ensureKV(layerIdx){ 
  if(kvCaches[layerIdx]) return; 
  const H=dims.nHeads,L=dims.maxSeq; 
  kvCaches[layerIdx]={K:new Array(H),V:new Array(H),len:0}; 
  for(let h=0;h<H;h++){ 
    kvCaches[layerIdx].K[h]=new Array(L); 
    kvCaches[layerIdx].V[h]=new Array(L); 
  } 
}

function kv_append(layerIdx, kh, vh){ 
  const kv = kvCaches[layerIdx];
  const t=kv.len; 
  for(let h=0;h<dims.nHeads;h++){ 
    kv.K[h][t]=kh[h]; 
    kv.V[h][t]=vh[h]; 
  } 
  kv.len++; 
}

function self_attn(layerIdx, q, H, dh){ 
  const kv = kvCaches[layerIdx];
  const T = kv.len;
  const scale=1/Math.sqrt(dh); 
  const ctx=new Float32Array(H*dh); 
  for(let h=0;h<H;h++){ 
    const qh=q.subarray(h*dh,(h+1)*dh); 
    const scores=new Float32Array(T); 
    for(let t=0;t<T;t++){ 
      let dot=0; 
      const Kt=kv.K[h][t]; 
      for(let j=0;j<dh;j++) dot+=qh[j]*Kt[j]; 
      scores[t]=dot*scale; 
    } 
    let m=-1e30; 
    for(let i=0;i<scores.length;i++) if(scores[i]>m) m=scores[i]; 
    let s=0; 
    for(let i=0;i<scores.length;i++){ 
      scores[i]=Math.exp(scores[i]-m); 
      s+=scores[i]; 
    } 
    s=s||1; 
    const out=ctx.subarray(h*dh,(h+1)*dh); 
    out.fill(0); 
    for(let t=0;t<T;t++){ 
      const wt=scores[t]/s; 
      const Vt=kv.V[h][t]; 
      for(let j=0;j<dh;j++) out[j]+=wt*Vt[j]; 
    } 
  } 
  return ctx; 
}

function forward_from_embed(x, layerWeights, layerIdx, appendKV){
  const D=dims.dModel,H=dims.nHeads,dh=dims.dHead;
  const x1=x.slice();
  layernorm_inplace(x1,layerWeights.ln1_g,layerWeights.ln1_b);
  const qkv=new Float32Array(3*D);
  gemv_right_rowmajor(x1,layerWeights.qkv,D,3*D,qkv);
  add_inplace(qkv,layerWeights.qkv_b);
  const s=split_qkv(qkv,D);
  const qH=new Array(H),kH=new Array(H),vH=new Array(H);
  for(let h=0;h<H;h++){
    qH[h]=s.q.subarray(h*dh,(h+1)*dh);
    kH[h]=s.k.subarray(h*dh,(h+1)*dh);
    vH[h]=s.v.subarray(h*dh,(h+1)*dh);
  }
  ensureKV(layerIdx);
  if(appendKV) kv_append(layerIdx, kH, vH);
  const ctx=self_attn(layerIdx, s.q, H, dh);
  const aOut=new Float32Array(D);
  gemv_right_rowmajor(ctx,layerWeights.o,D,D,aOut);
  add_inplace(aOut,layerWeights.o_b);
  add_residual_inplace(aOut,x);
  const x2=aOut.slice();
  layernorm_inplace(x2,layerWeights.ln2_g,layerWeights.ln2_b);
  const ff=new Float32Array(dims.mlpHidden);
  gemv_right_rowmajor(x2,layerWeights.ff1,D,dims.mlpHidden,ff);
  add_inplace(ff,layerWeights.ff1_b);
  for(let i=0;i<ff.length;i++) ff[i]=gelu(ff[i]);
  const mOut=new Float32Array(D);
  gemv_right_rowmajor(ff,layerWeights.ff2,dims.mlpHidden,D,mOut);
  add_inplace(mOut,layerWeights.ff2_b);
  add_residual_inplace(mOut,aOut);
  return mOut;
}

function fillMissing(layerWeights){ const D=dims.dModel,M=dims.mlpHidden; if(!layerWeights.ln1_g) layerWeights.ln1_g=ones(D); if(!layerWeights.ln1_b) layerWeights.ln1_b=zeros(D); if(!layerWeights.qkv_b) layerWeights.qkv_b=zeros(3*D); if(!layerWeights.o_b) layerWeights.o_b=zeros(D); if(!layerWeights.ln2_g) layerWeights.ln2_g=ones(D); if(!layerWeights.ln2_b) layerWeights.ln2_b=zeros(D); if(!layerWeights.ff1_b) layerWeights.ff1_b=zeros(M); if(!layerWeights.ff2_b) layerWeights.ff2_b=zeros(D); }

function connectWS(){
  try{ ws?.close(); }catch{}
  const url = wsURL(roomId, peerId, "worker");
  log("🔌 WS connect → " + url);
  ws = new WebSocket(url);
  ws.onopen = () => { log("🔗 WS open"); ws.send(JSON.stringify({ type:"join", role:"worker", roomId, peerId })); ws.send(JSON.stringify({ type:MSG.HELLO, role:"worker", proto:PROTO, build:BUILD })); };
  ws.onmessage = async (ev) => {
    let m; try{ m=JSON.parse(ev.data); }catch{ return; }
    if (m.type==="offer" && m.to===peerId){
      try{ pc?.close(); }catch{}
      pc = new RTCPeerConnection({ 
        iceServers:[
          {urls:"stun:stun.l.google.com:19302"},
          {urls:"stun:stun1.l.google.com:19302"},
          {urls:"stun:stun2.l.google.com:19302"},
          {urls:"stun:stun.cloudflare.com:3478"}
        ] 
      });
      pc.ondatachannel = (e) => { chan=e.channel; chan.onopen=()=>log("✅ DC open"); chan.onmessage=onChanMessage; chan.onclose=()=>log("⚠️ DC closed"); chan.onerror=(e)=>log("❌ DC error: "+(e?.message||e)); };
      pc.onicecandidate = (e)=>{ if(e.candidate) ws?.send(JSON.stringify({type:"ice",to:m.from,from:peerId,candidate:e.candidate})); };
      pc.onconnectionstatechange = () => { const s=pc.connectionState; if(s==="disconnected"||s==="failed"||s==="closed") log("⚠️ RTCPeerConnection "+s); };
      await pc.setRemoteDescription(m.sdp);
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      ws?.send(JSON.stringify({ type:"answer", to:m.from, from:peerId, sdp:answer }));
      log("Answer sent");
      return;
    }
    if (m.type==="ice" && m.to===peerId && pc){ try{ await pc.addIceCandidate(m.candidate); }catch{} return; }
  };
  ws.onclose = () => log("⚠️ WS closed (worker)");
  ws.onerror = (e) => log("❌ WS error (worker): " + (e?.message||""));
}

joinBtn.onclick = async () => {
  try{ if("wakeLock" in navigator){ const wl=await navigator.wakeLock.request("screen"); wl.addEventListener("release",()=>log("🔓 Wake Lock released")); log("🔒 Wake Lock acquired"); } }catch{}
  roomId = (roomInput.value || "default").trim();
  if(!peerId) peerId = "w-" + Math.random().toString(36).slice(2);
  log('Join requested: room="'+roomId+'" as '+peerId);
  connectWS();
};
reconnectBtn.onclick = () => { log("↻ Reconnect requested"); try{ chan?.close(); }catch{} try{ pc?.close(); }catch{} try{ ws?.close(); }catch{} setTimeout(connectWS,200); };
hardResetBtn.onclick = () => { log("🧹 Hard reset"); peerId=null; W=[]; kvCaches=Array(6).fill(null); try{ chan?.close(); }catch{} try{ pc?.close(); }catch{} try{ ws?.close(); }catch{} };
nukeBtn.onclick = async () => { log("💥 Nuke Cache requested"); try{ if("serviceWorker" in navigator){ const regs=await navigator.serviceWorker.getRegistrations(); for(const r of regs){ try{ await r.unregister(); }catch{} } } if("caches" in window){ const keys=await caches.keys(); await Promise.all(keys.map(k=>caches.delete(k))); } location.reload(); }catch{} };

async function onChanMessage(e){
  if(typeof e.data !== "string") return;
  let msg; 
  try{ msg=JSON.parse(e.data); }catch{ return; }

  if (msg.type===MSG.PING){ chan?.send(JSON.stringify({type:MSG.PONG})); return; }

  if (msg.type===MSG.INIT_MODEL){
    kvCaches = Array(6).fill(null);
    W=[]; 
    return;
  }

  if (msg.type===MSG.LOAD_SHARD){
    const layerIdx = msg.layer || 0;
    dims = msg.dims || { dModel:768,nHeads:12,dHead:64,mlpHidden:3072,nLayers:6,vocab:50257,maxSeq:1024 };
    const T = msg.weights || {};
    log("📥 LOAD_SHARD layer=" + layerIdx);
    
    if(!W[layerIdx]) W[layerIdx] = {};
    const layerW = W[layerIdx];
    
    const req = k => (T && typeof T[k]==="string") ? T[k] : null;

    log("⬇️ Downloading layer " + layerIdx + " weights...");
    layerW.qkv = await fetchMaybe(req("qkv"));
    layerW.o   = await fetchMaybe(req("o"));
    layerW.ff1 = await fetchMaybe(req("ff1"));
    layerW.ff2 = await fetchMaybe(req("ff2"));
    layerW.ln1_g = await fetchMaybe(req("ln1_g"));
    layerW.ln1_b = await fetchMaybe(req("ln1_b"));
    layerW.qkv_b = await fetchMaybe(req("qkv_b"));
    layerW.o_b   = await fetchMaybe(req("o_b"));
    layerW.ln2_g = await fetchMaybe(req("ln2_g"));
    layerW.ln2_b = await fetchMaybe(req("ln2_b"));
    layerW.ff1_b = await fetchMaybe(req("ff1_b"));
    layerW.ff2_b = await fetchMaybe(req("ff2_b"));

    fillMissing(layerW);

    if(!layerW.qkv || !layerW.o || !layerW.ff1 || !layerW.ff2){
      log("❌ Layer " + layerIdx + " missing tensors!");
      return;
    }
    
    log("✅ Layer " + layerIdx + " loaded");
    chan?.send(JSON.stringify({ type: MSG.SHARD_READY, layer: layerIdx, heads: msg.heads || [0,12] }));
    return;
  }

  if (msg.type===MSG.DECODE_STEP){
    const layerIdx = typeof msg.layer === 'number' ? msg.layer : 0;
    
    if(!dims || !W[layerIdx]) {
      log("❌ Layer " + layerIdx + " not loaded!");
      return;
    }
    
    const layerW = W[layerIdx];
    if(!layerW.qkv || !layerW.o || !layerW.ff1 || !layerW.ff2) {
      log("❌ Layer " + layerIdx + " incomplete!");
      return;
    }
    
    const emb = Array.isArray(msg.embed) ? new Float32Array(msg.embed) : null;
    if(!emb) {
      log("❌ No embed!");
      return;
    }
    
    ensureKV(layerIdx);
    
    const h = forward_from_embed(emb, layerW, layerIdx, layerIdx === 0);
    chan?.send(JSON.stringify({ type: MSG.STATE_OUT, stepId: msg.stepId, hidden: Array.from(h) }));
    return;
  }
}