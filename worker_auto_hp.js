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
let W={};
let kv=null;

async function fetchMaybe(url){ try{ const r=await fetch(url,{cache:"no-store"}); if(!r.ok) return null; const b=await r.arrayBuffer(); return new Float32Array(b); }catch{ return null; } }
function zeros(n){ return new Float32Array(n); }
function ones(n){ const a=new Float32Array(n); a.fill(1); return a; }
function gelu(x){ const c=Math.sqrt(2/Math.PI); return 0.5*x*(1+Math.tanh(c*(x+0.044715*x*x*x))); }
function layernorm_inplace(x,g,b){ const d=g.length; let mu=0; for(let i=0;i<d;i++) mu+=x[i]; mu/=d; let vs=0; for(let i=0;i<d;i++){ const t=x[i]-mu; vs+=t*t; } const inv=1/Math.sqrt(vs/d+1e-5); for(let i=0;i<d;i++) x[i]=(x[i]-mu)*inv*g[i]+b[i]; }
function gemv_right_rowmajor(x,M,rows,cols,out){ for(let c=0;c<cols;c++){ let acc=0; for(let r=0;r<rows;r++) acc+=x[r]*M[r*cols+c]; out[c]=acc; } }
function add_inplace(y,b){ for(let i=0;i<y.length;i++) y[i]+=b[i]; }
function add_residual_inplace(y,x){ for(let i=0;i<y.length;i++) y[i]+=x[i]; }
function split_qkv(v,d){ return { q:v.subarray(0,d), k:v.subarray(d,2*d), v:v.subarray(2*d,3*d) }; }

function ensureKV(){ if(kv) return; const H=dims.nHeads,L=dims.maxSeq; kv={K:new Array(H),V:new Array(H),len:0}; for(let h=0;h<H;h++){ kv.K[h]=new Array(L); kv.V[h]=new Array(L); } }
function kv_append(kh,vh){ const t=kv.len; for(let h=0;h<dims.nHeads;h++){ kv.K[h][t]=kh[h]; kv.V[h][t]=vh[h]; } kv.len++; }
function self_attn(q,kf,vf,H,dh,T){ const scale=1/Math.sqrt(dh); const ctx=new Float32Array(H*dh); for(let h=0;h<H;h++){ const qh=q.subarray(h*dh,(h+1)*dh); const scores=new Float32Array(T); for(let t=0;t<T;t++){ let dot=0; const Kt=kf[h][t]; for(let j=0;j<dh;j++) dot+=qh[j]*Kt[j]; scores[t]=dot*scale; } let m=-1e30; for(let i=0;i<scores.length;i++) if(scores[i]>m) m=scores[i]; let s=0; for(let i=0;i<scores.length;i++){ scores[i]=Math.exp(scores[i]-m); s+=scores[i]; } s=s||1; const out=ctx.subarray(h*dh,(h+1)*dh); out.fill(0); for(let t=0;t<T;t++){ const wt=scores[t]/s; const Vt=vf[h][t]; for(let j=0;j<dh;j++) out[j]+=wt*Vt[j]; } } return ctx; }

function forward_from_embed(x){
  const D=dims.dModel,H=dims.nHeads,dh=dims.dHead;
  const x1=x.slice();
  layernorm_inplace(x1,W.ln1_g,W.ln1_b);
  const qkv=new Float32Array(3*D);
  gemv_right_rowmajor(x1,W.qkv,D,3*D,qkv);
  add_inplace(qkv,W.qkv_b);
  const s=split_qkv(qkv,D);
  const qH=new Array(H),kH=new Array(H),vH=new Array(H);
  for(let h=0;h<H;h++){
    qH[h]=s.q.subarray(h*dh,(h+1)*dh);
    kH[h]=s.k.subarray(h*dh,(h+1)*dh);
    vH[h]=s.v.subarray(h*dh,(h+1)*dh);
  }
  ensureKV();
  kv_append(kH,vH);
  const ctx=self_attn(s.q,kv.K,kv.V,H,dh,kv.len);
  const aOut=new Float32Array(D);
  gemv_right_rowmajor(ctx,W.o,D,D,aOut);
  add_inplace(aOut,W.o_b);
  add_residual_inplace(aOut,x);
  const x2=aOut.slice();
  layernorm_inplace(x2,W.ln2_g,W.ln2_b);
  const ff=new Float32Array(dims.mlpHidden);
  gemv_right_rowmajor(x2,W.ff1,D,dims.mlpHidden,ff);
  add_inplace(ff,W.ff1_b);
  for(let i=0;i<ff.length;i++) ff[i]=gelu(ff[i]);
  const mOut=new Float32Array(D);
  gemv_right_rowmajor(ff,W.ff2,dims.mlpHidden,D,mOut);
  add_inplace(mOut,W.ff2_b);
  add_residual_inplace(mOut,aOut);
  return mOut;
}

function fillMissing(){ const D=dims.dModel,M=dims.mlpHidden; if(!W.ln1_g) W.ln1_g=ones(D); if(!W.ln1_b) W.ln1_b=zeros(D); if(!W.qkv_b) W.qkv_b=zeros(3*D); if(!W.o_b) W.o_b=zeros(D); if(!W.ln2_g) W.ln2_g=ones(D); if(!W.ln2_b) W.ln2_b=zeros(D); if(!W.ff1_b) W.ff1_b=zeros(M); if(!W.ff2_b) W.ff2_b=zeros(D); }

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
hardResetBtn.onclick = () => { log("🧹 Hard reset"); peerId=null; try{ chan?.close(); }catch{} try{ pc?.close(); }catch{} try{ ws?.close(); }catch{} };
nukeBtn.onclick = async () => { log("💥 Nuke Cache requested"); try{ if("serviceWorker" in navigator){ const regs=await navigator.serviceWorker.getRegistrations(); for(const r of regs){ try{ await r.unregister(); }catch{} } } if("caches" in window){ const keys=await caches.keys(); await Promise.all(keys.map(k=>caches.delete(k))); } location.reload(); }catch{} };

async function onChanMessage(e){
  log("📨 Raw message received, type: " + typeof e.data);
  if(typeof e.data !== "string") {
    log("⚠️ Non-string message ignored");
    return;
  }
  let msg; 
  try{ 
    msg=JSON.parse(e.data); 
    log("📩 Parsed message type: " + (msg?.type || "unknown"));
  }catch(err){ 
    log("❌ JSON parse failed: " + err.message);
    return; 
  }

  if (msg.type===MSG.PING){ chan?.send(JSON.stringify({type:MSG.PONG})); return; }

  if (msg.type===MSG.INIT_MODEL){
    kv=null;
    return;
  }

  if (msg.type===MSG.LOAD_SHARD){
    dims = msg.dims || { dModel:768,nHeads:12,dHead:64,mlpHidden:3072,nLayers:1,vocab:50257,maxSeq:1024 };
    const T = msg.weights || {};
    const req = k => (T && typeof T[k]==="string") ? T[k] : null;

    W.qkv = await fetchMaybe(req("qkv"));
    W.o   = await fetchMaybe(req("o"));
    W.ff1 = await fetchMaybe(req("ff1"));
    W.ff2 = await fetchMaybe(req("ff2"));

    W.ln1_g = await fetchMaybe(req("ln1_g"));
    W.ln1_b = await fetchMaybe(req("ln1_b"));
    W.qkv_b = await fetchMaybe(req("qkv_b"));
    W.o_b   = await fetchMaybe(req("o_b"));
    W.ln2_g = await fetchMaybe(req("ln2_g"));
    W.ln2_b = await fetchMaybe(req("ln2_b"));
    W.ff1_b = await fetchMaybe(req("ff1_b"));
    W.ff2_b = await fetchMaybe(req("ff2_b"));

    fillMissing();

    const have = { qkv: !!W.qkv, o: !!W.o, ff1: !!W.ff1, ff2: !!W.ff2 };
    chan?.send(JSON.stringify({ type: MSG.TELEMETRY, note: "weights", have, dims }));

    if(!have.qkv || !have.o || !have.ff1 || !have.ff2){
      chan?.send(JSON.stringify({ type: MSG.TELEMETRY, note: "missing required tensors" }));
      return;
    }

    chan?.send(JSON.stringify({ type: MSG.SHARD_READY, heads: msg.heads || [0,0] }));
    return;
  }

  if (msg.type===MSG.DECODE_STEP){
    if(!dims || !W.qkv || !W.o || !W.ff1 || !W.ff2) return;
    const emb = Array.isArray(msg.embed) ? new Float32Array(msg.embed) : null;
    if(!emb) return;
    if(kv==null) ensureKV();
    const h = forward_from_embed(emb);
    chan?.send(JSON.stringify({ type: MSG.STATE_OUT, stepId: msg.stepId, hidden: Array.from(h) }));
    return;
  }
}