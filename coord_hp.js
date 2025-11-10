import { MSG } from "./wire_v2.js";
import { loadGPT2Tokenizer } from "./tokenizer_gpt2.js";

const BUILD = "2025-09-09-prod-10";
const PROTO = 2;

const $ = (id) => document.getElementById(id);
const logBox = $("log");
const outBox = $("out");
function log(msg){ const t=new Date().toISOString().split("T")[1]?.replace("Z","")||""; if(logBox){ logBox.textContent += `[${t}] ${msg}\n`; logBox.scrollTop = logBox.scrollHeight; } }
function renderOut(txt){ if(outBox){ outBox.textContent = txt; } else { log("OUT: "+txt); } }

window.addEventListener("error", (e)=>log("❌ JS error: " + (e.message||e)));
window.addEventListener("unhandledrejection", (e)=>log("❌ Promise rejection: " + (e.reason?.message||e.reason)));

let tokenizer=null;
let roomId = $("room")?.value || "default";
let coordId = `coord-${Math.random().toString(36).slice(2)}`;
const peers = new Map();
let ws=null;
let wsHeartbeat=null;

let running=false, step=0, pos=0, maxTokens=64;
let temp=1.0, topP=0.9;

let promptTokens=[];
let lastToken=null;
let assembledText="";

let dims={ dModel:768, nHeads:12, dHead:64, mlpHidden:3072, nLayers:1, vocab:50257, maxSeq:1024 };
let WTE=null, WPE=null, wteReady=false;

function wsURL(){
  const proto = location.protocol==="https:"?"wss":"ws";
  return `${proto}://${location.host}?room=${encodeURIComponent(roomId)}&peer=${encodeURIComponent(coordId)}&role=coord`;
}
function safeSendWS(obj){ try{ ws?.readyState===1 && ws.send(JSON.stringify(obj)); }catch{} }

function softmax_inplace(a) {
  let m = -1e30;
  for (let i = 0; i < a.length; i++) {
    if (a[i] > m) m = a[i];
  }
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    a[i] = Math.exp((a[i] - m) / temp);
    s += a[i];
  }
  s = s || 1;
  for (let i = 0; i < a.length; i++) {
    a[i] /= s;
  }
}

function top_p_sample(p, top) {
  const idx = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
  let cum = 0;
  let cut = idx.length;
  for (let i = 0; i < idx.length; i++) {
    cum += idx[i][0];
    if (cum >= top) {
      cut = i + 1;
      break;
    }
  }
  const kept = idx.slice(0, cut);
  let s = 0;
  for (const item of kept) {
    s += item[0];
  }
  let r = Math.random() * s;
  for (const item of kept) {
    if (r <= item[0]) return item[1];
    r -= item[0];
  }
  return kept[kept.length - 1][1];
}

async function fetchF32(url){ 
  log(`⬇️ Downloading ${url.split('/').pop()}...`);
  const r=await fetch(url,{cache:"force-cache"});
  if(!r.ok) throw new Error("fetch "+url+" "+r.status); 
  const b=await r.arrayBuffer(); 
  log(`✅ Downloaded ${url.split('/').pop()} (${(b.byteLength/1024/1024).toFixed(1)}MB)`);
  return new Float32Array(b); 
}

function sincos_wpe(L,D){ const pe=new Float32Array(L*D); for(let p=0;p<L;p++){ for(let i=0;i<D;i++){ const div=Math.pow(10000,(2*Math.floor(i/2))/D); const v=p/div; pe[p*D+i]=(i%2===0)?Math.sin(v):Math.cos(v); } } return pe; }
function wte_row(out, id){ const D=dims.dModel; const base=id*D; for(let j=0;j<D;j++) out[j]=WTE[base+j]; }
function embed_for_token(id, position){ const D=dims.dModel; const x=new Float32Array(D); wte_row(x,id); if(WPE){ const base=position*D; for(let j=0;j<D;j++) x[j]+=WPE[base+j]; } else { const pe=sincos_wpe(dims.maxSeq,D); const base=position*D; for(let j=0;j<D;j++) x[j]+=pe[base+j]; } return x; }
function logits_from_hidden(h){ const V=dims.vocab,D=dims.dModel; const logits=new Float32Array(V); for(let i=0;i<V;i++){ let acc=0; const base=i*D; for(let j=0;j<D;j++) acc+=WTE[base+j]*h[j]; logits[i]=acc; } return logits; }

function startWS(){
  try{ ws?.close(); }catch{}
  if(wsHeartbeat) { clearInterval(wsHeartbeat); wsHeartbeat=null; }
  
  ws = new WebSocket(wsURL());
  ws.onopen   = ()=>{ 
    log(`✅ Signaling connected (${wsURL()})`); 
    safeSendWS({ type:"join", role:"coord", roomId, peerId:coordId }); 
    
    // HEARTBEAT: Keep connection alive every 10 seconds
    wsHeartbeat = setInterval(() => {
      if (ws?.readyState === 1) {
        safeSendWS({ type:"ping" });
      }
    }, 10000);
  };
  ws.onclose  = ()=>{ log("⚠️ WS closed (coord)"); if(wsHeartbeat) clearInterval(wsHeartbeat); };
  ws.onerror  = (e)=>{ log(`❌ WS error (coord): ${e?.message||""}`); if(wsHeartbeat) clearInterval(wsHeartbeat); };
  ws.onmessage = onSignalMessage;
}

async function onSignalMessage(ev){
  let msg; try{ msg=JSON.parse(ev.data); }catch{ return; }
  if(msg.type==="joined" && msg.role==="worker"){ log(`worker joined: ${msg.peerId}`); createOffer(msg.peerId); return; }
  if(msg.type==="left" && msg.role==="worker"){ log(`worker left: ${msg.peerId}`); cleanupPeer(msg.peerId); return; }
  if(msg.type==="answer" && msg.to===coordId && peers.has(msg.from)){ const p=peers.get(msg.from); try{ await p.pc.setRemoteDescription(msg.sdp); log(`✅ Answer from ${msg.from} applied`); }catch(e){ log(`❌ setRemoteDescription failed: ${e.message||e}`); } return; }
  if(msg.type==="ice" && msg.to===coordId && peers.has(msg.from)){ try{ await peers.get(msg.from).pc.addIceCandidate(msg.candidate); }catch{} return; }
}

function cleanupPeer(peerId){ const p=peers.get(peerId); if(!p) return; try{ p.dc?.close(); }catch{} try{ p.pc?.close(); }catch{} peers.delete(peerId); }

async function createOffer(peerId){
  if(peers.has(peerId)) cleanupPeer(peerId);
  const pc=new RTCPeerConnection({ 
    iceServers:[
      {urls:"stun:stun.l.google.com:19302"},
      {urls:"stun:stun1.l.google.com:19302"},
      {urls:"stun:stun2.l.google.com:19302"},
      {urls:"stun:stun.cloudflare.com:3478"}
    ] 
  });
  const dc=pc.createDataChannel("hp",{ordered:true});
  const state={ pc, dc, ready:false };
  peers.set(peerId,state);
  dc.onopen = ()=>{ log(`🟢 DC → ${peerId} open`); state.ready=true; };
  dc.onclose= ()=>{ log(`⚠️ DC → ${peerId} closed`); state.ready=false; };
  dc.onerror= (e)=>log(`❌ DC error → ${peerId}: ${e?.message||e}`);
  dc.onmessage = onPeerMessage(peerId);
  pc.onicecandidate=(e)=>{ if(e.candidate) safeSendWS({ type:"ice", to:peerId, from:coordId, candidate:e.candidate }); };
  pc.onconnectionstatechange=()=>{ const s=pc.connectionState; if(s==="failed"||s==="closed"||s==="disconnected"){ log(`⚠️ PC ${peerId} ${s}`); state.ready=false; } };
  const offer=await pc.createOffer();
  await pc.setLocalDescription(offer);
  safeSendWS({ type:"offer", to:peerId, from:coordId, sdp:offer });
  log(`Offer → ${peerId}`);
}

function onPeerMessage(peerId){
  return (ev)=>{
    if(typeof ev.data!=="string") return;
    let msg; try{ msg=JSON.parse(ev.data); }catch{ return; }
    log("📩 Coord received: " + msg.type);
    if(msg.type===MSG.SHARD_READY){ log(`✅ shard_ready from ${peerId} heads=${msg.heads[0]}-${msg.heads[1]}`); return; }
    if(msg.type===MSG.STATE_OUT){
      if(!wteReady){ log("⚠️ STATE_OUT received but WTE not ready"); return; }
      const hidden=new Float32Array(msg.hidden);
log("🧮 Hidden state length: " + hidden.length);
const logits=logits_from_hidden(hidden);
log("🎲 Logits length: " + logits.length + " first 5: [" + Array.from(logits.slice(0,5)).join(", ") + "]");
softmax_inplace(logits);
log("✨ After softmax first 5: [" + Array.from(logits.slice(0,5)).join(", ") + "]");
const nextId=top_p_sample(logits, topP);
log("🎯 Sampled token ID: " + nextId);
      const piece=tokenizer?tokenizer.decode([nextId]):"";
log("🔤 Token " + step + ": id=" + nextId + " text='" + piece + "'");
assembledText+=piece;
log("📝 Assembled so far: '" + assembledText + "'");
renderOut(assembledText);
      lastToken=nextId; step++; pos++; sendStep();
      return;
    }
    if(msg.type===MSG.TELEMETRY){ log(`ℹ️ ${peerId}: ${typeof msg.note==="string"?msg.note:JSON.stringify(msg)}`); return; }
  };
}

$("btnStart")?.addEventListener("click", async ()=>{
  roomId = $("room")?.value || "default";
  log("✅ CLICK: Start Coordinator");
  startWS();
  try{
    const man = await (await fetch("./assets/weights/manifest.json",{cache:"no-store"})).json();
    const v = man.tokenizer?.vocab || "./assets/tokenizer/vocab.json";
    const m = man.tokenizer?.merges || "./assets/tokenizer/merges.txt";
    tokenizer = await loadGPT2Tokenizer(v,m);
    log("✅ Tokenizer loaded");
    dims = Object.assign(dims, man.dims || {});
    const expected = 50257 * 768;
    const wteUrl = man.tensors?.wte || "./assets/weights/wte.bin";
    log("⏳ Loading WTE (147MB, may take 1-2 minutes)...");
    WTE = await fetchF32(wteUrl);
    wteReady = !!WTE && WTE.length === expected;
    if(!wteReady){ log(`❌ WTE not ready size=${WTE?.length||0} expected=${expected}`); } else { log("✅ WTE ready on coordinator"); }
    WPE = null;
  }catch(e){ log("❌ Init fetch failed: " + (e.message||e)); wteReady=false; }
});

$("btnInit")?.addEventListener("click", async ()=>{
  log("✅ CLICK: Init Model");
  const promptText = $("prompt")?.value || "";
  promptTokens = tokenizer ? tokenizer.encode(promptText) : [];
  lastToken=null;
  log(`Prompt tokens: ${promptTokens.length}`);
  for (const [_, p] of peers.entries()){ if(p.ready){ p.dc.send(JSON.stringify({ type:MSG.INIT_MODEL, proto:PROTO, build:BUILD })); } }
});

$("btnLoad")?.addEventListener("click", async ()=>{
  log("✅ CLICK: Load Shards");
  const readyPeers=[...peers.entries()].filter(([_,p])=>p.ready);
  if(readyPeers.length===0){ log("⚠️ No ready peers"); return; }
  const man = await (await fetch("./assets/weights/manifest.json",{cache:"no-store"})).json();
  const d = man.dims||{};
  dims = { dModel:d.dModel||768, nHeads:d.nHeads||12, dHead:d.dHead||Math.floor((d.dModel||768)/(d.nHeads||12)), mlpHidden:d.mlpHidden||3072, nLayers:d.nLayers||1, vocab:d.vocab||50257, maxSeq:d.maxSeq||1024 };
  const total=dims.nHeads, per=Math.ceil(total/readyPeers.length);
  let start=0;
  for(const [pid,p] of readyPeers){
    const end=Math.min(total,start+per);
    const weights=Object.assign({}, man.tensors||{});
    delete weights.wte; delete weights.wpe;
    p.dc.send(JSON.stringify({ type:MSG.LOAD_SHARD, layer:0, heads:[start,end], weights, dims }));
    log(`LOAD_SHARD → ${pid} heads=${start}..${end-1}`);
    start=end;
  }
});

$("btnDecode")?.addEventListener("click", ()=>{
  log("✅ CLICK: Start Decode");
  maxTokens = parseInt($("maxtok")?.value||"64",10)||64;
  temp = parseFloat($("temp")?.value||"1.0")||1.0;
  topP = parseFloat($("topp")?.value||"0.9")||0.9;
  if(!wteReady){ log("⚠️ WTE not ready; wait for '✅ WTE ready on coordinator'"); return; }
  if([...peers.values()].filter(p=>p.ready).length===0){ log("⚠️ No workers connected"); return; }
  running=true; step=0; pos=0; assembledText=""; renderOut(assembledText);
  sendStep();
});

$("btnStop")?.addEventListener("click", ()=>{ log("✅ CLICK: Stop Decode"); running=false; });
$("btnClear")?.addEventListener("click", ()=>{ if(logBox) logBox.textContent=""; assembledText=""; renderOut(assembledText); log("🧹 Logs & output cleared"); });

function currentTokenForPos(){ if(pos<promptTokens.length) return promptTokens[pos]; if(lastToken==null) return 50256; return lastToken; }
function sendStep(){
  if(!running) return;
  if(step>=maxTokens){ running=false; log("⏹️ decode finished"); return; }
  const readyPeers=[...peers.entries()].filter(([_,p])=>p.ready);
  if(readyPeers.length===0){ running=false; log("⚠️ No workers connected"); return; }
  if(!wteReady){ running=false; log("❌ sendStep without WTE"); return; }
  const tok=currentTokenForPos();
  const x=embed_for_token(tok,pos);
  const payload=Array.from(x);
  for(const [pid,p] of readyPeers){ try{ p.dc.send(JSON.stringify({ type:MSG.DECODE_STEP, stepId:step, pos, embed:payload })); }catch{} }
  log(`DECODE_STEP → ${readyPeers.length} worker(s) step=${step}`);
}