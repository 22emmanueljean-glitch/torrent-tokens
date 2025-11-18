import { MSG } from "./wire_v2.js";
import { loadGPT2Tokenizer } from "./tokenizer_gpt2.js";

const BUILD = "2025-09-09-v14";
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

let dims={ dModel:768, nHeads:12, dHead:64, mlpHidden:3072, nLayers:6, vocab:50257, maxSeq:1024 };
let WTE=null, WPE=null, wteReady=false;
let LN_F_G=null, LN_F_B=null;

// Multi-layer state
let currentLayer = 0;
let hiddenState = null;

function wsURL(){
  const proto = location.protocol==="https:"?"wss":"ws";
  return `${proto}://${location.host}?room=${encodeURIComponent(roomId)}&peer=${encodeURIComponent(coordId)}&role=coord`;
}
function safeSendWS(obj){ try{ ws?.readyState===1 && ws.send(JSON.stringify(obj)); }catch{} }

function softmax_inplace(a, temperature) {
  let m = -1e30;
  for (let i = 0; i < a.length; i++) {
    if (a[i] > m) m = a[i];
  }
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    a[i] = Math.exp((a[i] - m) / temperature);
    s += a[i];
  }
  s = s || 1;
  for (let i = 0; i < a.length; i++) {
    a[i] /= s;
  }
}

function top_p_sample(p, top) {
  // Create array of [probability, index] and sort by probability descending
  const sorted = Array.from(p).map((prob, idx) => [prob, idx]).sort((a, b) => b[0] - a[0]);
  
  // Accumulate probabilities until we reach top_p threshold
  let cumSum = 0;
  let cutoff = 0;
  for (let i = 0; i < sorted.length; i++) {
    cumSum += sorted[i][0];
    if (cumSum >= top) {
      cutoff = i + 1;
      break;
    }
  }
  
  // Sample from top-p tokens
  const candidates = sorted.slice(0, Math.max(1, cutoff));
  const totalProb = candidates.reduce((sum, item) => sum + item[0], 0);
  let r = Math.random() * totalProb;
  
  for (const [prob, idx] of candidates) {
    r -= prob;
    if (r <= 0) return idx;
  }
  
  return candidates[0][1]; // fallback
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
    if(msg.type===MSG.SHARD_READY){ log(`✅ Layer ${msg.layer} ready on ${peerId}`); return; }
    if(msg.type===MSG.STATE_OUT){
      if(!wteReady){ log("⚠️ STATE_OUT received but WTE not ready"); return; }
      hiddenState = new Float32Array(msg.hidden);
      log(`✅ Layer ${currentLayer} complete (step ${step})`);
      
      // Check if we've processed all layers
      currentLayer++;
      if(currentLayer < 6){
        // Send to next layer
        sendToLayer(currentLayer);
      } else {
        // All layers done - sample token
        // Check if we're still processing the prompt
        if(pos < promptTokens.length) {
          log(`✅ Prompt token ${pos} processed`);
          lastToken=null; step++; pos++;
          currentLayer = 0;
          sendStep();
          return;
        }
        
        log(`🎯 All 6 layers complete, sampling token...`);
        // Apply final layer norm
        const normalized = new Float32Array(dims.dModel);
        for(let i=0; i<dims.dModel; i++) normalized[i] = hiddenState[i];
        let mu=0; for(let i=0;i<dims.dModel;i++) mu+=normalized[i]; mu/=dims.dModel;
        let vs=0; for(let i=0;i<dims.dModel;i++){ const t=normalized[i]-mu; vs+=t*t; }
        const inv=1/Math.sqrt(vs+1e-5);
        for(let i=0;i<dims.dModel;i++) normalized[i]=(normalized[i]-mu)*inv*LN_F_G[i]+LN_F_B[i];
        const logits=logits_from_hidden(normalized);
        softmax_inplace(logits, temp);
        const nextId=top_p_sample(logits, topP);
        log(`🎯 Sampled ID: ${nextId} (vocab size: ${logits.length})`);
        const piece=tokenizer?tokenizer.decode([nextId]):"";
        log(`🔤 Token ${step}: "${piece}"`);
        assembledText+=piece;
        renderOut(assembledText);
        lastToken=nextId; step++; pos++;
        
        // Start next token
        currentLayer = 0;
        sendStep();
      }
      return;
    }
    if(msg.type===MSG.TELEMETRY){ log(`ℹ️ ${peerId}: ${typeof msg.note==="string"?msg.note:JSON.stringify(msg)}`); return; }
  };
}

function sendToLayer(layerIdx){
  const readyPeers=[...peers.entries()].filter(([_,p])=>p.ready);
  if(readyPeers.length===0){ running=false; log("⚠️ No workers"); return; }
  
  const payload=Array.from(hiddenState);
  for(const [pid,p] of readyPeers){ 
    try{ p.dc.send(JSON.stringify({ type:MSG.DECODE_STEP, stepId:step, pos, layer:layerIdx, embed:payload })); }catch{} 
  }
}

$("btnStart")?.addEventListener("click", async ()=>{
  roomId = $("room")?.value || "default";
  log("✅ CLICK: Start Coordinator");
  startWS();
  try{
    const man = await (await fetch("./assets/weights/manifest.json",{cache:"no-store"})).json();
log(`🔍 DEBUG: Loaded manifest tokenizer paths: vocab=${man.tokenizer?.vocab}, merges=${man.tokenizer?.merges}`);
const cacheBust = Date.now();
const v = man.tokenizer?.vocab || `./assets/tokenizer/vocab.json?v=${cacheBust}`;
const m = man.tokenizer?.merges || `./assets/tokenizer/merges.txt?v=${cacheBust}`;
tokenizer = await loadGPT2Tokenizer(v, m);
log("✅ Tokenizer loaded");

// DEBUG: Test tokenizer
const testPrompt = 'The capital of France is';
const testTokens = tokenizer.encode(testPrompt);
log(`🔍 Test tokenization: "${testPrompt}" -> [${testTokens.join(', ')}]`);
for(let i=0; i<Math.min(5, testTokens.length); i++) {
  const man = await (await fetch("./assets/weights/manifest.json",{cache:"no-store"})).json();
log(`🔍 DEBUG: Loaded manifest tokenizer paths: vocab=${man.tokenizer?.vocab}, merges=${man.tokenizer?.merges}`);
  log(`  Token ${i}: ${testTokens[i]} = "${tokenizer.decode([testTokens[i]])}"`);
}
    dims = Object.assign(dims, man.dims || {});
    const expected = 50257 * 768;
    const wteUrl = man.tensors?.wte || "./assets/weights/wte.bin";
    log("⏳ Loading WTE (147MB)...");
    WTE = await fetchF32(wteUrl);
wteReady = !!WTE && WTE.length === expected;
if(!wteReady){ log(`❌ WTE not ready`); } else { log("✅ WTE ready"); }

const wpeUrl = man.tensors?.wpe || "./assets/weights/wpe.bin";
WPE = await fetchF32(wpeUrl);
log("✅ WPE loaded");

const lnfgUrl = man.tensors?.ln_f_g || "./assets/weights/ln_f_g.bin";
const lnfbUrl = man.tensors?.ln_f_b || "./assets/weights/ln_f_b.bin";
LN_F_G = await fetchF32(lnfgUrl);
LN_F_B = await fetchF32(lnfbUrl);
log("✅ Final layer norm loaded");
  }catch(e){ log("❌ Init failed: " + (e.message||e)); wteReady=false; }
});

$("btnInit")?.addEventListener("click", async ()=>{
  log("✅ CLICK: Init Model");
  const promptText = $("prompt")?.value || "";
  promptTokens = tokenizer ? tokenizer.encode(promptText) : [];
  lastToken=null;
  log(`Prompt: ${promptTokens.length} tokens`);
  for (const [_, p] of peers.entries()){ if(p.ready){ p.dc.send(JSON.stringify({ type:MSG.INIT_MODEL, proto:PROTO, build:BUILD })); } }
});

$("btnLoad")?.addEventListener("click", async ()=>{
  log("✅ CLICK: Load ALL 6 Layers");
  const readyPeers=[...peers.entries()].filter(([_,p])=>p.ready);
  if(readyPeers.length===0){ log("⚠️ No ready peers"); return; }
  
  // Load all 6 layers
  for(let layerIdx = 0; layerIdx < 6; layerIdx++){
    log(`📥 Loading layer ${layerIdx}...`);
    const man = await (await fetch(`./assets/weights/manifest_layer${layerIdx}.json`,{cache:"no-store"})).json();
    const total=dims.nHeads, per=Math.ceil(total/readyPeers.length);
    let start=0;
    for(const [pid,p] of readyPeers){
      const end=Math.min(total,start+per);
      const weights=Object.assign({}, man.tensors||{});
      delete weights.wte; delete weights.wpe;
      p.dc.send(JSON.stringify({ type:MSG.LOAD_SHARD, layer:layerIdx, heads:[start,end], weights, dims }));
      start=end;
    }
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  log("✅ All 6 layers loaded!");
});

$("btnDecode")?.addEventListener("click", ()=>{
  log("✅ CLICK: Start Decode (6 layers)");
  maxTokens = parseInt($("maxtok")?.value||"64",10)||64;
  temp = parseFloat($("temp")?.value||"1.0")||1.0;
  topP = parseFloat($("topp")?.value||"0.9")||0.9;
  if(!wteReady){ log("⚠️ WTE not ready"); return; }
  if([...peers.values()].filter(p=>p.ready).length===0){ log("⚠️ No workers"); return; }
  running=true; step=0; pos=0; assembledText=""; currentLayer=0;
  renderOut(assembledText);
  sendStep();
});

$("btnStop")?.addEventListener("click", ()=>{ log("✅ CLICK: Stop"); running=false; });
$("btnClear")?.addEventListener("click", ()=>{ if(logBox) logBox.textContent=""; assembledText=""; renderOut(assembledText); log("🧹 Cleared"); });

function currentTokenForPos(){ if(pos<promptTokens.length) return promptTokens[pos]; if(lastToken==null) return 50256; return lastToken; }
function sendStep(){
  if(!running) return;
  if(step>=maxTokens){ running=false; log("⏹️ Finished"); return; }
  const readyPeers=[...peers.entries()].filter(([_,p])=>p.ready);
  if(readyPeers.length===0){ running=false; log("⚠️ No workers"); return; }
  if(!wteReady){ running=false; log("❌ No WTE"); return; }
  
  log(`▶️ Token ${step} starting (layer 0)...`);
  const tok=currentTokenForPos();
  hiddenState=embed_for_token(tok,pos);
  currentLayer = 0;
  
  const payload=Array.from(hiddenState);
  for(const [pid,p] of readyPeers){ 
    try{ p.dc.send(JSON.stringify({ type:MSG.DECODE_STEP, stepId:step, pos, layer:0, embed:payload })); }catch{}
  }
}