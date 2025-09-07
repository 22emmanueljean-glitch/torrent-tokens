const out = document.getElementById("out");
const startBtn = document.getElementById("start");
const applyBtn = document.getElementById("apply");
const copyOfferBtn = document.getElementById("copyOffer");
const runBtn = document.getElementById("run");
const joinerBlob = document.getElementById("joinerBlob");

let pc;
const dcs = new Set();         // data channels for all connected workers
let lastOfferText = "";

function log(s){ out.textContent += s + "\n"; }

function extractJSONObjects(text){
  const res = { sdp:null, ice:[] };
  const matches = (text || "").match(/\{[\s\S]*?\}/g) || [];
  for(const m of matches){
    try{
      const obj = JSON.parse(m);
      if (obj.sdp && obj.type && !res.sdp) res.sdp = obj;
      else if (obj.candidate) res.ice.push(obj);
    }catch(_){}
  }
  return res;
}

startBtn.onclick = async () => {
  pc = new RTCPeerConnection({ iceServers:[{urls:'stun:stun.l.google.com:19302'}] });

  // every time a worker connects, store its DC
  const dc = pc.createDataChannel("tiles");
  dc.onopen = () => { dcs.add(dc); log("🟢 DataChannel open"); };
  dc.onclose = () => dcs.delete(dc);
  dc.onmessage = onWorkerMessage;

  pc.ondatachannel = (ev) => {
    // if workers initiate DCs, we’d receive them here
    const c = ev.channel;
    c.onopen  = () => { dcs.add(c); log("🟢 DataChannel open"); };
    c.onclose = () => dcs.delete(c);
    c.onmessage = onWorkerMessage;
  };

  pc.onicecandidate = (e) => {
    if (e.candidate) log("Host ICE: " + JSON.stringify(e.candidate));
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  lastOfferText = JSON.stringify(offer);
  copyOfferBtn.disabled = false;
  log("=== HOST OFFER ===");
  log(lastOfferText);
  log("(Copy the offer above to the phone. When the phone prints the JOINER blob, paste it here and click Apply.)");
};

copyOfferBtn.onclick = async () => {
  if (!lastOfferText) return;
  try { await navigator.clipboard.writeText(lastOfferText); } catch {}
  log("Offer copied to clipboard.");
};

applyBtn.onclick = async () => {
  if (!pc) { alert("Click Start Coordinator first"); return; }
  const raw = joinerBlob.value;
  const { sdp, ice } = extractJSONObjects(raw);
  if (!sdp) { alert("No SDP answer found in pasted text"); return; }
  try {
    await pc.setRemoteDescription(sdp);
    for(const c of ice){ try{ await pc.addIceCandidate(c); }catch(_){} }
    log("Applied remote answer + ICE");
  } catch (e) {
    console.error(e);
    alert("Failed to apply answer/ICE. Check console.");
  }
};

// ---- Majority vote (2-of-3) over results ----
let stepId = 0;
const votes = new Map(); // stepId -> Map(resultHex -> count)

function onWorkerMessage(ev){
  try {
    const msg = JSON.parse(ev.data);
    if (msg && typeof msg === 'object') {
      const { stepId, tileId, result } = msg;
      log(`worker → tile ${tileId} step ${stepId} result: ${String(result).slice(0,16)}...`);
      const byStep = votes.get(stepId) || new Map();
      byStep.set(result, (byStep.get(result) || 0) + 1);
      votes.set(stepId, byStep);
      for (const [hex, count] of byStep.entries()) {
        if (count >= 2) {
          log(`✅ COMMIT step ${stepId} with majority result ${hex.slice(0,16)}...`);
        }
      }
      return;
    }
  } catch {}
  log("worker → " + ev.data);
}

runBtn.onclick = () => {
  if (!dcs.size){ alert("No workers connected"); return; }
  const prompt = "Once upon a time";
  const act = new TextEncoder().encode(prompt);
  const payload = { stepId: stepId++, tileId: 0, actBlob: Array.from(act) };

  for (const ch of dcs) {
    if (ch.readyState === "open") ch.send(JSON.stringify(payload));
  }
  log(`Prompt sent to ${dcs.size} worker(s): "${prompt}" (step ${payload.stepId})`);
};
