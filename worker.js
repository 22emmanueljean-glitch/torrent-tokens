const logDiv = document.getElementById("log");
const joinBtn = document.getElementById("join");
const applyBtn = document.getElementById("apply");
const hostBlob = document.getElementById("hostBlob");

function log(s){ logDiv.innerHTML += s + "<br>"; }

let pc, chan;

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

async function computeTile(tileId, act){
  // TODO: replace with your real WGSL/WASM matmul
  const hash = await crypto.subtle.digest('SHA-256', act);
  return Array.from(new Uint8Array(hash)).map(b=>b.toString(16).padStart(2,'0')).join('');
}

joinBtn.onclick = async () => {
  pc = new RTCPeerConnection({ iceServers:[{urls:'stun:stun.l.google.com:19302'}] });
  pc.ondatachannel = (ev) => {
    chan = ev.channel;
    chan.onopen = () => log('🟢 DataChannel open');
    chan.onmessage = async (ev2) => {
      const { stepId, tileId, actBlob } = JSON.parse(ev2.data);
      const act = new Uint8Array(actBlob);
      const result = await computeTile(tileId, act);
      chan.send(JSON.stringify({ stepId, tileId, result }));
    };
  };
  pc.onicecandidate = (e) => { if (e.candidate) log("Joiner ICE: " + JSON.stringify(e.candidate)); };
  log("Worker ready. Paste HOST blob and click Apply.");
};

applyBtn.onclick = async () => {
  if (!pc) { alert("Click Join Swarm first"); return; }
  const raw = hostBlob.value;
  const { sdp, ice } = extractJSONObjects(raw);
  if (!sdp) { alert("No SDP offer found in pasted text"); return; }
  await pc.setRemoteDescription(sdp);
  for(const c of ice){ try{ await pc.addIceCandidate(c); }catch(_){/* ignore */} }
  const answer = await pc.createAnswer();
  await pc.setLocalDescription(answer);
  log("=== JOINER ANSWER ===");
  log(JSON.stringify(answer));
  log("(Paste the answer back to the laptop; ICE lines will appear below if needed)");
};
