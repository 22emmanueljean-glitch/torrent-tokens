// Worker (auto) — sends hello, accepts offers, replies to ACTV_JSON
const joinBtn = document.getElementById("join");
const refreshBtn = document.getElementById("refresh");
const roomInput = document.getElementById("room");
const logDiv = document.getElementById("log");
const log = (s)=>{ logDiv.textContent += s + "\n"; };

let ws=null, pc=null, chan=null;
let peerId=null, roomId="default";
let helloTimer=null;

function wsURL(){
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}?room=${encodeURIComponent(roomId)}&peer=${encodeURIComponent(peerId||"")}&role=worker`;
}
function sendWS(o){ try{ if(ws && ws.readyState===1) ws.send(JSON.stringify(o)); }catch{} }
function startHello(){
  stopHello();
  helloTimer = setInterval(()=> sendWS({ type:"hello", role:"worker", peerId }), 3000);
  sendWS({ type:"hello", role:"worker", peerId });
}
function stopHello(){ if(helloTimer){ clearInterval(helloTimer); helloTimer=null; } }
function safeClose(x, fn="close"){ try{ x?.[fn]?.(); } catch{} }

// simple deterministic hash reply so coord sees results
function fnv32(bytes){
  let h = 0x811c9dc5 >>> 0, p = 0x01000193;
  for (let i=0;i<bytes.length;i++){ h ^= bytes[i]; h = Math.imul(h, p) >>> 0; }
  const out = new Uint8Array(4);
  new DataView(out.buffer).setUint32(0, h);
  return out;
}
const hexOf = (u8)=> Array.from(u8).map(b=>b.toString(16).padStart(2,"0")).join("");

function connectWS(){
  safeClose(ws);
  ws = new WebSocket(wsURL());
  ws.onopen = () => {
    log(`🔗 WS open ${wsURL()}`);
    sendWS({ type:"join", role:"worker", roomId, peerId });
    startHello();
  };
  ws.onmessage = async (ev) => {
    let msg; try{ msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "offer" && msg.to === peerId) {
      stopHello();
      safeClose(pc);
      pc = new RTCPeerConnection({ iceServers:[{urls:"stun:stun.l.google.com:19302"}] });

      pc.ondatachannel = (ev) => {
        chan = ev.channel;
        chan.binaryType = "arraybuffer";
        chan.onopen  = () => log("🟢 DataChannel open");
        chan.onclose = () => log("⚠️ DataChannel closed");
        chan.onerror = (e) => log("❌ DataChannel error: " + (e?.message||e));
        chan.onmessage = (e) => {
          if (typeof e.data === "string") {
            try {
              const m = JSON.parse(e.data);
              if (m.test === "ping") {
                chan.send(JSON.stringify({ test:"pong", from: peerId }));
              } else if (m.type === "ACTV_JSON") {
                const act = new Uint8Array(m.actBlob||[]);
                const res = fnv32(act);
                chan.send(JSON.stringify({ stepId:m.stepId, tileId:m.tileId, result: hexOf(res) }));
              } else if (m.type === "load_shard") {
                // acknowledge shard load (synthetic or urls)
                chan.send(JSON.stringify({ type:"shard_ready", heads: m.heads }));
              }
            } catch {}
          }
        };
      };
      pc.onicecandidate = (e)=>{ if(e.candidate) sendWS({ type:"ice", to: msg.from, from: peerId, candidate: e.candidate }); };
      pc.onconnectionstatechange = ()=>{
        const st = pc.connectionState;
        if (st === "disconnected" || st === "failed" || st === "closed") {
          log(`⚠️ RTCPeer ${st} — waiting for new offer`);
          startHello();
        }
      };
      await pc.setRemoteDescription(msg.sdp);
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      sendWS({ type:"answer", to: msg.from, from: peerId, sdp: answer });
      log("Answer sent");
    }
    if (msg.type === "ice" && msg.to === peerId && pc) {
      try { await pc.addIceCandidate(msg.candidate); } catch {}
    }
  };
  ws.onclose = ()=>{ log("⚠️ WS closed (worker)"); stopHello(); };
  ws.onerror = (e)=> log("❌ WS error: " + (e?.message||"see console"));
}

joinBtn.onclick = ()=>{
  roomId = roomInput.value || "default";
  if (!peerId) peerId = "w-" + Math.random().toString(36).slice(2);
  log(`Join requested: room="${roomId}" as ${peerId}`);
  connectWS();
};
refreshBtn.onclick = ()=> location.reload();
