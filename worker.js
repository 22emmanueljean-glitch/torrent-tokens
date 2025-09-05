// MIT – minimal worker for TinyStories-33M 4-bit tiles
const log = document.getElementById('log');
function println(x){ log.innerHTML += x + '<br>'; }

const peer = new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'}]});
const chan = peer.createDataChannel('tiles');

chan.onopen = () => println('🟢 data channel open');
chan.onmessage = async (ev) => {
  const {tileId, actBlob} = JSON.parse(ev.data);
  const result = await computeTile(tileId, new Uint8Array(actBlob));
  chan.send(JSON.stringify({tileId, result}));
};

async function computeTile(tileId, act) {
  // WASM kernel placeholder – returns checksum for now
  const hash = await crypto.subtle.digest('SHA-256', act);
  return Array.from(new Uint8Array(hash)).map(b=>b.toString(16).padStart(2,'0')).join('');
}

join.onclick = async () => {
  const offer = await peer.createOffer();
  await peer.setLocalDescription(offer);
  println('Offer created – paste into coordinator');
  println(JSON.stringify(offer));
};