// MIT – minimal WebRTC data-channel for Torrent-Tokens
const log = document.getElementById('log');
function println(x){ log.textContent += x + '\n'; }

let pc = null;
let chan = null;

document.getElementById('host').onclick = async () => {
  pc = new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'}]});
  chan = pc.createDataChannel('tiles');
  chan.onopen = () => println('🟢 data channel open (host)');
  chan.onmessage = (e) => println('Host received: ' + e.data);
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  println('=== HOST OFFER ===\n' + JSON.stringify(offer));
  pc.onicecandidate = ({candidate}) => {
    if (candidate) println('Host ICE: ' + JSON.stringify(candidate));
  };
};

document.getElementById('join').onclick = async () => {
  const offerStr = prompt('Paste host offer:');
  const offer = JSON.parse(offerStr);
  pc = new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'}]});
  pc.ondatachannel = (ev) => {
    chan = ev.channel;
    chan.onopen = () => println('🟢 data channel open (joiner)');
    chan.onmessage = (e) => println('Joiner received: ' + e.data);
  };
  await pc.setRemoteDescription(offer);
  const answer = await pc.createAnswer();
  await pc.setLocalDescription(answer);
  println('=== JOINER ANSWER ===\n' + JSON.stringify(answer));
  pc.onicecandidate = ({candidate}) => {
    if (candidate) println('Joiner ICE: ' + JSON.stringify(candidate));
  };
};