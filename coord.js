const offerIn = document.getElementById('offerIn');
const connectBtn = document.getElementById('connect');
const runBtn = document.getElementById('run');
const out = document.getElementById('out');

const peer = new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'}]});
let chan;

peer.ondatachannel = (ev) => {
  chan = ev.channel;
  chan.onopen = () => out.textContent += 'Worker connected\n';
  chan.onmessage = (ev) => {
    const {tileId, result} = JSON.parse(ev.data);
        out.textContent += `tile ${tileId} result: ${result.slice(0,16)}...\n`;
    };
};

connectBtn.onclick = async () => {
  const offer = JSON.parse(offerIn.value);
  await peer.setRemoteDescription(offer);
  const answer = await peer.createAnswer();
  await peer.setLocalDescription(answer);
  out.textContent = 'Give this answer to worker:\n' + JSON.stringify(answer) + '\n';
};

runBtn.onclick = () => {
  if (!chan) { alert("Connect a worker first"); return; }
  const prompt = "Once upon a time";
  const act = new TextEncoder().encode(prompt);
  chan.send(JSON.stringify({tileId: 0, actBlob: Array.from(act)}));
  out.textContent += `\nPrompt: ${prompt}\n`;
};