// Coordinator – Head-Parallel (restored signaling + offers + simple test send)
// Drop-in replacement for the diagnostic stub. Keeps your current HTML IDs.

(function () {
  // ---------- UI grab ----------
  const must = (id) => {
    const el = document.getElementById(id);
    if (!el) throw new Error(`Missing element #${id}`);
    return el;
  };
  const logDiv       = must('log');
  const roomInput    = must('roomInput');
  const startBtn     = must('startBtn');
  const initBtn      = must('initBtn');
  const loadBtn      = must('loadBtn');
  const startDecode  = must('startDecodeBtn');
  const stopDecode   = must('stopDecodeBtn');
  const clearBtn     = must('clearBtn');
  const useReal      = must('useReal');
  const maxTokens    = must('maxTokens');
  const tempInput    = must('temp');
  const toppInput    = must('topp');
  const timeoutMs    = must('timeoutMs');
  const promptInput  = must('promptInput');

  const log = (s) => { logDiv.textContent += s + "\n"; };
  const warn = (s) => { logDiv.textContent += "⚠️ " + s + "\n"; };
  const ok = (s) => { logDiv.textContent += "✅ " + s + "\n"; };

  // ---------- Signaling state ----------
  let ws = null;
  let roomId = 'default';
  let coordId = null;
  const peers = new Map(); // peerId -> { pc, chan, open:boolean }

  // step/test state
  let running = false;
  let stepId = 0;
  let primeCount = 0;
  let timer = null;

  const wsURL = () => {
    const proto = (location.protocol === "https:") ? "wss" : "ws";
    return `${proto}://${location.host}?room=${encodeURIComponent(roomId)}&peer=${encodeURIComponent(coordId)}&role=coord`;
  };

  function safeClose(x){ try { x?.close(); } catch {} }

  // ---------- WebSocket (signaling) ----------
  function startWS(){
    safeClose(ws);
    ws = new WebSocket(wsURL());
    ws.onopen = () => {
      ok(`Signaling connected (${wsURL()})`);
      sendWS({ type:"join", role:"coord", roomId, peerId: coordId });
    };
    ws.onmessage = async (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch { return; }
      // Worker presence via hello (the worker sends periodic "hello")
      if (msg.type === "hello" && msg.role === "worker" && msg.peerId) {
        const pid = msg.peerId;
        if (!peers.has(pid)) {
          log(`worker joined: ${pid}`);
          connectTo(pid);
        }
        return;
      }

      // Standard relay
      if (msg.type === "answer" && msg.to === coordId) {
        const p = peers.get(msg.from);
        if (p && p.pc) {
          await p.pc.setRemoteDescription(msg.sdp);
          ok(`Answer from ${msg.from} applied`);
        }
        return;
      }
      if (msg.type === "ice" && msg.to === coordId) {
        const p = peers.get(msg.from);
        if (p && p.pc) {
          try { await p.pc.addIceCandidate(msg.candidate); } catch {}
        }
        return;
      }
    };
    ws.onclose = () => warn("WS closed (coord)");
    ws.onerror = (e) => warn("WS error (coord): " + (e?.message || "see console"));
  }
  function sendWS(o){ try { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); } catch {} }

  // ---------- WebRTC to a worker ----------
  async function connectTo(peerId){
    if (peers.has(peerId)) return;
    const pc = new RTCPeerConnection({ iceServers:[{urls:"stun:stun.l.google.com:19302"}] });
    const chan = pc.createDataChannel("tiles");
    peers.set(peerId, { pc, chan, open:false });

    chan.onopen = () => {
      peers.get(peerId).open = true;
      ok(`DC → ${peerId} open`);
      // quick liveness test
      try { chan.send(JSON.stringify({ test:"ping", from: coordId })); } catch {}
    };
    chan.onmessage = (e) => {
      // Accept both JSON test replies and any v2 messages (we just log)
      try {
        if (typeof e.data === "string") {
          const m = JSON.parse(e.data);
          if (m.test === "pong") log(`↩️ pong from ${peerId}`);
          if (m.stepId && m.result) {
            log(`result from ${peerId} tile ${m.tileId} step ${m.stepId}: ${String(m.result).slice(0,8)}…`);
          }
          if (m.type === "partial_out") {
            // v2 partials — log-only for now
            log(`partial_out from ${peerId} step ${m.stepId} [${(m.headRange||[]).join('-')}]`);
          }
        }
      } catch {}
    };
    pc.onicecandidate = (e) => { if (e.candidate) sendWS({ type:"ice", to: peerId, from: coordId, candidate: e.candidate }); };
    pc.onconnectionstatechange = () => {
      const st = pc.connectionState;
      if (st === "disconnected" || st === "failed" || st === "closed") {
        warn(`peer ${peerId} ${st} (removed)`);
        try { chan.close(); } catch {}
        try { pc.close(); } catch {}
        peers.delete(peerId);
      }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    sendWS({ type:"offer", to: peerId, from: coordId, sdp: offer });
    log(`Offer → ${peerId}`);
  }

  // ---------- Buttons wiring ----------
  startBtn.addEventListener('click', () => {
    roomId = roomInput.value || "default";
    if (!coordId) coordId = "coord-" + Math.random().toString(36).slice(2);
    ok(`Coordinator ready as ${coordId}`);
    startWS();
  });

  initBtn.addEventListener('click', () => {
    // v2 INIT_MODEL broadcast (workers that speak v1 ignore it harmlessly)
    broadcastJSON({ type:"init_model", modelId:"distilgpt2-layer0", dModel:768, nHeads:12, dHead:64, mlpHidden:3072, nLayers:1, vocab:50257, rotaryDim:0 });
    ok("INIT_MODEL broadcast");
  });

  loadBtn.addEventListener('click', () => {
    const real = !!useReal.checked;
    for (const [pid, p] of peers) {
      if (!p.open) continue;
      if (real) {
        // instruct worker to fetch weight URLs from /assets/weights/*
        const heads = shardHeadsFor(pid);
        const msg = {
          type: "load_shard",
          layer: 0,
          heads,
          weights: {
            qkv: "./assets/weights/qkv.bin",
            o:   "./assets/weights/o.bin",
            ff1: "./assets/weights/ff1.bin",
            ff2: "./assets/weights/ff2.bin",
            wte: "./assets/weights/wte.bin"
          }
        };
        try { p.chan.send(JSON.stringify(msg)); ok(`LOAD_SHARD → ${pid} heads=${heads[0]}..${heads[1]-1} (with urls)`); } catch {}
      } else {
        const heads = shardHeadsFor(pid);
        try { p.chan.send(JSON.stringify({ type:"load_shard", layer:0, heads, synthetic:true })); ok(`LOAD_SHARD → ${pid} heads=${heads[0]}..${heads[1]-1} (synthetic)`); } catch {}
      }
    }
  });

  startDecode.addEventListener('click', () => {
    if (running) return;
    running = true;
    stepId = 0;
    primeCount = 0;
    ok("Start decode");

    // We’ll send a **simple priming loop** so you immediately see activity
    loop();
  });

  stopDecode.addEventListener('click', () => {
    running = false;
    if (timer) { clearTimeout(timer); timer = null; }
    ok("Stop decode");
  });

  clearBtn.addEventListener('click', () => { logDiv.textContent = ""; ok("Log cleared"); });

  // ---------- helpers ----------
  function broadcastJSON(obj){
    for (const [pid, p] of peers) {
      if (p.open) { try { p.chan.send(JSON.stringify(obj)); } catch {} }
    }
  }

  // very simple sharding: split 12 heads evenly among connected peers (2 phones → 0..5 and 6..11)
  function shardHeadsFor(peerId){
    const openPeers = [...peers.keys()].sort();
    const idx = openPeers.indexOf(peerId);
    const n = openPeers.length || 1;
    const totalHeads = 12;
    const per = Math.ceil(totalHeads / n);
    const start = idx * per;
    const end = Math.min(totalHeads, start + per);
    return [start, end]; // [start, end)
  }

  function loop(){
    if (!running) return;

    const prompt = promptInput.value || "<BOS>";
    const tokenToSend = (primeCount < 8)
      ? prompt.charCodeAt(primeCount % prompt.length) || 32
      : Math.floor(Math.random()*100) + 1;

    const openPeers = [...peers.values()].filter(p => p.open);
    if (openPeers.length === 0) {
      warn("No workers connected");
      running = false;
      return;
    }

    for (const p of openPeers) {
      try {
        // v1-compatible test message so older workers respond immediately
        p.chan.send(JSON.stringify({
          type: "ACTV_JSON",
          stepId,
          tileId: 0,
          actBlob: Array.from(new TextEncoder().encode(prompt))
        }));
      } catch {}
    }

    log(`DECODE_STEP → ${openPeers.length} worker(s) (step ${stepId}${primeCount<8?', priming':''})`);
    stepId++;
    primeCount++;

    // pace by timeoutMs
    const ms = Math.max(100, Number(timeoutMs.value) || 1200);
    timer = setTimeout(loop, ms);
  }

  // On load marker
  ok("BOOT OK: coord_hp.js loaded & signaling logic active");
})();
