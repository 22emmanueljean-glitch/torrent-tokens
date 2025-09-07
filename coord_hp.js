// DIAGNOSTIC COORDINATOR SCRIPT — just wires buttons and logs.
// If this doesn't log, the HTML isn't loading the script (path/cache/CSP).

(function () {
  const must = (id) => {
    const el = document.getElementById(id);
    if (!el) throw new Error(`Missing element #${id}`);
    return el;
  };

  const logDiv = must('log');
  const log = (msg) => { logDiv.textContent += msg + "\n"; };

  // Collect all required IDs up front — fail fast if any mismatch
  const ids = [
    'roomInput','startBtn','initBtn','loadBtn','startDecodeBtn',
    'stopDecodeBtn','clearBtn','useReal','maxTokens','temp',
    'topp','timeoutMs','promptInput'
  ];
  const els = {};
  try {
    ids.forEach(id => els[id] = must(id));
  } catch (e) {
    // Show a big red error on the page if an element is missing
    logDiv.style.color = '#ff5c5c';
    log(`❌ UI wiring error: ${e.message}`);
    return;
  }

  // Wire handlers (pure logging)
  els.startBtn.addEventListener('click', () => log('✅ CLICK: Start Coordinator'));
  els.initBtn.addEventListener('click',  () => log('✅ CLICK: Init Model'));
  els.loadBtn.addEventListener('click',  () => log('✅ CLICK: Load Shards'));
  els.startDecodeBtn.addEventListener('click', () => log('✅ CLICK: Start Decode'));
  els.stopDecodeBtn.addEventListener('click', () => log('✅ CLICK: Stop Decode'));
  els.clearBtn.addEventListener('click', () => { logDiv.textContent = ''; log('🧹 Log cleared'); });

  // Extra input logs
  els.useReal.addEventListener('change', () => log(`☑️ Use real weights: ${els.useReal.checked}`));
  ['maxTokens','temp','topp','timeoutMs','roomInput','promptInput'].forEach(id => {
    els[id].addEventListener('input', () => log(`✏️ ${id} = ${els[id].value}`));
  });

  // signal script is actually loaded
  log('🚀 BOOT OK: coord_hp.js loaded, handlers attached.');
})();

// surface any runtime errors into the log area too
window.addEventListener('error', (e) => {
  const logDiv = document.getElementById('log');
  if (logDiv) {
    logDiv.style.color = '#ff5c5c';
    logDiv.textContent += `\n❌ JS ERROR: ${e.message}\n`;
  }
});
