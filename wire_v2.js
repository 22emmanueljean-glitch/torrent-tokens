// MIT – Torrent-Tokens wire v2 (head-parallel)
export const MSG = {
    INIT_MODEL: "init_model",
    LOAD_SHARD: "load_shard",
    SHARD_READY: "shard_ready",
    DECODE_STEP: "decode_step",
    PARTIAL_OUT: "partial_out",
    TELEMETRY: "telemetry",
    ERROR: "error"
  };
  
  export const is = (m, t) => m && m.type === t;
  
  export function encINIT(cfg) {
    return JSON.stringify({ type: MSG.INIT_MODEL, ...cfg });
  }
  
  // heads: [start,end) ; urls: { qkv, o, ff1, ff2 } – array or single url per tensor
  export function encLOAD({ layer, heads, urls=null }) {
    return JSON.stringify({ type: MSG.LOAD_SHARD, layer, heads, urls });
  }
  
  // stepId, tokenId, pos (0-based)
  export function encDECODE({ stepId, tokenId, pos }) {
    return JSON.stringify({ type: MSG.DECODE_STEP, stepId, tokenId, pos });
  }
  