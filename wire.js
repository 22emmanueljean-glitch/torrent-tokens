// wire.js — minimal TLV binary framing for Torrent-Tokens demo
// Frame (big-endian):
// u8 version | u8 msg_type | u16 header_len | u32 body_len | header | body
// msg_type: 1 = ACTV_MSG, 2 = RESULT_MSG
// ACTV header:  u32 session_id | u32 step_id | u32 tile_id | u8[16] checksum128
// RESULT header: u32 session_id | u32 step_id | u32 tile_id | u8 vote_group | u8[16] checksum128

export const VERSION = 1;
export const MT = { ACTV: 1, RESULT: 2 };

// ---- simple checksum128 (FNV-1a 32 repeated 4x → 16 bytes) ----
export function checksum128(bytes) {
  function fnv32(arr, seed = 0x811c9dc5) {
    let h = seed >>> 0, prime = 0x01000193;
    for (let i = 0; i < arr.length; i++) { h ^= arr[i]; h = Math.imul(h, prime) >>> 0; }
    return h >>> 0;
  }
  const h1 = fnv32(bytes);
  const h2 = fnv32(bytes, 0x9e3779b9);
  const h3 = fnv32(bytes, 0x85ebca6b);
  const h4 = fnv32(bytes, 0xc2b2ae35);
  const buf = new ArrayBuffer(16);
  const dv = new DataView(buf);
  dv.setUint32(0, h1); dv.setUint32(4, h2); dv.setUint32(8, h3); dv.setUint32(12, h4);
  return new Uint8Array(buf);
}

function bePackHeader(prefixLen, bodyLen) {
  const buf = new ArrayBuffer(1 + 1 + 2 + 4 + prefixLen + bodyLen);
  const dv = new DataView(buf);
  dv.setUint8(0, VERSION);
  // msg_type filled later at [1]
  dv.setUint16(2, prefixLen); // header_len
  dv.setUint32(4, bodyLen);   // body_len
  return { buf, dv, off: 8 };
}

// ---- ACTV_MSG ----
export function encACTV({ session_id, step_id, tile_id, actBytes }) {
  const c128 = checksum128(actBytes);
  const headerLen = 4 + 4 + 4 + 16; // sId + step + tile + c128
  const { buf, dv, off } = bePackHeader(headerLen, actBytes.length);
  dv.setUint8(1, MT.ACTV);
  let p = off;
  dv.setUint32(p, session_id); p += 4;
  dv.setUint32(p, step_id);    p += 4;
  dv.setUint32(p, tile_id);    p += 4;
  new Uint8Array(buf, p, 16).set(c128); p += 16;
  new Uint8Array(buf, p, actBytes.length).set(actBytes);
  return buf;
}

export function decACTV(buf) {
  const dv = new DataView(buf);
  const ver = dv.getUint8(0), mt = dv.getUint8(1);
  if (ver !== VERSION || mt !== MT.ACTV) throw new Error("not ACTV");
  const hlen = dv.getUint16(2), blen = dv.getUint32(4);
  let p = 8;
  const session_id = dv.getUint32(p); p += 4;
  const step_id    = dv.getUint32(p); p += 4;
  const tile_id    = dv.getUint32(p); p += 4;
  const c128 = new Uint8Array(buf, p, 16); p += 16;
  const body = new Uint8Array(buf, 8 + hlen, blen);
  return { session_id, step_id, tile_id, c128: new Uint8Array(c128), body };
}

// ---- RESULT_MSG ----
export function encRESULT({ session_id, step_id, tile_id, vote_group = 0, resultBytes }) {
  const c128 = checksum128(resultBytes);
  const headerLen = 4 + 4 + 4 + 1 + 16;
  const { buf, dv, off } = bePackHeader(headerLen, resultBytes.length);
  dv.setUint8(1, MT.RESULT);
  let p = off;
  dv.setUint32(p, session_id); p += 4;
  dv.setUint32(p, step_id);    p += 4;
  dv.setUint32(p, tile_id);    p += 4;
  dv.setUint8(p, vote_group);  p += 1;
  new Uint8Array(buf, p, 16).set(c128); p += 16;
  new Uint8Array(buf, p, resultBytes.length).set(resultBytes);
  return buf;
}

export function decRESULT(buf) {
  const dv = new DataView(buf);
  const ver = dv.getUint8(0), mt = dv.getUint8(1);
  if (ver !== VERSION || mt !== MT.RESULT) throw new Error("not RESULT");
  const hlen = dv.getUint16(2), blen = dv.getUint32(4);
  let p = 8;
  const session_id = dv.getUint32(p); p += 4;
  const step_id    = dv.getUint32(p); p += 4;
  const tile_id    = dv.getUint32(p); p += 4;
  const vote_group = dv.getUint8(p);  p += 1;
  const c128 = new Uint8Array(buf, p, 16); p += 16;
  const body = new Uint8Array(buf, 8 + hlen, blen);
  return { session_id, step_id, tile_id, vote_group, c128: new Uint8Array(c128), body };
}

// helpers
export function hexOf(bytes) {
  let s = ""; for (let i = 0; i < bytes.length; i++) s += bytes[i].toString(16).padStart(2,"0");
  return s;
}
export function u32ToBytesLE(n) {
  const b = new Uint8Array(4); const dv = new DataView(b.buffer); dv.setUint32(0, n, true); return b;
}
