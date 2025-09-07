export class BPETokenizer {
    constructor(){ this.vocab=null; this.merges=null; this.cache=new Map(); this.byteEncoder=null; this.byteDecoder=null; }
  
    async load(url){
      const res = await fetch(url);
      if (!res.ok) throw new Error(`tokenizer fetch failed: ${res.status}`);
      const j = await res.json();
      const model = j.model || j;
      this.vocab = model.vocab || model.token_to_id || {};
      this.merges = model.merges || model.bpe_ranks || [];
      this.byteEncoder = new Map(); this.byteDecoder = new Map();
      for (let i=0;i<256;i++){ this.byteEncoder.set(i, String.fromCharCode(i)); this.byteDecoder.set(String.fromCharCode(i), i); }
      this.ranks = new Map();
      for (let i=0;i<this.merges.length;i++){
        const m = this.merges[i];
        const k = Array.isArray(m) ? m.join(' ') : m;
        this.ranks.set(k, i);
      }
    }
    encode(text){
      if (!this.vocab) throw new Error("tokenizer not loaded");
      const bytes = new TextEncoder().encode(text);
      const tokens = [];
      for (let i=0;i<bytes.length;i++){
        const ch = this.byteEncoder.get(bytes[i]);
        const id = this.vocab[ch];
        tokens.push(id !== undefined ? id : (bytes[i] + 1));
      }
      return tokens;
    }
    decode(ids){
      if (!this.vocab) throw new Error("tokenizer not loaded");
      if (!this.id2tok){ this.id2tok = new Map(Object.entries(this.vocab).map(([k,v])=>[v,k])); }
      const bytes = [];
      for (const id of ids){
        const t = this.id2tok.get(id);
        if (t && t.length === 1) bytes.push(this.byteDecoder.get(t) ?? 32);
        else { const b=((id-1)&0xFF); bytes.push((b>=32 && b<=126)? b : 46); }
      }
      return new TextDecoder().decode(new Uint8Array(bytes));
    }
  }
  