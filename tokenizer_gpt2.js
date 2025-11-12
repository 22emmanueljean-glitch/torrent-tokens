// tokenizer_gpt2.js — minimal GPT-2 BPE tokenizer (runtime loads files)
export async function loadGPT2Tokenizer(vocabUrl, mergesUrl){
    console.log("🔍 TOKENIZER DEBUG: Loading vocab from:", vocabUrl);
    console.log("🔍 TOKENIZER DEBUG: Loading merges from:", mergesUrl);
    const bust = Date.now();
  const [vocabRes, mergesRes] = await Promise.all([
    fetch(vocabUrl + "?v=" + bust, {cache:"no-store"}), 
    fetch(mergesUrl + "?v=" + bust, {cache:"no-store"})
  ]);
    const vocab = await vocabRes.json();                // token -> id
    const mergesTxt = await mergesRes.text();
    const merges = mergesTxt.split("\n").slice(1).filter(Boolean).map(l => l.trim().split(" "));
    const bpeRanks = new Map(merges.map((p,i)=>[p.join(" "), i]));
    const byteEncoder = bytes_to_unicode();
    const byteDecoder = {};
    for (const [k,v] of Object.entries(byteEncoder)) byteDecoder[v] = Number(k);
  
    const id2tok = new Array(Object.keys(vocab).length);
    for (const [tok, id] of Object.entries(vocab)) id2tok[id] = tok;
  
    function getPairs(word){
      const pairs = new Set();
      for (let i=0; i<word.length-1; i++) pairs.add(word[i]+" "+word[i+1]);
      return pairs;
    }
  
    function bpe(token){
      let word = token.split("");
      if (word.length === 1) return token;
      let pairs = getPairs(word);
      while (true){
        let minPair = null, minRank = 1e12;
        for (const p of pairs){
          const r = bpeRanks.get(p);
          if (r !== undefined && r < minRank){ minRank = r; minPair = p; }
        }
        if (minPair === null) break;
        const [a,b] = minPair.split(" ");
        const newWord = [];
        for (let i=0; i<word.length; ){
          const j = word.indexOf(b, i+1);
          if (i < word.length-1 && word[i] === a && word[i+1] === b){
            newWord.push(a+b);
            i += 2;
          } else {
            newWord.push(word[i]);
            i += 1;
          }
        }
        word = newWord;
        if (word.length === 1) break;
        pairs = getPairs(word);
      }
      return word.join(" ");
    }
  
    function encode(text){
      // byte encode
      const bytes = new TextEncoder().encode(text);
      let chars = "";
      for (const b of bytes) chars += byteEncoder[b];
      // split on pattern (rough tokenization)
      const re = /'s|'t|'re|'ve|'m|'ll|'d| ?[^\s\w]+|\s+|[\w]+/g;
      const tokens = [];
      for (const m of chars.matchAll(re)) {
        const part = m[0];
        const bpeOut = bpe(part).split(" ");
        for (const t of bpeOut) tokens.push(vocab[t]);
      }
      return tokens;
    }
  
    function decode(tokenIds){
      let out = "";
      for (const id of tokenIds){
        const tok = id2tok[id];
        if (tok == null) continue;
        for (const ch of tok) {
          const code = Object.keys(byteDecoder).find(k=>k===ch);
          if (code !== undefined) out += String.fromCharCode(byteDecoder[ch]);
          else out += ch;
        }
      }
      // fix common artifacts
      return out.replace(/\s+/g,' ').replace(/Ġ/g,' ');
    }
  
    return { encode, decode };
  }
  
  function bytes_to_unicode(){
    const bs = Array.from({length:256}, (_,i)=>i);
    const cs = bs.slice();
    let n = 0;
    for (let b=0;b<256;b++){
      if (b>=33 && b<=126 || b>=161 && b<=172 || b>=174 && b<=255){
        // printable, keep
      } else {
        cs[b] = 256 + n;
        n++;
      }
    }
    const res = {};
    for (let i=0;i<256;i++) res[i] = String.fromCharCode(cs[i]);
    return res;
  }
  