export async function loadGPT2Tokenizer(vocabUrl, mergesUrl){
    console.log("🔍 TOKENIZER DEBUG: Loading vocab from:", vocabUrl);
    console.log("🔍 TOKENIZER DEBUG: Loading merges from:", mergesUrl);
    const bust = Date.now();
    const [vocabRes, mergesRes] = await Promise.all([
      fetch(vocabUrl + "?v=" + bust, {cache:"no-store"}), 
      fetch(mergesUrl + "?v=" + bust, {cache:"no-store"})
    ]);
    
    const vocab = await vocabRes.json();
    const mergesTxt = await mergesRes.text();
    const merges = mergesTxt.split("\n").slice(1).filter(Boolean).map(l => l.trim().split(/\s+/));
    const bpeRanks = new Map(merges.map((p,i)=>[p.join(" "), i]));
    const byteEncoder = bytes_to_unicode();
    const byteDecoder = Object.fromEntries(Object.entries(byteEncoder).map(([k,v])=>[v,Number(k)]));
    
    const id2tok = new Array(Object.keys(vocab).length);
    for (const [tok, id] of Object.entries(vocab)) id2tok[id] = tok;
  
    function getPairs(word){
      const pairs = [];
      for (let i=0; i<word.length-1; i++) {
        pairs.push([word[i], word[i+1]]);
      }
      return pairs;
    }
  
    function bpe(token){
      if (token.length === 1) return [token];
      
      let word = token.split('');
      let pairs = getPairs(word);
      
      if (pairs.length === 0) return [token];
      
      while (true) {
        let minPair = null;
        let minRank = Infinity;
        
        for (const pair of pairs) {
          const rank = bpeRanks.get(pair.join(' '));
          if (rank !== undefined && rank < minRank) {
            minRank = rank;
            minPair = pair;
          }
        }
        
        if (minPair === null) break;
        
        const [first, second] = minPair;
        const newWord = [];
        let i = 0;
        
        while (i < word.length) {
          const j = word.indexOf(first, i);
          if (j === -1) {
            newWord.push(...word.slice(i));
            break;
          }
          
          newWord.push(...word.slice(i, j));
          
          if (j < word.length - 1 && word[j+1] === second) {
            newWord.push(first + second);
            i = j + 2;
          } else {
            newWord.push(first);
            i = j + 1;
          }
        }
        
        word = newWord;
        if (word.length === 1) break;
        pairs = getPairs(word);
      }
      
      return word;
    }
  
    function encode(text){
      const bytes = new TextEncoder().encode(text);
      let chars = "";
      for (const b of bytes) chars += byteEncoder[b];
      
      const pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
      const tokens = [];
      
      for (const match of chars.match(pat) || []) {
        const bpeTokens = bpe(match);
        for (const token of bpeTokens) {
          if (vocab[token] !== undefined) {
            tokens.push(vocab[token]);
          }
        }
      }
      
      return tokens;
    }
  
    function decode(tokenIds){
      let text = '';
      for (const id of tokenIds) {
        const token = id2tok[id];
        if (token !== undefined) text += token;
      }
      
      const bytes = [];
      for (const char of text) {
        const byte = byteDecoder[char];
        if (byte !== undefined) bytes.push(byte);
      }
      
      return new TextDecoder().decode(new Uint8Array(bytes));
    }
  
    return { encode, decode };
}
  
function bytes_to_unicode(){
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    
    const cs = bs.slice();
    let n = 0;
    
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) {
        bs.push(b);
        cs.push(256 + n);
        n++;
      }
    }
    
    const result = {};
    for (let i = 0; i < bs.length; i++) {
      result[bs[i]] = String.fromCharCode(cs[i]);
    }
    
    return result;
}
