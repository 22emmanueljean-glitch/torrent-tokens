import { AutoTokenizer } from '@xenova/transformers';

let tokenizer = null;

export async function loadGPT2Tokenizer() {
  if (!tokenizer) {
    console.log("🔍 Loading GPT-2 tokenizer from @xenova/transformers...");
    tokenizer = await AutoTokenizer.from_pretrained('gpt2');
  }
  
  return {
    encode: (text) => {
      const result = tokenizer.encode(text);
      return Array.from(result);
    },
    decode: (tokens) => {
      return tokenizer.decode(tokens);
    }
  };
}
