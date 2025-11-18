// After line that says: const logits=logits_from_hidden(normalized);
// Add this:

// DEBUG: Check logits
const top5 = Array.from(logits)
  .map((val, idx) => [val, idx])
  .sort((a, b) => b[0] - a[0])
  .slice(0, 5);
log(`🔍 Top 5 logits BEFORE softmax: ${top5.map(([val, idx]) => `${idx}:${val.toFixed(2)}`).join(', ')}`);

// After softmax line, add:
const top5_after = Array.from(logits)
  .map((val, idx) => [val, idx])
  .sort((a, b) => b[0] - a[0])
  .slice(0, 5);
log(`🔍 Top 5 probs AFTER softmax: ${top5_after.map(([val, idx]) => `${idx}:${val.toFixed(4)}`).join(', ')}`);
