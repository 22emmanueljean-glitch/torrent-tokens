// MIT – private 4-bit matmul kernel for mobile GPU
// Emmanuel Dessallien 2024

struct Tile {
    scale: f32,
    meta: f32,
    data: array<u32>,   // 2 weights per u32 (lower 4-bit, upper 4-bit)
};

@group(0) @binding(0) var<storage, read> A: Tile;   // activations
@group(0) @binding(1) var<storage, read> W: Tile;   // 4-bit weights
@group(0) @binding(2) var<storage, read_write> C: array<f32>; // output

const TILE_SIZE: u32 = 256u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= TILE_SIZE) { return; }

    var sum: f32 = 0.0;
    for (col in 0u..<TILE_SIZE) {
        let idx = col >> 1;          // 2 weights per u32
        let pair = W.data[idx];
        let w0 = f32(pair & 0x0Fu) * W.scale / 7.0 - W.meta;
        let w1 = f32((pair >> 4u) & 0x0Fu) * W.scale / 7.0 - W.meta;
        let a = A.data[col];         // assume activations are 8-bit unpacked for MVP
        sum += a * (select(w0, w1, col & 1u));
    }
    C[row] = sum;
}