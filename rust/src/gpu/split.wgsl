// Attention split between the GPU and the CPU (`GpuEncoder::set_cpu_heads`): `pack` copies q, k, v of heads first ..
// first + heads from qkv rows [frames][3 * 1024] into the CPU engine's layouts, one after the other in `packed`:
// q [frames + MR][heads * 64] (the MR padding rows zero), keys [heads][padded / NR][64][NR] and values
// [heads][padded][64] (frames past the end zero).  `merge` writes the CPU's output rows [frames][heads * 64] into
// columns first * 64 .. of the attention output [frames][1024].

struct Params {
    frames: u32, first: u32, heads: u32, padded: u32,
    mr: u32, nr: u32, _p0: u32, _p1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 256u;
    let width = p.heads * 64u;
    let q_len = (p.frames + p.mr) * width;
    let kv_len = width * p.padded;
    if (i >= q_len + 2u * kv_len) {
        return;
    }
    var t: u32;
    var col: u32; // column of qkv's rows
    if (i < q_len) {
        t = i / width;
        col = p.first * 64u + i % width;
    } else if (i < q_len + kv_len) {
        let j = i - q_len;
        let h = j / (64u * p.padded);
        let r = j % (64u * p.padded);
        let panel = r / (64u * p.nr);
        let d = (r % (64u * p.nr)) / p.nr;
        t = panel * p.nr + r % p.nr;
        col = 1024u + (p.first + h) * 64u + d;
    } else {
        let j = i - q_len - kv_len;
        let h = j / (64u * p.padded);
        t = (j % (64u * p.padded)) / 64u;
        col = 2048u + (p.first + h) * 64u + j % 64u;
    }
    var v = 0.0;
    if (t < p.frames) {
        v = src[t * 3072u + col];
    }
    dst[i] = v;
}

@compute @workgroup_size(256)
fn merge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 256u;
    let width = p.heads * 64u;
    if (i >= p.frames * width) {
        return;
    }
    let t = i / width;
    dst[t * 1024u + p.first * 64u + i % width] = src[i];
}
