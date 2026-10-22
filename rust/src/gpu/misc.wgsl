// Element-wise passes: GLU, rotary positions, layer-mix accumulation, GRN's per-channel gain.

struct Params {
    n: u32, rows: u32, width: u32, flags: u32, // mix: flags 1 = first state (overwrite)
    weight: f32, _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> aux: array<f32>;  // rope: cos then sin, [frames][32] each; grn: gamma
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

// out[t, c] = x[t, c] * sigmoid(x[t, width + c]), x rows 2 * width wide
@compute @workgroup_size(256)
fn glu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 256u;
    if (i >= p.n) {
        return;
    }
    let t = i / p.width;
    let ch = i % p.width;
    let a = x[t * 2u * p.width + ch];
    let g = x[t * 2u * p.width + p.width + ch];
    out[i] = a / (1.0 + exp(-g));
}

// rotary positions on q and k of qkv rows [frames][3 * 1024], in place: 32 heads (q then k) x 32 pairs per row
@compute @workgroup_size(256)
fn rope(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 256u;
    if (i >= p.n) {
        return;
    }
    let t = i / 1024u;
    let head = (i % 1024u) / 32u;
    let k = i % 32u;
    let at = t * 3072u + head * 64u + k;
    let c = aux[t * 32u + k];
    let s = aux[p.rows * 32u + t * 32u + k];
    let a = out[at];
    let b = out[at + 32u];
    out[at] = a * c - b * s;
    out[at + 32u] = b * c + a * s;
}

// out (+)= weight * x
@compute @workgroup_size(256)
fn mix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 256u;
    if (i >= p.n) {
        return;
    }
    if ((p.flags & 1u) != 0u) {
        out[i] = p.weight * x[i];
    } else {
        out[i] += p.weight * x[i];
    }
}

// GRN, pass 1: out[c] = sum over rows of x[row, c]^2 (Kahan-compensated), one thread per channel
@compute @workgroup_size(256)
fn col_squares(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    if (ch >= p.width) {
        return;
    }
    var s = 0.0;
    var comp = 0.0;
    for (var r = 0u; r < p.rows; r++) {
        let v = x[r * p.width + ch];
        let y = v * v - comp;
        let t = s + y;
        comp = (t - s) - y;
        s = t;
    }
    out[ch] = s;
}

var<workgroup> red: array<f32, 256>;

// GRN, pass 2 (one workgroup): out[width + c] = 1 + gamma[c] * |x_c| / (mean_c |x_c| + 1e-6), from pass 1's sums
@compute @workgroup_size(256)
fn grn_gain(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    var s = 0.0;
    for (var ch = lid; ch < p.width; ch += 256u) {
        s += sqrt(out[ch]);
    }
    red[lid] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (lid < k) {
            red[lid] += red[lid + k];
        }
        workgroupBarrier();
    }
    let mean = red[0] / f32(p.width);
    for (var ch = lid; ch < p.width; ch += 256u) {
        out[p.width + ch] = 1.0 + aux[ch] * (sqrt(out[ch]) / (mean + 1e-6));
    }
}
