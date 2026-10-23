// Flash-style attention of one head, one query per thread and 256 queries per workgroup: the thread keeps its q
// row and its output row in registers and reads the keys and values straight from the buffer, at addresses every
// thread shares (a GPU can load those once per wave, into scalar registers), with no workgroup memory and no
// barriers.  Keys come in blocks of 16; softmax needs no reductions between threads.  q, k, v are rows of qkv
// [frames][3 * 1024] (the scale folded into q); the output goes to out [frames][1024].  On an integrated GPU
// with the CPU held at its base clock this was 20% faster than sharing keys through workgroup memory
// (examples/attention_lds.wgsl), which was 15% faster with the CPU boosting.

struct Params {
    frames: u32, _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> qkv: array<vec4<f32>>; // rows of 768
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>; // rows of 256

const QB: u32 = 256u;
const KB: u32 = 16u;
const DV: u32 = 16u; // head dims / 4
const FLUSH: f32 = -40.0; // as the CPU engine: probabilities below e^-40 of the best are zero


@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let tid = lid3.x;
    let head = wg.y;
    let row = wg.x * QB + tid;
    let live = row < p.frames;
    var q: array<vec4<f32>, 16>;
    var o: array<vec4<f32>, 16>;
    for (var d = 0u; d < DV; d++) {
        q[d] = vec4<f32>(0.0);
        if (live) {
            q[d] = qkv[row * 768u + head * DV + d];
        }
        o[d] = vec4<f32>(0.0);
    }
    var m = -1e30;
    var l = 0.0;
    let base = head * DV;
    for (var k0 = 0u; k0 < p.frames; k0 += KB) {
        let at = k0 + vec4<u32>(0u, 1u, 2u, 3u);
        let s0 = masked(scores(&q, k0, base), at, p.frames);
        let s1 = masked(scores(&q, k0 + 4u, base), at + 4u, p.frames);
        let s2 = masked(scores(&q, k0 + 8u, base), at + 8u, p.frames);
        let s3 = masked(scores(&q, k0 + 12u, base), at + 12u, p.frames);
        let top = max(max(s0, s1), max(s2, s3));
        let best = max(m, max(max(top.x, top.y), max(top.z, top.w)));
        let scale = exp(m - best);
        m = best;
        for (var d = 0u; d < DV; d++) {
            o[d] *= scale;
        }
        let p0 = probabilities(s0, m);
        let p1 = probabilities(s1, m);
        let p2 = probabilities(s2, m);
        let p3 = probabilities(s3, m);
        l = l * scale + dot(p0 + p1 + p2 + p3, vec4<f32>(1.0));
        add_values(&o, p0, k0 + 0u, base);
        add_values(&o, p1, k0 + 4u, base);
        add_values(&o, p2, k0 + 8u, base);
        add_values(&o, p3, k0 + 12u, base);
    }
    if (live) {
        let inv = 1.0 / l;
        for (var d = 0u; d < DV; d++) {
            out[row * 256u + head * DV + d] = o[d] * inv;
        }
    }
}

// The scores of keys c .. c + 3 of the block.  Loops have constant bounds and arrays constant indices once the calls
// are inlined, so everything unrolls into registers.
fn scores(q: ptr<function, array<vec4<f32>, 16>>, c: u32, base: u32) -> vec4<f32> {
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var d = 0u; d < DV; d++) {
        let qd = (*q)[d];
        a0 += qd * qkv[key_row(c) + 256u + base + d];
        a1 += qd * qkv[key_row((c + 1u)) + 256u + base + d];
        a2 += qd * qkv[key_row((c + 2u)) + 256u + base + d];
        a3 += qd * qkv[key_row((c + 3u)) + 256u + base + d];
    }
    let one = vec4<f32>(1.0);
    return vec4<f32>(dot(a0, one), dot(a1, one), dot(a2, one), dot(a3, one));
}

fn masked(s: vec4<f32>, keys: vec4<u32>, frames: u32) -> vec4<f32> {
    return select(vec4<f32>(-1e30), s, keys < vec4<u32>(frames));
}

fn probabilities(s: vec4<f32>, m: f32) -> vec4<f32> {
    let x = s - m;
    return select(vec4<f32>(0.0), exp(x), x >= vec4<f32>(FLUSH));
}

fn add_values(o: ptr<function, array<vec4<f32>, 16>>, pk: vec4<f32>, c: u32, base: u32) {
    for (var d = 0u; d < DV; d++) {
        var v = (*o)[d];
        v += pk.x * qkv[key_row(c) + 512u + base + d];
        v += pk.y * qkv[key_row((c + 1u)) + 512u + base + d];
        v += pk.z * qkv[key_row((c + 2u)) + 512u + base + d];
        v += pk.w * qkv[key_row((c + 3u)) + 512u + base + d];
        (*o)[d] = v;
    }
}

// rows past the end read the last row (their scores are masked)
fn key_row(key: u32) -> u32 {
    return min(key, p.frames - 1u) * 768u;
}
