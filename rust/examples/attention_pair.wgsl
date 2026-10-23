// A variant of src/gpu/attention.wgsl for `gpu_bench`: two threads per query, each keeping half of the q row and
// half of the output row (32 dims) in registers, so a thread needs about half the registers and twice as many
// waves fit on a SIMD.  The two halves of each score are exchanged through workgroup memory.  128 queries per
// workgroup, keys in blocks of 16.  q, k, v are rows of qkv [frames][3 * 1024] (the scale folded into q); the
// output goes to out [frames][1024].

struct Params {
    frames: u32, _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> qkv: array<vec4<f32>>; // rows of 768
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>; // rows of 256

const QB: u32 = 128u;
const KB: u32 = 16u;
const DV: u32 = 16u; // head dims / 4
const HV: u32 = 8u; // half of them
const HALF: u32 = 129u; // offset of the second halves: KB * HV + 1, so the two halves sit in different banks
const FLUSH: f32 = -40.0; // as the CPU engine: probabilities below e^-40 of the best are zero

var<workgroup> ks: array<vec4<f32>, 258>; // [half][KB][HV]
var<workgroup> vs: array<vec4<f32>, 258>;
var<workgroup> partial: array<vec4<f32>, 1024>; // [thread][4]: its half of the block's 16 scores

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let tid = lid3.x;
    let head = wg.y;
    let h = tid & 1u;
    let row = wg.x * QB + tid / 2u;
    let live = row < p.frames;
    var q: array<vec4<f32>, 8>;
    var o: array<vec4<f32>, 8>;
    for (var d = 0u; d < HV; d++) {
        q[d] = vec4<f32>(0.0);
        if (live) {
            q[d] = qkv[row * 768u + head * DV + h * HV + d];
        }
        o[d] = vec4<f32>(0.0);
    }
    var m = -1e30;
    var l = 0.0;
    // this thread's share of a block: part tid % 16 of key tid / 16, stored at its half
    let lk = tid / DV;
    let ld = tid % DV;
    let slot = (ld / HV) * HALF + lk * HV + ld % HV;
    var nk = vec4<f32>(0.0);
    var nv = vec4<f32>(0.0);
    if (lk < p.frames) {
        nk = qkv[lk * 768u + 256u + head * DV + ld];
        nv = qkv[lk * 768u + 512u + head * DV + ld];
    }
    let mine = h * HALF;
    for (var k0 = 0u; k0 < p.frames; k0 += KB) {
        workgroupBarrier();
        ks[slot] = nk;
        vs[slot] = nv;
        workgroupBarrier();
        let key = k0 + KB + lk;
        nk = vec4<f32>(0.0);
        nv = vec4<f32>(0.0);
        if (key < p.frames) {
            nk = qkv[key * 768u + 256u + head * DV + ld];
            nv = qkv[key * 768u + 512u + head * DV + ld];
        }
        let h0 = scores(&q, mine, 0u);
        let h1 = scores(&q, mine, 4u);
        let h2 = scores(&q, mine, 8u);
        let h3 = scores(&q, mine, 12u);
        partial[tid * 4u] = h0;
        partial[tid * 4u + 1u] = h1;
        partial[tid * 4u + 2u] = h2;
        partial[tid * 4u + 3u] = h3;
        workgroupBarrier();
        let other = (tid ^ 1u) * 4u;
        let at = k0 + vec4<u32>(0u, 1u, 2u, 3u);
        let s0 = masked(h0 + partial[other], at, p.frames);
        let s1 = masked(h1 + partial[other + 1u], at + 4u, p.frames);
        let s2 = masked(h2 + partial[other + 2u], at + 8u, p.frames);
        let s3 = masked(h3 + partial[other + 3u], at + 12u, p.frames);
        let top = max(max(s0, s1), max(s2, s3));
        let best = max(m, max(max(top.x, top.y), max(top.z, top.w)));
        let scale = exp(m - best);
        m = best;
        for (var d = 0u; d < HV; d++) {
            o[d] *= scale;
        }
        let p0 = probabilities(s0, m);
        let p1 = probabilities(s1, m);
        let p2 = probabilities(s2, m);
        let p3 = probabilities(s3, m);
        l = l * scale + dot(p0 + p1 + p2 + p3, vec4<f32>(1.0));
        add_values(&o, p0, mine, 0u);
        add_values(&o, p1, mine, 4u);
        add_values(&o, p2, mine, 8u);
        add_values(&o, p3, mine, 12u);
    }
    if (live) {
        let inv = 1.0 / l;
        for (var d = 0u; d < HV; d++) {
            out[row * 256u + head * DV + h * HV + d] = o[d] * inv;
        }
    }
}

// This thread's halves of the scores of keys c .. c + 3 of the block (its half of the keys starts at `base`).
fn scores(q: ptr<function, array<vec4<f32>, 8>>, base: u32, c: u32) -> vec4<f32> {
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var d = 0u; d < HV; d++) {
        let qd = (*q)[d];
        a0 += qd * ks[base + c * HV + d];
        a1 += qd * ks[base + (c + 1u) * HV + d];
        a2 += qd * ks[base + (c + 2u) * HV + d];
        a3 += qd * ks[base + (c + 3u) * HV + d];
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

fn add_values(o: ptr<function, array<vec4<f32>, 8>>, pk: vec4<f32>, base: u32, c: u32) {
    for (var d = 0u; d < HV; d++) {
        var v = (*o)[d];
        v += pk.x * vs[base + c * HV + d];
        v += pk.y * vs[base + (c + 1u) * HV + d];
        v += pk.z * vs[base + (c + 2u) * HV + d];
        v += pk.w * vs[base + (c + 3u) * HV + d];
        (*o)[d] = v;
    }
}
