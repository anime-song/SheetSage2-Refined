// Flash-style attention of one head, one query per thread and 256 queries per workgroup: the thread keeps its q
// row and its output row in registers, so the keys and values of a block of 16, shared through workgroup memory,
// are read by every thread at the same address (a broadcast) and softmax needs no reductions between threads.  The
// next block is loaded into registers while the current one is used.  q, k, v are rows of qkv [frames][3 * 1024]
// (the scale folded into q); the output goes to out [frames][1024].

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

var<workgroup> ks: array<vec4<f32>, 256>; // [KB][DV]
var<workgroup> vs: array<vec4<f32>, 256>; // [KB][DV]

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
    // this thread's share of a block: part tid % 16 of key tid / 16
    let lk = tid / DV;
    let ld = tid % DV;
    var nk = vec4<f32>(0.0);
    var nv = vec4<f32>(0.0);
    if (lk < p.frames) {
        nk = qkv[lk * 768u + 256u + head * DV + ld];
        nv = qkv[lk * 768u + 512u + head * DV + ld];
    }
    for (var k0 = 0u; k0 < p.frames; k0 += KB) {
        workgroupBarrier();
        ks[tid] = nk;
        vs[tid] = nv;
        workgroupBarrier();
        let key = k0 + KB + lk;
        nk = vec4<f32>(0.0);
        nv = vec4<f32>(0.0);
        if (key < p.frames) {
            nk = qkv[key * 768u + 256u + head * DV + ld];
            nv = qkv[key * 768u + 512u + head * DV + ld];
        }
        let at = k0 + vec4<u32>(0u, 1u, 2u, 3u);
        let s0 = masked(scores(&q, 0u), at, p.frames);
        let s1 = masked(scores(&q, 4u), at + 4u, p.frames);
        let s2 = masked(scores(&q, 8u), at + 8u, p.frames);
        let s3 = masked(scores(&q, 12u), at + 12u, p.frames);
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
        add_values(&o, p0, 0u);
        add_values(&o, p1, 4u);
        add_values(&o, p2, 8u);
        add_values(&o, p3, 12u);
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
fn scores(q: ptr<function, array<vec4<f32>, 16>>, c: u32) -> vec4<f32> {
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var d = 0u; d < DV; d++) {
        let qd = (*q)[d];
        a0 += qd * ks[c * DV + d];
        a1 += qd * ks[(c + 1u) * DV + d];
        a2 += qd * ks[(c + 2u) * DV + d];
        a3 += qd * ks[(c + 3u) * DV + d];
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

fn add_values(o: ptr<function, array<vec4<f32>, 16>>, pk: vec4<f32>, c: u32) {
    for (var d = 0u; d < DV; d++) {
        var v = (*o)[d];
        v += pk.x * vs[c * DV + d];
        v += pk.y * vs[(c + 1u) * DV + d];
        v += pk.z * vs[(c + 2u) * DV + d];
        v += pk.w * vs[(c + 3u) * DV + d];
        (*o)[d] = v;
    }
}
