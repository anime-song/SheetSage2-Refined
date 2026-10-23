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
        var s: array<f32, 16>;
        var best = m;
        for (var c = 0u; c < KB; c++) {
            var t = 0.0;
            for (var d = 0u; d < DV; d++) {
                t += dot(q[d], ks[c * DV + d]);
            }
            s[c] = select(-1e30, t, k0 + c < p.frames);
            best = max(best, s[c]);
        }
        let scale = exp(m - best);
        m = best;
        l *= scale;
        for (var d = 0u; d < DV; d++) {
            o[d] *= scale;
        }
        for (var c = 0u; c < KB; c++) {
            let x = s[c] - m;
            let pk = select(0.0, exp(x), x >= FLUSH);
            l += pk;
            for (var d = 0u; d < DV; d++) {
                o[d] += pk * vs[c * DV + d];
            }
        }
    }
    if (live) {
        let inv = 1.0 / l;
        for (var d = 0u; d < DV; d++) {
            out[row * 256u + head * DV + d] = o[d] * inv;
        }
    }
}
