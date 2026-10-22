// Flash-style attention of one head over one block of 32 queries per workgroup (256 threads): keys and values in
// blocks of 32 through workgroup memory, a running max and sum per query.  q, k, v are rows of qkv
// [frames][3 * 1024] (the scale folded into q); the output goes to out [frames][1024].  Thread (r, j) = (tid / 8,
// tid % 8) scores keys j, j + 8, j + 16, j + 24 for query r and keeps output dims 8j .. 8j + 7.  Rows of q, k and
// the scores are padded (17 vec4, 33 floats) so the threads of a wave read different memory banks.

struct Params {
    frames: u32, _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> qkv: array<vec4<f32>>; // rows of 768
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

const QB: u32 = 32u;
const KB: u32 = 32u;
const DV: u32 = 16u; // head dims / 4
const PAD: u32 = 17u;
const SPAD: u32 = 33u;
const FLUSH: f32 = -40.0; // as the CPU engine: probabilities below e^-40 of the best are zero

var<workgroup> qs: array<vec4<f32>, 544>; // [QB][PAD]
var<workgroup> ks: array<vec4<f32>, 544>; // [KB][PAD]
var<workgroup> vs: array<vec4<f32>, 512>; // [KB][DV]
var<workgroup> ss: array<f32, 1056>; // [QB][SPAD]: scores, then probabilities
var<workgroup> row_max: array<f32, 32>;
var<workgroup> row_sum: array<f32, 32>;
var<workgroup> row_scale: array<f32, 32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let tid = lid3.x;
    let head = wg.y;
    let q0 = wg.x * QB;
    let r = tid / 8u;
    let j = tid % 8u;
    for (var i = tid; i < QB * DV; i += 256u) {
        let row = q0 + i / DV;
        qs[(i / DV) * PAD + i % DV] = select(vec4<f32>(0.0), qkv[row * 768u + head * DV + i % DV], row < p.frames);
    }
    if (tid < QB) {
        row_max[tid] = -1e30;
        row_sum[tid] = 0.0;
    }
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < p.frames; k0 += KB) {
        workgroupBarrier();
        for (var i = tid; i < KB * DV; i += 256u) {
            let key = k0 + i / DV;
            let ok = key < p.frames;
            ks[(i / DV) * PAD + i % DV] = select(vec4<f32>(0.0), qkv[key * 768u + 256u + head * DV + i % DV], ok);
            vs[i] = select(vec4<f32>(0.0), qkv[key * 768u + 512u + head * DV + i % DV], ok);
        }
        workgroupBarrier();
        for (var c = 0u; c < 4u; c++) {
            let key = j + 8u * c;
            var s = 0.0;
            for (var d = 0u; d < DV; d++) {
                s += dot(qs[r * PAD + d], ks[key * PAD + d]);
            }
            ss[r * SPAD + key] = select(-1e30, s, k0 + key < p.frames);
        }
        workgroupBarrier();
        if (tid < QB) {
            var m = row_max[tid];
            for (var c = 0u; c < KB; c++) {
                m = max(m, ss[tid * SPAD + c]);
            }
            row_scale[tid] = exp(row_max[tid] - m);
            row_max[tid] = m;
        }
        workgroupBarrier();
        let m = row_max[r];
        for (var c = 0u; c < 4u; c++) {
            let at = r * SPAD + j + 8u * c;
            let x = ss[at] - m;
            ss[at] = select(0.0, exp(x), x >= FLUSH);
        }
        workgroupBarrier();
        if (tid < QB) {
            var s = 0.0;
            for (var c = 0u; c < KB; c++) {
                s += ss[tid * SPAD + c];
            }
            row_sum[tid] = row_sum[tid] * row_scale[tid] + s;
        }
        let scale = row_scale[r];
        acc0 *= scale;
        acc1 *= scale;
        for (var k = 0u; k < KB; k++) {
            let pk = ss[r * SPAD + k];
            acc0 += pk * vs[k * DV + 2u * j];
            acc1 += pk * vs[k * DV + 2u * j + 1u];
        }
    }
    workgroupBarrier();
    let row = q0 + r;
    if (row < p.frames) {
        let inv = 1.0 / row_sum[r];
        let base = row * 1024u + head * 64u + j * 8u;
        for (var c = 0u; c < 4u; c++) {
            out[base + c] = acc0[c] * inv;
            out[base + 4u + c] = acc1[c] * inv;
        }
    }
}
