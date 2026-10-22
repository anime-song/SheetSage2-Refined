// c[m, n] (+)= act(sum_k a[m, k] * scale[k] * W[n, k] + bias[n]), W as float16 pairs in 16-column panels
// [ceil(N / 16)][K][16] (the CPU engine's layout).  A 128 x 128 tile per workgroup of 16 x 16 threads, k in steps
// of 8; each thread keeps an 8 x 8 block (rows ty*4 + {0..3} and 64 + ty*4 + {0..3}, the same for columns), so
// four vec4 reads of workgroup memory feed 64 multiply-adds.

struct Params {
    m: u32, n: u32, n_pad: u32, k: u32,
    lda: u32, ldc: u32, a_off: u32, c_off: u32,
    flags: u32, // 1: GELU, 2: add to c, 4: scale a's columns
    _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> scale: array<f32>;
@group(0) @binding(5) var<storage, read_write> c: array<f32>;

const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 8u;

var<workgroup> As: array<vec4<f32>, 256>; // [BK][BM / 4]
var<workgroup> Bs: array<vec4<f32>, 256>; // [BK][BN / 4]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let m0 = wg.y * BM;
    let n0 = wg.x * BN;
    var acc: array<vec4<f32>, 16>; // [row i of 8][column half]: acc[2 i + h]
    for (var i = 0u; i < 16u; i++) {
        acc[i] = vec4<f32>(0.0);
    }
    let use_scale = (p.flags & 4u) != 0u;
    // a: each thread loads 4 k of one row; w: 4 consecutive columns at one k
    let ar = tid / 2u;
    let ak = (tid % 2u) * 4u;
    let bk = tid / 32u;
    let bn = (tid % 32u) * 4u;
    for (var k0 = 0u; k0 < p.k; k0 += BK) {
        let row = m0 + ar;
        for (var i = 0u; i < 4u; i++) {
            let kk = k0 + ak + i;
            var v = 0.0;
            if (row < p.m && kk < p.k) {
                v = a[p.a_off + row * p.lda + kk];
                if (use_scale) {
                    v *= scale[kk];
                }
            }
            As[(ak + i) * 32u + ar / 4u][ar % 4u] = v;
        }
        let kk = k0 + bk;
        let col = n0 + bn;
        var b = vec4<f32>(0.0);
        if (kk < p.k && col < p.n_pad) {
            let e = ((col / 16u) * p.k + kk) * 16u + col % 16u;
            let lo = unpack2x16float(w[e / 2u]);
            let hi = unpack2x16float(w[e / 2u + 1u]);
            b = vec4<f32>(lo.x, lo.y, hi.x, hi.y);
        }
        Bs[bk * 32u + bn / 4u] = b;
        workgroupBarrier();
        for (var q = 0u; q < BK; q++) {
            let a0 = As[q * 32u + ty];
            let a1 = As[q * 32u + 16u + ty];
            let b0 = Bs[q * 32u + tx];
            let b1 = Bs[q * 32u + 16u + tx];
            acc[0] += a0.x * b0; acc[1] += a0.x * b1;
            acc[2] += a0.y * b0; acc[3] += a0.y * b1;
            acc[4] += a0.z * b0; acc[5] += a0.z * b1;
            acc[6] += a0.w * b0; acc[7] += a0.w * b1;
            acc[8] += a1.x * b0; acc[9] += a1.x * b1;
            acc[10] += a1.y * b0; acc[11] += a1.y * b1;
            acc[12] += a1.z * b0; acc[13] += a1.z * b1;
            acc[14] += a1.w * b0; acc[15] += a1.w * b1;
        }
        workgroupBarrier();
    }
    let gelu_on = (p.flags & 1u) != 0u;
    let add = (p.flags & 2u) != 0u;
    for (var i = 0u; i < 8u; i++) {
        let row = m0 + (i / 4u) * 64u + ty * 4u + i % 4u;
        if (row >= p.m) {
            continue;
        }
        for (var h = 0u; h < 2u; h++) {
            let v4 = acc[2u * i + h];
            for (var j = 0u; j < 4u; j++) {
                let col = n0 + h * 64u + tx * 4u + j;
                if (col >= p.n) {
                    continue;
                }
                var v = v4[j] + bias[col];
                if (gelu_on) {
                    v = gelu(v);
                }
                let at = p.c_off + row * p.ldc + col;
                if (add) {
                    v += c[at];
                }
                c[at] = v;
            }
        }
    }
}
