// Per row: LayerNorm (optionally with scale and shift, then GELU), depthwise convolution + LayerNorm, GLU.
// One workgroup of 256 threads per row; rows are workgroup_id.x + 32768 * workgroup_id.y.

struct Params {
    width: u32, rows: u32, in_stride: u32, out_stride: u32,
    in_off: u32, out_off: u32, flags: u32, taps: u32, // flags: 1 scale + shift, 2 GELU, 4 conv bias
    eps: f32, _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read> taps: array<f32>;   // [taps][width]
@group(0) @binding(5) var<storage, read> conv_bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 256>;

fn reduce_sum(v: f32, lid: u32) -> f32 {
    red[lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            red[lid] += red[lid + s];
        }
        workgroupBarrier();
    }
    let total = red[0];
    workgroupBarrier();
    return total;
}

// the values of row `row` this thread holds (channels lid, lid + 256, ...; width <= 1024), normalised and written
fn finish(vals: ptr<function, array<f32, 4>>, row: u32, lid: u32) {
    let n = p.width;
    var s = 0.0;
    for (var j = 0u; j < 4u; j++) {
        let ch = lid + 256u * j;
        if (ch < n) {
            s += (*vals)[j];
        }
    }
    let mean = reduce_sum(s, lid) / f32(n);
    var d = 0.0;
    for (var j = 0u; j < 4u; j++) {
        let ch = lid + 256u * j;
        if (ch < n) {
            let e = (*vals)[j] - mean;
            d += e * e;
        }
    }
    let inv = 1.0 / sqrt(reduce_sum(d, lid) / f32(n) + p.eps);
    for (var j = 0u; j < 4u; j++) {
        let ch = lid + 256u * j;
        if (ch < n) {
            var v = ((*vals)[j] - mean) * inv;
            if ((p.flags & 1u) != 0u) {
                v = v * gamma[ch] + beta[ch];
            }
            if ((p.flags & 2u) != 0u) {
                v = gelu(v);
            }
            out[p.out_off + row * p.out_stride + ch] = v;
        }
    }
}

@compute @workgroup_size(256)
fn layer_norm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let row = wg.x + 32768u * wg.y;
    if (row >= p.rows) {
        return;
    }
    let lid = lid3.x;
    var vals: array<f32, 4>;
    for (var j = 0u; j < 4u; j++) {
        let ch = lid + 256u * j;
        vals[j] = select(0.0, x[p.in_off + row * p.in_stride + ch], ch < p.width);
    }
    finish(&vals, row, lid);
}

// depthwise convolution over rows (zero padding (taps - 1) / 2 either side), then LayerNorm
@compute @workgroup_size(256)
fn conv_norm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let row = wg.x + 32768u * wg.y;
    if (row >= p.rows) {
        return;
    }
    let lid = lid3.x;
    let half = i32(p.taps / 2u);
    var vals: array<f32, 4>;
    for (var j = 0u; j < 4u; j++) {
        let ch = lid + 256u * j;
        if (ch >= p.width) {
            continue;
        }
        var v = 0.0;
        if ((p.flags & 4u) != 0u) {
            v = conv_bias[ch];
        }
        for (var t = 0u; t < p.taps; t++) {
            let src = i32(row) + i32(t) - half;
            if (src >= 0 && src < i32(p.rows)) {
                v += taps[t * p.width + ch] * x[p.in_off + u32(src) * p.in_stride + ch];
            }
        }
        vals[j] = v;
    }
    finish(&vals, row, lid);
}
