//! Matrix products for the linear layers and attention.
//!
//! `linear` computes `out[i, j] = act(sum_k a[i, k] * W[j, k] + b[j])` (or adds the product to `out`, a residual),
//! for row-major `a` and weights stored as 16-wide panels.  The rows are split into blocks of `rows()` that run in
//! parallel; inside a block the loop order is BLIS's: for each `kc()` slice of `k` the block's rows are packed into
//! 6-row panels (they stay in L2), and each weight panel is converted to float32 once (it stays in L1) and swept
//! over all the row panels by the 6x16 micro-kernel, 12 AVX2 accumulators.  Weights stored as float16 or int8 are
//! thus converted once per row block, which is cheap next to the multiply-adds each converted value feeds.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::math;
use crate::simd;
use crate::weights::{Linear, PANEL, Panels};

pub const MR: usize = 6;
pub const NR: usize = PANEL;

// Blocking: rows per parallel block (a multiple of MR) and the slice of k per packed panel.  A KC of 256 keeps a
// converted weight panel (16 KiB) in a Zen 3 core's 32 KiB L1 next to the streamed rows.  `SHEETSAGE_ROWS` and
// `SHEETSAGE_KC` override them (`examples/bench.rs --tune` searches for the best pair on a machine).
static ROWS: AtomicUsize = AtomicUsize::new(0);
static KC: AtomicUsize = AtomicUsize::new(0);

fn setting(cell: &AtomicUsize, var: &str, default: usize) -> usize {
    match cell.load(Ordering::Relaxed) {
        0 => {
            let v = std::env::var(var).ok().and_then(|v| v.parse().ok()).filter(|&v: &usize| v > 0).unwrap_or(default);
            cell.store(v, Ordering::Relaxed);
            v
        }
        v => v,
    }
}

pub fn rows() -> usize {
    setting(&ROWS, "SHEETSAGE_ROWS", 96).div_ceil(MR) * MR
}

pub fn kc() -> usize {
    setting(&KC, "SHEETSAGE_KC", 256)
}

pub fn set_blocking(rows: usize, kc: usize) {
    ROWS.store(rows, Ordering::Relaxed);
    KC.store(kc, Ordering::Relaxed);
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Act {
    None,
    Gelu,
}

thread_local! {
    static SCRATCH: RefCell<(Vec<f32>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// `out = act(a W^T + b)` (`accumulate`: `out += a W^T + b`) over `m` rows; `a_scale` multiplies column `k` of `a`
/// (GRN's per-channel gain).  Runs the row blocks in parallel.
#[allow(clippy::too_many_arguments)]
pub fn linear(
    a: &[f32],
    lda: usize,
    m: usize,
    lin: &Linear,
    out: &mut [f32],
    ldo: usize,
    act: Act,
    accumulate: bool,
    a_scale: Option<&[f32]>,
) {
    assert!(!(accumulate && act != Act::None));
    assert!(lda >= lin.k && ldo >= lin.n);
    if m == 0 {
        return;
    }
    assert!(a.len() >= (m - 1) * lda + lin.k && out.len() >= (m - 1) * ldo + lin.n);
    let out = &mut out[..(m - 1) * ldo + lin.n];
    let block_rows = rows();
    out.par_chunks_mut(block_rows * ldo).enumerate().for_each(|(block, out)| {
        let rows = out.len().div_ceil(ldo);
        let a = &a[block * block_rows * lda..];
        linear_block(a, lda, rows, lin, out, ldo, act, accumulate, a_scale);
    });
}

/// `linear` on the calling thread only.
#[allow(clippy::too_many_arguments)]
pub fn linear_block(
    a: &[f32],
    lda: usize,
    m: usize,
    lin: &Linear,
    out: &mut [f32],
    ldo: usize,
    act: Act,
    accumulate: bool,
    a_scale: Option<&[f32]>,
) {
    SCRATCH.with(|s| {
        let (apack, bpanel) = &mut *s.borrow_mut();
        let row_panels = m.div_ceil(MR);
        let block_k = kc();
        apack.resize(row_panels * MR * block_k, 0.0);
        bpanel.resize(block_k * NR, 0.0);
        let k = lin.k;
        let mut tile = [0f32; MR * NR];
        for k0 in (0..k).step_by(block_k) {
            let kc = block_k.min(k - k0);
            let (first, last) = (k0 == 0, k0 + kc == k);
            pack_a(a, lda, m, k0, kc, a_scale, apack);
            for panel in 0..lin.panels() {
                let (b, ldb) = panel_slice(lin, panel, k0, kc, bpanel);
                let n0 = panel * NR;
                let cols = NR.min(lin.n - n0);
                for rp in 0..row_panels {
                    let rows = MR.min(m - rp * MR);
                    kernel(kc, &apack[rp * MR * kc..], 1, MR, b, ldb, &mut tile);
                    let scale = lin.scale.map(|s| &s[n0..n0 + cols]);
                    for r in 0..rows {
                        let o = &mut out[(rp * MR + r) * ldo + n0..][..cols];
                        let t = &tile[r * NR..r * NR + cols];
                        epilogue(o, t, scale, &lin.bias[n0..n0 + cols], first && !accumulate, last, act);
                    }
                }
            }
        }
    });
}

#[inline]
fn epilogue(o: &mut [f32], t: &[f32], scale: Option<&[f32]>, bias: &[f32], overwrite: bool, last: bool, act: Act) {
    for j in 0..o.len() {
        let mut v = t[j];
        if let Some(s) = scale {
            v *= s[j];
        }
        if !overwrite {
            v += o[j];
        }
        if last {
            v += bias[j];
        }
        o[j] = v;
    }
    if last && act == Act::Gelu {
        math::gelu_slice(o);
    }
}

fn pack_a(a: &[f32], lda: usize, m: usize, k0: usize, kc: usize, a_scale: Option<&[f32]>, apack: &mut [f32]) {
    for rp in 0..m.div_ceil(MR) {
        let dst = &mut apack[rp * MR * kc..(rp + 1) * MR * kc];
        for r in 0..MR {
            let i = rp * MR + r;
            if i < m {
                let row = &a[i * lda + k0..i * lda + k0 + kc];
                match a_scale {
                    Some(s) => {
                        for (p, (&v, &g)) in row.iter().zip(&s[k0..k0 + kc]).enumerate() {
                            dst[p * MR + r] = v * g;
                        }
                    }
                    None => {
                        for (p, &v) in row.iter().enumerate() {
                            dst[p * MR + r] = v;
                        }
                    }
                }
            } else {
                for p in 0..kc {
                    dst[p * MR + r] = 0.0;
                }
            }
        }
    }
}

/// Rows `k0 .. k0 + kc` of weight panel `panel` as float32 `[kc][16]`: in place for float32 weights, else converted
/// into `buf`.
fn panel_slice<'a>(lin: &'a Linear, panel: usize, k0: usize, kc: usize, buf: &'a mut [f32]) -> (&'a [f32], usize) {
    let start = (panel * lin.k + k0) * NR;
    match lin.w {
        Panels::F32(w) => (&w[start..start + kc * NR], NR),
        Panels::F16(w) => {
            convert_f16(&w[start..start + kc * NR], &mut buf[..kc * NR]);
            (&buf[..kc * NR], NR)
        }
        Panels::I8(w) => {
            for (d, &s) in buf[..kc * NR].iter_mut().zip(&w[start..start + kc * NR]) {
                *d = s as f32;
            }
            (&buf[..kc * NR], NR)
        }
    }
}

pub fn convert_f16(src: &[u16], dst: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd::avx2() {
        unsafe { convert_f16_avx2(src, dst) };
        return;
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = half::f16::from_bits(s).to_f32();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn convert_f16_avx2(src: &[u16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
        i += 8;
    }
    for j in i..n {
        dst[j] = half::f16::from_bits(src[j]).to_f32();
    }
}

/// `tile[r][c] = sum_p A[r, p] * B[p, c]` for a 6 x 16 tile: `A[r, p] = a[r * rsa + p * csa]`, `B[p, c] = b[p * ldb + c]`.
#[inline]
pub fn kernel(kc: usize, a: &[f32], rsa: usize, csa: usize, b: &[f32], ldb: usize, tile: &mut [f32; MR * NR]) {
    kernel_into(kc, a, rsa, csa, b, ldb, tile, NR, false);
}

/// The 6 x 16 product written to (`add`: added to) `c[r * ldc + j]`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn kernel_into(
    kc: usize,
    a: &[f32],
    rsa: usize,
    csa: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
    add: bool,
) {
    assert!(kc == 0 || (a.len() > (MR - 1) * rsa + (kc - 1) * csa && b.len() >= (kc - 1) * ldb + NR));
    assert!(c.len() >= (MR - 1) * ldc + NR);
    #[cfg(target_arch = "x86_64")]
    if simd::avx2() {
        unsafe { kernel_avx2(kc, a.as_ptr(), rsa, csa, b.as_ptr(), ldb, c.as_mut_ptr(), ldc, add) };
        return;
    }
    for r in 0..MR {
        let mut row = [0.0f32; NR];
        for p in 0..kc {
            let av = a[r * rsa + p * csa];
            for (j, v) in row.iter_mut().enumerate() {
                *v += av * b[p * ldb + j];
            }
        }
        for (j, v) in row.iter().enumerate() {
            let o = &mut c[r * ldc + j];
            *o = if add { *o + v } else { *v };
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel_avx2(
    kc: usize,
    a: *const f32,
    rsa: usize,
    csa: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    add: bool,
) {
    let mut c00 = _mm256_setzero_ps();
    let mut c01 = _mm256_setzero_ps();
    let mut c10 = _mm256_setzero_ps();
    let mut c11 = _mm256_setzero_ps();
    let mut c20 = _mm256_setzero_ps();
    let mut c21 = _mm256_setzero_ps();
    let mut c30 = _mm256_setzero_ps();
    let mut c31 = _mm256_setzero_ps();
    let mut c40 = _mm256_setzero_ps();
    let mut c41 = _mm256_setzero_ps();
    let mut c50 = _mm256_setzero_ps();
    let mut c51 = _mm256_setzero_ps();
    let (mut ap, mut bp) = (a, b);
    for _ in 0..kc {
        let b0 = _mm256_loadu_ps(bp);
        let b1 = _mm256_loadu_ps(bp.add(8));
        let a0 = _mm256_broadcast_ss(&*ap);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        let a1 = _mm256_broadcast_ss(&*ap.add(rsa));
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        let a2 = _mm256_broadcast_ss(&*ap.add(2 * rsa));
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        let a3 = _mm256_broadcast_ss(&*ap.add(3 * rsa));
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        let a4 = _mm256_broadcast_ss(&*ap.add(4 * rsa));
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);
        let a5 = _mm256_broadcast_ss(&*ap.add(5 * rsa));
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
        ap = ap.add(csa);
        bp = bp.add(ldb);
    }
    for (r, (lo, hi)) in
        [(c00, c01), (c10, c11), (c20, c21), (c30, c31), (c40, c41), (c50, c51)].into_iter().enumerate()
    {
        let p = c.add(r * ldc);
        if add {
            _mm256_storeu_ps(p, _mm256_add_ps(_mm256_loadu_ps(p), lo));
            _mm256_storeu_ps(p.add(8), _mm256_add_ps(_mm256_loadu_ps(p.add(8)), hi));
        } else {
            _mm256_storeu_ps(p, lo);
            _mm256_storeu_ps(p.add(8), hi);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leak<T>(v: Vec<T>) -> &'static [T] {
        Box::leak(v.into_boxed_slice())
    }

    fn pack(w: &[f32], n: usize, k: usize) -> Vec<f32> {
        let panels = n.div_ceil(NR);
        let mut out = vec![0.0; panels * k * NR];
        for j in 0..n {
            for p in 0..k {
                out[((j / NR) * k + p) * NR + j % NR] = w[j * k + p];
            }
        }
        out
    }

    #[test]
    fn matches_naive() {
        let (m, n, k) = (37, 45, 1100);
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0).collect();
        let w: Vec<f32> = (0..n * k).map(|i| ((i * 104729) % 97) as f32 / 48.0 - 1.0).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let lin = Linear { n, k, w: Panels::F32(leak(pack(&w, n, k))), scale: None, bias: leak(b.clone()) };
        let mut out = vec![0.0; m * n];
        linear(&a, k, m, &lin, &mut out, n, Act::None, false, None);
        for i in 0..m {
            for j in 0..n {
                let want: f64 = (0..k).map(|p| a[i * k + p] as f64 * w[j * k + p] as f64).sum::<f64>() + b[j] as f64;
                assert!((out[i * n + j] as f64 - want).abs() < 1e-3 * (1.0 + want.abs()), "{i} {j}");
            }
        }
    }
}
