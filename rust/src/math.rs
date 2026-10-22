//! Element-wise kernels: GELU (erf), exp, sigmoid, tanh and LayerNorm, AVX2 + FMA with a scalar fallback.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::simd;

/// erf as Eigen's float approximation (a 13/6 rational on [-4, 4]); a few ulp from libm.
#[inline]
pub fn erf(x: f32) -> f32 {
    let x = x.clamp(-4.0, 4.0);
    let x2 = x * x;
    let mut p = x2 * -2.726_142_3e-10 + 2.770_681_4e-08;
    p = x2 * p + -2.101_024e-06;
    p = x2 * p + -5.692_506_4e-05;
    p = x2 * p + -7.349_906_3e-04;
    p = x2 * p + -2.954_6e-3;
    p = x2 * p + -1.609_603_3e-02;
    p *= x;
    let mut q = x2 * -1.456_607_2e-05 + -2.133_740_6e-04;
    q = x2 * q + -1.682_827e-03;
    q = x2 * q + -7.373_329_3e-03;
    q = x2 * q + -1.426_474e-02;
    p / q
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// In-place GELU over a slice.
pub fn gelu_slice(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd::avx2() {
        unsafe { gelu_avx2(x) };
        return;
    }
    for v in x {
        *v = gelu(*v);
    }
}

/// # Safety
/// The CPU must support AVX2 and FMA (`simd::avx2()`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn erf8(x: __m256) -> __m256 {
    let x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(4.0)), _mm256_set1_ps(-4.0));
    let x2 = _mm256_mul_ps(x, x);
    let mut p = _mm256_fmadd_ps(x2, _mm256_set1_ps(-2.726_142_3e-10), _mm256_set1_ps(2.770_681_4e-08));
    p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(-2.101_024e-06));
    p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(-5.692_506_4e-05));
    p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(-7.349_906_3e-04));
    p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(-2.954_6e-3));
    p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(-1.609_603_3e-02));
    p = _mm256_mul_ps(p, x);
    let mut q = _mm256_fmadd_ps(x2, _mm256_set1_ps(-1.456_607_2e-05), _mm256_set1_ps(-2.133_740_6e-04));
    q = _mm256_fmadd_ps(x2, q, _mm256_set1_ps(-1.682_827e-03));
    q = _mm256_fmadd_ps(x2, q, _mm256_set1_ps(-7.373_329_3e-03));
    q = _mm256_fmadd_ps(x2, q, _mm256_set1_ps(-1.426_474e-02));
    _mm256_div_ps(p, q)
}

/// # Safety
/// The CPU must support AVX2 and FMA (`simd::avx2()`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gelu8(x: __m256) -> __m256 {
    let e = erf8(_mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2)));
    _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5), x), _mm256_add_ps(_mm256_set1_ps(1.0), e))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_avx2(x: &mut [f32]) {
    let mut chunks = x.chunks_exact_mut(8);
    for c in &mut chunks {
        let v = _mm256_loadu_ps(c.as_ptr());
        _mm256_storeu_ps(c.as_mut_ptr(), gelu8(v));
    }
    for v in chunks.into_remainder() {
        *v = gelu(*v);
    }
}

/// exp(x) for x <= 0 (softmax), Cephes-style: 2^n * p(r); flushes to 0 below -87.
///
/// # Safety
/// The CPU must support AVX2 and FMA (`simd::avx2()`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn exp8(x: __m256) -> __m256 {
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.0));
    let fx = _mm256_round_ps(
        _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693_359_4), x);
    let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.121_944_4e-4), r);
    let mut y = _mm256_set1_ps(1.987_569_1e-4);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.398_199_9e-3));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.333_452e-3));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.166_579_6e-2));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.666_666_5e-1));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(0.5));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0)));
    let n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    _mm256_mul_ps(y, _mm256_castsi256_ps(n))
}

/// Normalise one row to zero mean and unit variance (biased, as torch's LayerNorm), without scale or shift.
pub fn normalize(x: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / n;
    let inv = (1.0 / (var + eps as f64).sqrt()) as f32;
    let mean = mean as f32;
    for (o, &v) in out.iter_mut().zip(x) {
        *o = (v - mean) * inv;
    }
}

/// LayerNorm of one row with scale and shift.
pub fn layer_norm(x: &[f32], out: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    normalize(x, out, eps);
    for ((o, g), b) in out.iter_mut().zip(gamma).zip(beta) {
        *o = *o * g + b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_reference_values() {
        for (x, want) in [
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (-1.5, -0.966_105_146_475_310_8),
            (0.01, 0.011_283_415_555_849_6),
        ] {
            assert!((erf(x as f32) as f64 - want).abs() < 2e-7, "erf({x})");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vector_kernels_match_scalar() {
        if !simd::avx2() {
            return;
        }
        let xs: Vec<f32> = (0..800).map(|i| (i as f32 - 400.0) / 40.0).collect();
        let mut g = xs.clone();
        gelu_slice(&mut g);
        for (x, g) in xs.iter().zip(&g) {
            assert!((gelu(*x) - g).abs() <= 1e-6 * (1.0 + g.abs()), "gelu({x})");
        }
        for chunk in xs.chunks_exact(8) {
            let mut out = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(out.as_mut_ptr(), exp8(_mm256_loadu_ps(chunk.as_ptr()))) };
            for (x, e) in chunk.iter().zip(&out) {
                if *x <= 0.0 {
                    assert!((x.exp() - e).abs() <= 2e-7 * x.exp(), "exp({x})");
                }
            }
        }
    }
}
