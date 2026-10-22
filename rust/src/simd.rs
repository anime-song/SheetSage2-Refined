//! Runtime CPU feature detection.  The AVX2 + FMA (+ F16C) kernels are the ones a Zen 2 / Zen 3 Ryzen runs;
//! everything else falls back to portable scalar code.  `SHEETSAGE_NO_SIMD=1` forces the fallback (for tests).

use std::sync::OnceLock;

pub fn avx2() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        if std::env::var_os("SHEETSAGE_NO_SIMD").is_some_and(|v| v != "0") {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}
