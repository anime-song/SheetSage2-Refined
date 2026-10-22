//! Runtime CPU feature detection.  The AVX2 + FMA (+ F16C) kernels are the fast path;
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

/// Flush denormal results to zero and read denormal inputs as zero (MXCSR FTZ | DAZ) on the calling thread: the
/// model never needs numbers below 1e-38, and x86 cores slow down 10-50x on them.
pub fn flush_denormals() {
    #[cfg(target_arch = "x86_64")]
    #[allow(deprecated)]
    unsafe {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        _mm_setcsr(_mm_getcsr() | 0x8040);
    }
}

/// The global thread pool (`threads`, default: one per logical CPU), every thread flushing denormals.
pub fn init_threads(threads: Option<usize>) -> Result<(), rayon::ThreadPoolBuildError> {
    flush_denormals();
    let mut builder = rayon::ThreadPoolBuilder::new().start_handler(|_| flush_denormals());
    if let Some(n) = threads {
        builder = builder.num_threads(n);
    }
    builder.build_global()
}
