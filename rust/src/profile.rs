//! Wall-clock time per part of the model, printed by `report` when `SHEETSAGE_PROFILE=1`.

use std::sync::Mutex;
use std::time::Instant;

static TIMES: Mutex<Vec<(&'static str, f64)>> = Mutex::new(Vec::new());

fn enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("SHEETSAGE_PROFILE").is_some_and(|v| v != "0"))
}

/// The part of the model running now (for `MemoryWatch`'s peaks per part).
pub static PHASE: Mutex<&'static str> = Mutex::new("");

pub fn time<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    *PHASE.lock().unwrap() = name;
    let start = Instant::now();
    let out = f();
    let elapsed = start.elapsed().as_secs_f64();
    let mut times = TIMES.lock().unwrap();
    match times.iter_mut().find(|(n, _)| *n == name) {
        Some((_, t)) => *t += elapsed,
        None => times.push((name, elapsed)),
    }
    out
}

pub fn report() {
    if enabled() {
        for (name, t) in TIMES.lock().unwrap().drain(..) {
            eprintln!("  {name:16} {t:7.2} s");
        }
    }
}
